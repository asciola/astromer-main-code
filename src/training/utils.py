'''
DISTRIBUTED TRAINING
'''
import tensorflow as tf
import argparse
import math
import toml
import os
from tqdm import tqdm

import psutil

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from datetime import datetime

from src.training.scheduler import CustomSchedule
from presentation.pipelines.steps.model_design import build_model, load_pt_model
from presentation.pipelines.steps.load_data import build_loader
from presentation.pipelines.steps.metrics import evaluate_ft

from src.losses.rmse import custom_rmse
from src.metrics import custom_r2

def replace_config(source, target):
    for key in ['data', 'no_cache', 'exp_name', 'checkpoint', 
                'gpu', 'lr', 'bs', 'patience', 'num_epochs', 'scheduler']:
        target[key] = source[key]
    return target

def tensorboard_log(name, value, writer, step=0):
	with writer.as_default():
		tf.summary.scalar(name, value, step=step)

@tf.function()
def train_step(model, inputs, optimizer):
    x, y = inputs
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        
        rmse = custom_rmse(y_true=y['target'],
                            y_pred=y_pred,
                            mask=y['mask_out'],
                            root=True if model.loss_format == 'rmse' else False)
                    
        r2_value = custom_r2(y_true=y['target'], 
                            y_pred=y_pred, 
                            mask=y['mask_out'])
        loss = rmse

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return {'loss':loss, 'rmse': rmse, 'rsquare':r2_value}

@tf.function()
def test_step(model, inputs):
    x, y = inputs

    y_pred = model(x, training=False)
    rmse = custom_rmse(y_true=y['target'],
                       y_pred=y_pred,
                       mask=y['mask_out'],
                       root=True if model.loss_format == 'rmse' else False)
                
    r2_value = custom_r2(y_true=y['target'], 
                        y_pred=y_pred, 
                        mask=y['mask_out'])
    loss = rmse
    return {'loss':loss, 'rmse': rmse, 'rsquare':r2_value}

@tf.function
def distributed_train_step(model, batch, optimizer, strategy):
    per_replica_losses = strategy.run(train_step, args=(model, batch, optimizer))
    return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses,
                            axis=None)

@tf.function
def distributed_test_step(model, batch, strategy):
    per_replica_losses = strategy.run(test_step, args=(model, batch))
    return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses,
                            axis=None)

# Initialize NVML for GPU monitoring

def log_system_metrics(writer, step, epoch=None, batch=None):
    """Logs CPU, RAM, GPU memory, and GPU utilization to TensorBoard."""
    cpu_percent = psutil.cpu_percent()
    ram = psutil.virtual_memory()

    with writer.as_default():
        tf.summary.scalar('CPU_percent', cpu_percent, step=step)
        tf.summary.scalar('RAM_used_GB', ram.used / 1e9, step=step)
        if epoch is not None:
            tf.summary.scalar('epoch', epoch, step=step)
        if batch is not None:
            tf.summary.scalar('batch', batch, step=step)
    
def check_attention_health(model, sample_batch):
    """
    Performs a forward pass on one batch and prints attention stats.
    Add this to utils.py.
    """
    x, _ = sample_batch
    
    # Attempt to find the first AttentionBlock in your model hierarchy
    target_block = None
    for submodule in model.submodules:
        if "AttentionBlock" in str(type(submodule)):
            target_block = submodule
            break
            
    if target_block is None:
        print("\n[DEBUG] Could not find an AttentionBlock to monitor.")
        return

    # Call the block directly with return_weights=True
    # Note: This is an approximation as it uses the raw input 'x' 
    # rather than the processed activations from previous layers, 
    # but it is usually enough to see if the weights are uniform.
    try:
        _, att_w, _, _, _ = target_block(x, training=False, return_weights=True)
        
        std_weight = tf.math.reduce_std(att_w).numpy()
        max_weight = tf.reduce_max(att_w).numpy()
        
        print(f"\n[ATTENTION MONITOR] Epoch Step")
        print(f"  - Weight Std Dev: {std_weight:.8f}")
        print(f"  - Max Weight:     {max_weight:.8f}")
        
        if std_weight < 1e-5:
            print("  - STATUS: CRITICAL (Uniform Attention detected)")
        else:
            print("  - STATUS: OK (Model is differentiating tokens)")
    except Exception as e:
        print(f"\n[DEBUG] Monitor failed: {e}")

def train(model, optimizer, train_data, validation_data, num_epochs=1000, es_patience=20, test_data=None, project_folder=''):
    train_writer = tf.summary.create_file_writer(os.path.join(project_folder, 'tensorboard', 'train'))
    valid_writer = tf.summary.create_file_writer(os.path.join(project_folder, 'tensorboard', 'validation'))

    pbar  = tqdm(range(num_epochs), total=num_epochs)
    pbar.set_description("Epoch 0 (p={}) - rmse: -/- rsquare: -/-", refresh=True)
    pbar.set_postfix(item=0)    

    # ========= Training Loop ==================================
    es_count = 0
    min_loss = 1e9
    step = 0
    
    for epoch in pbar:
        if epoch % 5 == 0:  # Check every 5 epochs
            # Get a single batch for monitoring
            sample_batch = next(iter(train_data))
            check_attention_health(model, sample_batch)
        else:
            print("[INFO] Skipping attention check for epoch {}".format(epoch))

        pbar.set_postfix(item1=epoch)
        epoch_tr_rmse    = []
        epoch_tr_rsquare = []
        epoch_tr_loss    = []
        epoch_vl_rmse    = []
        epoch_vl_rsquare = []
        epoch_vl_loss    = []

        for numbatch, batch in enumerate(train_data):
            pbar.set_postfix(item=numbatch)
            metrics = train_step(model, batch, optimizer)
            epoch_tr_rmse.append(metrics['rmse'])
            epoch_tr_rsquare.append(metrics['rsquare'])
            epoch_tr_loss.append(metrics['loss'])
            log_system_metrics(train_writer, step, epoch=epoch, batch=numbatch)
            step += 1

        # Clear caches before test_steps
        if hasattr(model, 'encoder'):
            model.encoder.reset_caches()

        for batch in validation_data:
            metrics = test_step(model, batch)
            epoch_vl_rmse.append(metrics['rmse'])
            epoch_vl_rsquare.append(metrics['rsquare'])
            epoch_vl_loss.append(metrics['loss'])

        tr_rmse    = tf.reduce_mean(epoch_tr_rmse)
        tr_rsquare = tf.reduce_mean(epoch_tr_rsquare)
        vl_rmse    = tf.reduce_mean(epoch_vl_rmse)
        vl_rsquare = tf.reduce_mean(epoch_vl_rsquare)
        tr_loss    = tf.reduce_mean(epoch_tr_loss)
        vl_loss    = tf.reduce_mean(epoch_vl_loss)

        tensorboard_log('loss', tr_loss, train_writer, step=epoch)
        tensorboard_log('loss', vl_loss, valid_writer, step=epoch)
        
        tensorboard_log('rmse', tr_rmse, train_writer, step=epoch)
        tensorboard_log('rmse', vl_rmse, valid_writer, step=epoch)
        
        tensorboard_log('rsquare', tr_rsquare, train_writer, step=epoch)
        tensorboard_log('rsquare', vl_rsquare, valid_writer, step=epoch)
        
        if tf.math.greater(min_loss, vl_rmse):
            min_loss = vl_rmse
            es_count = 0
            model.save_weights(os.path.join(project_folder, 'out.weights.h5'))
        else:
            es_count = es_count + 1

        if es_count == es_patience:
            print('[INFO] Early Stopping Triggered at epoch {:03d}'.format(epoch))
            break
        
        pbar.set_description("Epoch {} (p={}) - rmse: {:.3f}/{:.3f} rsquare: {:.3f}/{:.3f}".format(epoch, 
                                                                                            es_count,
                                                                                            tr_rmse,
                                                                                            vl_rmse,
                                                                                            tr_rsquare,
                                                                                            vl_rsquare))


    print('[INFO] Testing...')
    model.compile(optimizer=optimizer)
    if test_data is not None:
        evaluate_ft(model, test_data)
    return model
