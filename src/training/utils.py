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
    
def check_attention_health(model, sample_batch, pbar=None):
    """
    Performs a forward pass on one batch and prints attention stats.
    Using pbar.write ensures it doesn't get swallowed by the progress bar.
    """
    x, _ = sample_batch
    
    # Improved search: look through all layers and sub-layers
    target_block = None
    for layer in model.layers:
        # Check if the layer is an AttentionBlock (but NOT HeadAttentionMulti)
        layer_type = str(type(layer))
        if "AttentionBlock" in layer_type and "HeadAttentionMulti" not in layer_type:
            target_block = layer
            break
        # Support for nested models (like an Encoder containing blocks)
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                sublayer_type = str(type(sublayer))
                if "AttentionBlock" in sublayer_type and "HeadAttentionMulti" not in sublayer_type:
                    target_block = sublayer
                    break
        if target_block: break

    def log_func(msg):
        print(msg, flush=True)  # Force immediate flush
        if pbar:
            pbar.write(msg)

    if target_block is None:
        log_func("[DEBUG] Could not find an AttentionBlock to monitor.")
        return

    try:
        # Extract just the input tensor if x is a dict
        input_x = x['input'] if isinstance(x, dict) else x
        # We use training=False to avoid dropout/noise during the check
        # and return_weights=True as defined in attblock.py
        _, att_w, _, _, _ = target_block(input_x, training=False, return_weights=True)
        
        std_weight = tf.math.reduce_std(att_w).numpy()
        max_weight = tf.reduce_max(att_w).numpy()
        
        log_func(f"\n--- ATTENTION HEALTH (Epoch Step) ---")
        log_func(f"  - Weight Std Dev: {std_weight:.8f}")
        log_func(f"  - Max Weight:     {max_weight:.8f}")
        
        if std_weight < 1e-5:
            log_func("  - STATUS: CRITICAL (Weights are uniform/averaging)")
        else:
            log_func("  - STATUS: OK (Model is learning token variance)")
        log_func("--------------------------------------\n")
    except Exception as e:
        log_func(f"[DEBUG] Health monitor failed: {e}")

def train(model, optimizer, train_data, validation_data, num_epochs=1000, es_patience=20, test_data=None, project_folder=''):
    train_writer = tf.summary.create_file_writer(os.path.join(project_folder, 'tensorboard', 'train'))
    valid_writer = tf.summary.create_file_writer(os.path.join(project_folder, 'tensorboard', 'validation'))

    sample_batch = next(iter(train_data))

    pbar  = tqdm(range(num_epochs), total=num_epochs)
    pbar.set_description("Epoch 0 (p={}) - rmse: -/- rsquare: -/-", refresh=True)
    pbar.set_postfix(item=0)    

    # ========= Training Loop ==================================
    es_count = 0
    min_loss = 1e9
    step = 0
    
    # Store steps per epoch once determined
    steps_per_epoch = None
    
    for epoch in pbar:
        if epoch % 5 == 0:  # Check every 5 epochs
            check_attention_health(model, sample_batch, pbar)
        else:
            # Use pbar.write to avoid standard print buffering issues
            print(f"[INFO] Training Epoch {epoch}...", flush=True)
            pbar.write(f"[INFO] Training Epoch {epoch}...")
        
        pbar.set_postfix(item1=epoch)
        epoch_tr_rmse    = []
        epoch_tr_rsquare = []
        epoch_tr_loss    = []
        epoch_vl_rmse    = []
        epoch_vl_rsquare = []
        epoch_vl_loss    = []

        for numbatch, batch in enumerate(train_data):
            if steps_per_epoch is not None:
                pbar.set_postfix(item="{}/{}".format(numbatch, steps_per_epoch))
            else:
                pbar.set_postfix(item="{}".format(numbatch))

            metrics = train_step(model, batch, optimizer)
            epoch_tr_rmse.append(metrics['rmse'])
            epoch_tr_rsquare.append(metrics['rsquare'])
            epoch_tr_loss.append(metrics['loss'])
            log_system_metrics(train_writer, step, epoch=epoch, batch=numbatch)
            step += 1
            
        # Set steps_per_epoch after the first epoch
        if steps_per_epoch is None:
            steps_per_epoch = numbatch + 1

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
