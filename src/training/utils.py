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

        grad_norms = [tf.norm(g) for g in gradients if g is not None]
        max_grad = tf.reduce_max(grad_norms) if len(grad_norms) > 0 else tf.constant(0.0)
        mean_grad = tf.reduce_mean(grad_norms) if len(grad_norms) > 0 else tf.constant(0.0)
        min_grad = tf.reduce_min(grad_norms) if len(grad_norms) > 0 else tf.constant(0.0)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return {'loss':loss, 'rmse': rmse, 'rsquare':r2_value, 'max_grad':max_grad, 'mean_grad':mean_grad, 'min_grad':min_grad}

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
    best_epoch = 0
    best_tr_rmse = 0.
    best_tr_rsquare = 0.
    best_vl_rmse = 0.
    best_vl_rsquare = 0.

    # Store steps per epoch once determined
    steps_per_epoch = None
    
    for epoch in pbar:
        pbar.set_postfix(item1=epoch)
        epoch_tr_rmse    = []
        epoch_tr_rsquare = []
        epoch_tr_loss    = []
        epoch_tr_max_grad = []
        epoch_tr_mean_grad = []
        epoch_tr_min_grad = []
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
            epoch_tr_max_grad.append(metrics['max_grad'])
            epoch_tr_mean_grad.append(metrics['mean_grad'])
            epoch_tr_min_grad.append(metrics['min_grad'])
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
        tr_max_grad = tf.reduce_mean(epoch_tr_max_grad)
        tr_mean_grad = tf.reduce_mean(epoch_tr_mean_grad)
        tr_min_grad = tf.reduce_mean(epoch_tr_min_grad)

        tensorboard_log('loss', tr_loss, train_writer, step=epoch)
        tensorboard_log('loss', vl_loss, valid_writer, step=epoch)
        
        tensorboard_log('rmse', tr_rmse, train_writer, step=epoch)
        tensorboard_log('rmse', vl_rmse, valid_writer, step=epoch)
        
        tensorboard_log('rsquare', tr_rsquare, train_writer, step=epoch)
        tensorboard_log('rsquare', vl_rsquare, valid_writer, step=epoch)
        
        tensorboard_log('gradient/max', tr_max_grad, train_writer, step=epoch)
        tensorboard_log('gradient/mean', tr_mean_grad, train_writer, step=epoch)
        tensorboard_log('gradient/min', tr_min_grad, train_writer, step=epoch)
        
        print('[DEBUG] Epoch {} Gradient stats: max={:.4e}, mean={:.4e}, min={:.4e}'.format(epoch, tr_max_grad, tr_mean_grad, tr_min_grad), flush=True)
        if tf.math.greater(min_loss, vl_rmse):
            min_loss = vl_rmse
            es_count = 0
            best_epoch = epoch
            best_tr_rmse = tr_rmse
            best_tr_rsquare = tr_rsquare
            best_vl_rmse = vl_rmse
            best_vl_rsquare = vl_rsquare
            print('[INFO] New best epoch {} - rmse: {:.3f}/{:.3f} rsquare: {:.3f}/{:.3f}'.format(epoch, tr_rmse, vl_rmse, tr_rsquare, vl_rsquare), flush=True)
            model.save_weights(os.path.join(project_folder, 'out.weights.h5'))
        else:
            es_count = es_count + 1

        if es_count == es_patience:
            print('[INFO] Early Stopping Triggered at epoch {:03d}'.format(epoch))
            print('[INFO] Best epoch: {:03d} - rmse: {:.4f}/{:.4f} rsquare: {:.4f}/{:.4f}'.format(
                best_epoch, best_tr_rmse, best_vl_rmse, best_tr_rsquare, best_vl_rsquare), flush=True)
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
