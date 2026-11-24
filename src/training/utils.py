'''
DISTRIBUTED TRAINING
'''
import shutil
import tensorflow as tf
import argparse
import math
import toml
import os
import pickle
from tqdm import tqdm

import psutil

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from datetime import datetime

from src.training.scheduler import CustomSchedule
from presentation.pipelines.steps.model_design import build_model, load_pt_model, load_weights_compat
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

def optimizer_state(optimizer):
    """Adam's state in a form that does not depend on the Keras version.

    Both the TF 2.11+ Keras 2 Adam and the Keras 3 Adam keep one first- and one
    second-moment slot per trainable variable, in trainable-variable order, in
    _momentums and _velocities. optimizer.variables is NOT portable: its order
    and length differ between the two (Keras 3 adds a learning-rate variable and
    lists all momentums before all velocities).

    No 'weights' key is written, so code from before this format fails loudly on
    these files instead of assigning values to the wrong variables.
    """
    return {'opt_format': 2,
            'iterations': int(optimizer.iterations.numpy()),
            'm': [v.numpy() for v in optimizer._momentums],
            'v': [v.numpy() for v in optimizer._velocities]}


def restore_optimizer_state(optimizer, model, opt_state):
    """Inverse of optimizer_state(); also reads the older format, the list of
    optimizer.variables saved by TF 2.14: [iteration, m0, v0, m1, v1, ...].
    If the saved state does not fit this model, warns and leaves the optimizer
    fresh (resumed weights, reset moments) rather than restoring it wrongly."""
    # Materialize optimizer slot variables without applying a gradient step
    optimizer.build(model.trainable_variables)
    mom, vel = list(optimizer._momentums), list(optimizer._velocities)
    if 'm' in opt_state:
        it, ms, vs = opt_state['iterations'], opt_state['m'], opt_state['v']
    else:
        w = opt_state['weights']
        if len(w) != 1 + 2 * len(mom):
            print('[WARN] optimizer_state.pkl has {} variables, expected {} for this '
                  'model; optimizer state NOT restored'.format(len(w), 1 + 2 * len(mom)))
            return False
        it, ms, vs = w[0], w[1::2], w[2::2]
    shapes_ok = (len(ms) == len(mom) and len(vs) == len(vel) and
                 all(tuple(a.shape) == tuple(b.shape) for a, b in zip(ms + vs, mom + vel)))
    if not shapes_ok:
        print('[WARN] optimizer_state.pkl does not match this model\'s variables; '
              'optimizer state NOT restored')
        return False
    for var, val in zip(mom + vel, list(ms) + list(vs)):
        var.assign(val)
    optimizer.iterations.assign(int(it))
    return True


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

def get_sec_per_iteration(pbar):
    f_dict = pbar.format_dict
    if f_dict['n'] == 0:
        return 0.
    return f_dict['elapsed'] / f_dict['n']
    

def train(model, optimizer, train_data, validation_data, num_epochs=1000, es_patience=20, test_data=None, project_folder='', resume_from=None):
    train_writer = tf.summary.create_file_writer(os.path.join(project_folder, 'tensorboard', 'train'))
    valid_writer = tf.summary.create_file_writer(os.path.join(project_folder, 'tensorboard', 'validation'))

    sample_batch = next(iter(train_data))



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
    start_epoch = 0
    if resume_from is not None:
        # Copy previous weights to new folder (for resuming where we left off)
        for rel in ['out.weights.h5', os.path.join('best', 'out.weights.h5')]:
            src = os.path.join(resume_from, rel)
            dst = os.path.join(project_folder, rel)
            if os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                print(f'[INFO] Carried forward best checkpoint: {rel}')
        
        # Try to load latest checkpoint first (for resuming where we left off)
        latest_opt_path = os.path.join(resume_from, 'latest', 'optimizer_state.pkl')
        latest_weights_path = os.path.join(resume_from, 'latest', 'out.weights.h5')
        # Fall back to root-level checkpoint (backward compatibility)
        root_opt_path = os.path.join(resume_from, 'optimizer_state.pkl')
        
        if os.path.exists(latest_opt_path):
            opt_path = latest_opt_path
            # Load latest weights (overriding what load_pt_model loaded)
            if os.path.exists(latest_weights_path):
                load_weights_compat(model, latest_weights_path)
                print(f'[INFO] Loaded latest weights from {latest_weights_path}')
        elif os.path.exists(root_opt_path):
            opt_path = root_opt_path
        else:
            opt_path = None
            
        if opt_path is not None:
            with open(opt_path, 'rb') as f:
                opt_state = pickle.load(f)
            opt_restored = restore_optimizer_state(optimizer, model, opt_state)
            start_epoch = opt_state['epoch'] + 1
            step = opt_state['step']
            min_loss = opt_state['min_loss']
            es_count = opt_state.get('es_count', 0)
            best_epoch = opt_state.get('best_epoch', 0)
            best_tr_rmse = opt_state.get('best_tr_rmse', 0.)
            best_tr_rsquare = opt_state.get('best_tr_rsquare', 0.)
            best_vl_rmse = opt_state.get('best_vl_rmse', 0.)
            best_vl_rsquare = opt_state.get('best_vl_rsquare', 0.)
            print(f'[INFO] Resuming after epoch {opt_state["epoch"]} (es_count={es_count}, min_loss={min_loss:.4f}); '
                  f'optimizer state {"restored" if opt_restored else "NOT restored (fresh moments)"}')

    pbar = tqdm(range(start_epoch, num_epochs), total=num_epochs, initial=start_epoch)
    pbar.set_description("Epoch 0 (p={}) - rmse: -/- rsquare: -/-", refresh=True)
    pbar.set_postfix(item=0)    

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
        print('[DEBUG] Sec per iteration: {:.4f}'.format(get_sec_per_iteration(pbar)), flush=True)

        if tf.math.greater(min_loss, vl_rmse):
            min_loss = vl_rmse
            es_count = 0
            best_epoch = epoch
            best_tr_rmse = tr_rmse
            best_tr_rsquare = tr_rsquare
            best_vl_rmse = vl_rmse
            best_vl_rsquare = vl_rsquare
            print('[INFO] New best epoch {:03d} - rmse: {:.4f}/{:.4f} rsquare: {:.4f}/{:.4f}'.format(epoch, tr_rmse, vl_rmse, tr_rsquare, vl_rsquare), flush=True)
            # Save best weights (root level for backward compat + best/ subdirectory)
            model.save_weights(os.path.join(project_folder, 'out.tmp.weights.h5'))
            os.replace(os.path.join(project_folder, 'out.tmp.weights.h5'), os.path.join(project_folder, 'out.weights.h5'))
            best_dir = os.path.join(project_folder, 'best')
            os.makedirs(best_dir, exist_ok=True)
            model.save_weights(os.path.join(best_dir, 'out.tmp.weights.h5'))
            os.replace(os.path.join(best_dir, 'out.tmp.weights.h5'), os.path.join(best_dir, 'out.weights.h5'))
        else:
            es_count = es_count + 1

        if es_count == es_patience:
            print('[INFO] Early Stopping Triggered at epoch {:03d}'.format(epoch))
            print('[INFO] Best epoch: {:03d} - rmse: {:.4f}/{:.4f} rsquare: {:.4f}/{:.4f}'.format(
                best_epoch, best_tr_rmse, best_vl_rmse, best_tr_rsquare, best_vl_rsquare), flush=True)
            break

        # Always save latest checkpoint for resuming
        latest_dir = os.path.join(project_folder, 'latest')
        os.makedirs(latest_dir, exist_ok=True)
        model.save_weights(os.path.join(latest_dir, 'out.tmp.weights.h5'))
        os.replace(os.path.join(latest_dir, 'out.tmp.weights.h5'), os.path.join(latest_dir, 'out.weights.h5'))
        opt_state = {
            **optimizer_state(optimizer),
            'epoch': int(epoch),
            'step': int(step),
            'min_loss': float(min_loss),
            'es_count': int(es_count),
            'best_epoch': int(best_epoch),
            'best_tr_rmse': float(best_tr_rmse),
            'best_tr_rsquare': float(best_tr_rsquare),
            'best_vl_rmse': float(best_vl_rmse),
            'best_vl_rsquare': float(best_vl_rsquare),
        }
        tmp = os.path.join(latest_dir, 'optimizer_state.pkl.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(opt_state, f)
        os.replace(tmp, os.path.join(latest_dir, 'optimizer_state.pkl'))
        
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
