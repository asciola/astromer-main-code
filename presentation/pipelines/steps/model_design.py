import tensorflow as tf 
import toml 
import os
import tempfile

import h5py
import keras

from src.models.astromer_0 import build_input as build_input_zero
from src.models.astromer_0 import get_ASTROMER as get_Zero
from src.models.astromer_1 import build_input as build_input_base
from src.models.astromer_1 import get_ASTROMER as get_Base

from tensorflow.keras import layers
from src.layers.input import GammaWeight
from src.layers.pooling import MaskedMean, Stack
from src.layers.encoders import EncoderByLayer

def build_model(params, return_weights=False):
    if not 'mask_format' in params.keys():
        params['mask_format'] = None
        
    if 'temperature' not in params.keys():
        params['temperature'] = 0.

    if not 'no_msk_token' in params.keys():
        params['no_msk_token'] = False

    # Configs written before the [MSK] token fix lack this key, and those models
    # were trained without the token (see get_ASTROMER in src/models/astromer_1.py).
    if not 'apply_msk_token' in params.keys():
        params['apply_msk_token'] = False

    if not 'latent_dim' in params.keys():
        params['latent_dim'] = None

    if not 'linear_attention' in params.keys():
        params['linear_attention'] = False
    
    
    if params['arch'] == 'zero':
        model = get_Zero(num_layers=params['num_layers'],
                           d_model=params['head_dim']*params['num_heads'],
                           num_heads=params['num_heads'],
                           dff=params['mixer'],
                           base=params['pe_base'],
                           rate=params['dropout'],
                           use_leak=False,
                           maxlen=params['window_size'],
                           m_alpha=params['m_alpha'],
                           mask_format=params['mask_format'],
                           return_weights=return_weights,
                           loss_format=params['loss_format'],
                           correct_loss=params['correct_loss'],
                           temperature=params['temperature'])

    if params['arch'] == 'base':
        model = get_Base(num_layers=params['num_layers'],
                         num_heads=params['num_heads'],
                         head_dim=params['head_dim'],
                         mixer_size=params['mixer'],
                         dropout=params['dropout'],
                         pe_base=params['pe_base'],
                         pe_dim=params['pe_dim'],
                         pe_c=params['pe_exp'],
                         window_size=params['window_size'],
                         m_alpha=params['m_alpha'],
                         mask_format=params['mask_format'],
                         use_leak=params['use_leak'],
                         loss_format=params['loss_format'],
                         correct_loss=params['correct_loss'],
                         trainable_mask=not params['no_msk_token'],
                         temperature=params['temperature'],
                         use_cache=params['use_kv_cache'],
                         latent_dim=params['latent_dim'],
                         linear_attention=params['linear_attention'],
                         apply_msk_token=params['apply_msk_token']
                         )

    return model

def load_pt_model(pt_path, optimizer=None):
    config_file = os.path.join(pt_path, 'config.toml')
    with open(config_file, 'r') as file:
        pt_config = toml.load(file)

    model = build_model(pt_config)
    weights_path = os.path.join(pt_path, 'out.weights.h5')
    if optimizer is not None:
        model.compile(optimizer=optimizer)
    load_weights_compat(model, weights_path)
    return model, pt_config


KERAS2_GROUP = '_layer_checkpoint_dependencies'


def _keras2_paths(path):
    """True if a .weights.h5 file names its groups the Keras 2 way."""
    found = []
    with h5py.File(path, 'r') as f:
        f.visit(lambda name: found.append(True) if KERAS2_GROUP in name.split('/') else None)
    return bool(found)


def convert_keras2_weights(src, dst):
    """Copy a .weights.h5 written by TF 2.14 (Keras 2), renaming every
    '_layer_checkpoint_dependencies' group to 'layers', the name Keras 3 uses.
    That is the only difference Keras 3 does not handle itself for these models:
    it translates the name for top-level layers but not for nested ones."""
    def rename(name):
        return '/'.join('layers' if p == KERAS2_GROUP else p for p in name.split('/'))

    with h5py.File(src, 'r') as a, h5py.File(dst, 'w') as b:
        for k, v in a.attrs.items():
            b.attrs[k] = v

        def copy(name, obj):
            if isinstance(obj, h5py.Group):
                g = b.require_group(rename(name))
            else:
                g = b.create_dataset(rename(name), data=obj[()])
            for k, v in obj.attrs.items():
                g.attrs[k] = v
        a.visititems(copy)


def load_weights_compat(model, path):
    """model.load_weights(path), also for Keras 2 checkpoints under Keras 3.

    The file on disk is never modified: an old checkpoint is converted into a
    temporary copy, which is loaded and deleted."""
    if int(keras.__version__.split('.')[0]) < 3 or not _keras2_paths(path):
        model.load_weights(path)
        return
    with tempfile.TemporaryDirectory() as tmp:
        converted = os.path.join(tmp, 'converted.weights.h5')
        convert_keras2_weights(path, converted)
        model.load_weights(converted)


def get_avg_clf(inputs, mask, num_cls):
    x = MaskedMean(axis=1)([inputs, mask])
    x = layers.LayerNormalization(name='layer_norm')(x)
    y_pred = layers.Dense(num_cls, name='output_layer')(x)
    return y_pred
    
def get_avg_mlp(inputs, mask, num_cls):
    x = MaskedMean(axis=1)([inputs, mask])

    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.LayerNormalization(name='layer_norm')(x)
    y_pred = layers.Dense(num_cls, name='output_layer')(x)
    return y_pred

def get_avg_mlp_dp(inputs, mask, num_cls):
    x = MaskedMean(axis=1)([inputs, mask])

    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.LayerNormalization(name='layer_norm')(x)
    y_pred = layers.Dense(num_cls, name='output_layer')(x)
    return y_pred

def get_mlp_avg(inputs, mask, num_cls):
    x = layers.TimeDistributed(layers.Dense(1024, activation='relu'))(inputs)
    x = layers.TimeDistributed(layers.Dense(512, activation='relu'))(x)
    x = layers.TimeDistributed(layers.Dense(256, activation='relu'))(x)
    x = layers.LayerNormalization(name='layer_norm')(x)
    y_pred = layers.TimeDistributed(layers.Dense(num_cls, name='output_layer'))(x)
    y_pred = MaskedMean(axis=1)([y_pred, mask])
    return y_pred

def get_linear(inputs, mask, num_cls):
    y_pred = layers.TimeDistributed(layers.Dense(num_cls, name='output_layer'))(inputs)
    y_pred = MaskedMean(axis=1)([y_pred, mask])
    return y_pred

def get_skip_avg_mlp(inputs, mask, num_cls):

    x = Stack(axis=0)(inputs)                 # (layers, batch, steps, dim)
    x = MaskedMean(axis=2)([x, mask])         # (layers, batch, dim)

    x = GammaWeight(name='gamma_weight')(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.LayerNormalization(name='layer_norm')(x)
    y_pred = layers.Dense(num_cls, name='output_layer')(x)

    return y_pred

def build_classifier(astromer, params, astromer_trainable, num_cls=None, arch='avg_mlp', verbose=1):
    # Build classifier
    # EncoderByLayer, not encoder(..., z_by_layer=True): see its docstring.
    if params['arch'] == 'zero':
        inp_placeholder = build_input_zero(params['window_size'])
        encoder = astromer.get_layer('encoder')
        encoder.trainable = astromer_trainable
        embedding = EncoderByLayer(encoder, name='encoder_by_layer')(inp_placeholder)

    if params['arch'] == 'base' or params['arch'] == 'normal':
        inp_placeholder = build_input_base(params['window_size'])
        encoder = astromer.get_layer('encoder')
        encoder.trainable = astromer_trainable
        # skip_avg_mlp also averages the input embedding, prepended as element 0
        embedding = EncoderByLayer(encoder,
                                   with_input_embedding=(arch == 'skip_avg_mlp'),
                                   name='encoder_by_layer')(inp_placeholder)
        
        
    mask = 1.- inp_placeholder['mask_in']
    
    if verbose == 1:
        print('[INFO] Using {} clf architecture with {}'.format(arch, params['arch']))
    
    if arch == 'avg_clf':
        output = get_avg_clf(embedding[-1], mask, num_cls)
        
    if arch == 'mlp_avg':
        output = get_mlp_avg(embedding[-1], mask, num_cls)

    if arch == 'avg_mlp_dp':
        output = get_avg_mlp_dp(embedding[-1], mask, num_cls)
    
    if arch == 'avg_mlp':
        output = get_avg_mlp(embedding[-1], mask, num_cls)

    if arch == 'linear_att':
        output = get_linear(embedding[-1], mask, num_cls)

    if arch == 'skip_avg_mlp':
        output = get_skip_avg_mlp(embedding, mask, num_cls)

    clf = CustomModel(inputs=inp_placeholder, 
                      outputs=output, 
                      name=arch)

    return clf


class CustomModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs, training=None):
        """Override call to cast output to float32 for mixed precision"""
        output = super().call(inputs, training=training)
        return tf.cast(output, tf.float32)
    
    def predict_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        
        return {'y_pred': y_pred, 
                'y_true': y}