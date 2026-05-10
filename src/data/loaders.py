import tensorflow as tf
import multiprocessing
import numpy as np
import glob
import toml
import os
from src.data import preprocessing as pp
from src.data.masking import mask_dataset
from src.data.record import deserialize

#to_windows, standardize, min_max_scaler, nothing
#unstandardize, shift_times, create_loss_weigths, random_mean



def get_validation(path, validation=0.2, test_folder=None, target_path=''):
    file = os.path.join(target_path, 'train_val_split.toml')
    if os.path.exists(file):
        print('[INFO] Loading train and validation datasets')
        with open(file, 'r') as f:
            output = toml.load(f)
        return output
    
    records_path = glob.glob(os.path.join(path, '*', 'fold_*', '*', '*.record'))

    number_records = len(records_path)
    indices = np.arange(number_records)
    np.random.shuffle(indices)
    val_indices = indices[:int(validation*number_records)]
    train_indices = indices[int(validation*number_records):]
    validation_paths  = np.array(records_path)[val_indices].tolist()
    train_paths       = np.array(records_path)[train_indices].tolist()

    output = {'validation': validation_paths, 'train': train_paths}
    
    if test_folder is not None:
        test_paths = glob.glob(os.path.join(test_folder, '*.record'))
        output['test'] = test_paths
        
    if target_path is not None:
        with open(file, 'w') as f:
            toml.dump(output, f )
        print('[INFO] Train and validation samples saved at {}'.format(file))
    return output

def load_records_v2(record_files):
    """
    Load records files containing serialized light curves.

    Args:
        records_dir (str): records folder
    Returns:
        type: tf.Dataset instance
    """

    records_dir = os.path.dirname(record_files[0])
    raw_dataset = tf.data.TFRecordDataset(record_files)
    raw_dataset = raw_dataset.map(lambda x: deserialize(x, records_dir))
    return raw_dataset
    
def load_records(records_dir):
    """
    Load records files containing serialized light curves.

    Args:
        records_dir (str): records folder
    Returns:
        type: tf.Dataset instance
    """
    record_files = glob.glob(os.path.join(records_dir, '*.record'))
    if len(record_files) == 0:
        record_files = glob.glob(os.path.join(records_dir, '*', '*.record'))
    raw_dataset = tf.data.TFRecordDataset(record_files)

    raw_dataset = raw_dataset.map(lambda x: deserialize(x, records_dir))
    return raw_dataset

def load_records_distributed(records_dir, validation, target_path):
    """
    Load records files containing serialized light curves.

    Args:
        records_dir (str): records folder
    Returns:
        type: tf.Dataset instance
    """

    paths = get_validation(records_dir,
                           validation=validation, 
                           target_path=target_path)
    
    datasets = []
    for sset in paths.keys():
        raw_dataset = tf.data.TFRecordDataset(paths[sset])
        raw_dataset = raw_dataset.map(lambda x: deserialize(x, os.path.join(records_dir, sset)))
        datasets.append(raw_dataset)

    return datasets

def create_generator(list_of_arrays, labels=None, ids=None):
    """
    Create an iterator over a list of numpy-arrays light curves
    Args:
        list_of_arrays (list): list of variable-length numpy arrays.

    Returns:
        type: Iterator of dictonaries
    """

    if ids is None:
        ids = list(range(len(list_of_arrays)))
    if labels is None:
        labels = list(range(len(list_of_arrays)))

    for i, j, k in zip(list_of_arrays, labels, ids):
        yield {'input': i,
               'label':int(j),
               'lcid':str(k),
               'length':int(i.shape[0])}

def load_numpy(samples,
               labels=None,
               ids=None):
    """
    Load light curves in numpy format

    Args:
        samples (list): list of numpy arrays containing vary-lenght light curves
    Returns:
        type: tf.Dataset
    """

    dataset = tf.data.Dataset.from_generator(lambda: create_generator(samples,labels,ids),
                                         output_types= {'input':tf.float32,
                                                        'label':tf.int32,
                                                        'lcid':tf.string,
                                                        'length':tf.int32},
                                         output_shapes={'input':(None,3),
                                                        'label':(),
                                                        'lcid':(),
                                                        'length':()})
    return dataset


def _iter_tar_samples(tar_paths):
    """
    Iterate over light-curve samples stored in WebDataset-style .tar.gz shards.

    Each sample in a shard consists of files sharing a common key prefix:
        <key>.ID.txt
        <key>.time_0.pth
        <key>.brightness_0.pth
        <key>.brightness_err_0.pth
        [<key>.label.cls]   # only present for classification data

    Yields dicts compatible with the existing TF pipeline:
        {'input': (N, 3) float32 array,  # columns: [time, brightness, brightness_err]
         'label': int,
         'lcid': str,
         'length': int}

    Args:
        tar_paths (list[str]): list of .tar.gz shard paths.

    Yields:
        dict: one sample at a time.
    """
    import tarfile
    import io
    import torch  # local import so TF-only environments aren't forced to load torch

    for tar_path in tar_paths:
        with tarfile.open(tar_path, 'r:gz') as tar:
            current_key = None
            buffer = {}

            def flush(buf, key):
                """Assemble a sample dict from a buffer of {ext: bytes}."""
                if not buf or key is None:
                    return None
                # Required components
                try:
                    time_bytes = buf['time_0.pth']
                    bright_bytes = buf['brightness_0.pth']
                    err_bytes = buf['brightness_err_0.pth']
                except KeyError:
                    # Incomplete sample (e.g. partial shard); skip silently
                    return None

                time_t = torch.load(io.BytesIO(time_bytes), map_location='cpu', weights_only=False)
                bright_t = torch.load(io.BytesIO(bright_bytes), map_location='cpu', weights_only=False)
                err_t = torch.load(io.BytesIO(err_bytes), map_location='cpu', weights_only=False)

                time_np = time_t.numpy().astype(np.float32)
                bright_np = bright_t.numpy().astype(np.float32)
                err_np = err_t.numpy().astype(np.float32)

                # Defensive length alignment
                n = min(time_np.shape[0], bright_np.shape[0], err_np.shape[0])
                arr = np.stack([time_np[:n], bright_np[:n], err_np[:n]], axis=-1)

                # ID
                if 'ID.txt' in buf:
                    lcid = buf['ID.txt'].decode('utf-8').strip()
                else:
                    lcid = key

                # Label (only present for classification shards)
                if 'label.cls' in buf:
                    label = int(buf['label.cls'].decode('utf-8').strip())
                else:
                    label = 0  # placeholder; ignored when num_cls is None

                return {
                    'input': arr,
                    'label': label,
                    'lcid': lcid,
                    'length': int(n),
                }

            for member in tar:
                if not member.isfile():
                    continue
                name = member.name
                if '.' not in name:
                    continue
                key, ext = name.split('.', 1)

                if current_key is None:
                    current_key = key

                if key != current_key:
                    sample = flush(buffer, current_key)
                    if sample is not None:
                        yield sample
                    buffer = {}
                    current_key = key

                f = tar.extractfile(member)
                if f is not None:
                    buffer[ext] = f.read()

            # Flush the last sample in the shard
            sample = flush(buffer, current_key)
            if sample is not None:
                yield sample


def load_tar_shards(shard_dir_or_paths):
    """
    Build a tf.data.Dataset that streams light curves from .tar.gz shards.

    Args:
        shard_dir_or_paths: either a directory containing 'lightcurves-*.tar.gz'
            shards, or an explicit list of .tar.gz paths.

    Returns:
        tf.data.Dataset yielding {'input', 'label', 'lcid', 'length'} dicts,
        matching the schema produced by load_numpy / load_records.
    """
    if isinstance(shard_dir_or_paths, str):
        tar_paths = sorted(glob.glob(os.path.join(shard_dir_or_paths, '*.tar.gz')))
        if not tar_paths:
            # Allow one level of nesting (e.g. .../train/fold_0/*.tar.gz)
            tar_paths = sorted(glob.glob(os.path.join(shard_dir_or_paths, '*', '*.tar.gz')))
        if not tar_paths:
            raise FileNotFoundError(
                '[ERROR] No .tar.gz shards found under {}'.format(shard_dir_or_paths))
    else:
        tar_paths = list(shard_dir_or_paths)

    print('[INFO] Found {} tar.gz shard(s)'.format(len(tar_paths)))

    dataset = tf.data.Dataset.from_generator(
        lambda: _iter_tar_samples(tar_paths),
        output_signature={
            'input':  tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            'label':  tf.TensorSpec(shape=(),        dtype=tf.int32),
            'lcid':   tf.TensorSpec(shape=(),        dtype=tf.string),
            'length': tf.TensorSpec(shape=(),        dtype=tf.int32),
        },
    )
    return dataset


def _is_tar_dir(path):
    """Return True if `path` is a directory containing .tar.gz shards."""
    if not isinstance(path, str) or not os.path.isdir(path):
        return False
    if glob.glob(os.path.join(path, '*.tar.gz')):
        return True
    if glob.glob(os.path.join(path, '*', '*.tar.gz')):
        return True
    return False


def format_inp_astromer(batch, 
                        num_cls=None, 
                        nsp_test=False,
                        aversion='base'):
    """
    Buildng ASTROMER input

    Args:
        batch (type): a batch of windows and their properties

    Returns:
        type: A tuple (x, y) tuple where x are the inputs and y the labels
    """

    inputs, outputs = {}, {}
    if aversion=='zero':
        print('[INFO] Zero')
        inputs['input']    =  batch['input_modified']
        inputs['times']    =  tf.slice(batch['input'], [0, 0, 0], [-1, -1, 1])
        
        inputs['mask_in']  = batch['mask_in']
        inputs['mask_out'] = batch['mask_out']
        
        outputs['target']   = tf.slice(batch['input'], [0,0,1], [-1,-1,1])
        outputs['error']    = tf.slice(batch['input'], [0,0,2], [-1,-1,1])
        outputs['mask_out'] = batch['mask_out']
        outputs['lcid']     = batch['lcid']
        outputs['w_error']  = tf.ones_like(inputs['times'], dtype=tf.float32)

    if aversion == 'base':
        input_original  = pp.unstandardize(batch)
        inputs['input'] =  batch['input_modified']
        inputs['times'] =  tf.slice(batch['input'], [0, 0, 0], [-1, -1, 1])
        # zero_time       = tf.reduce_min(inputs['times'], 1) 
        # inputs['times'] = inputs['times'] - tf.expand_dims(zero_time, 1)
        inputs['mask_in'] =  batch['mask_in']

        errors = tf.slice(input_original, [0, 0, 2], [-1,-1, 1])
        outputs['target']   =  tf.slice(batch['input'], [0,0,1], [-1,-1,1])
        outputs['w_error']  =  pp.create_loss_weigths(errors)
        outputs['mask_out'] =  batch['mask_out']
        outputs['lcid']     = batch['lcid']
               
    if num_cls is not None:
        outputs = tf.one_hot(batch['label'], num_cls)        

    return inputs, outputs

def filter_fn(input_dict):
    if tf.less(tf.shape(input_dict['input'])[0], 5):
        return False
    else:
        return True
    
def get_loader(dataset,
               batch_size=None,
               window_size=200,
               probed_frac=.2,
               random_frac=.1,
               same_frac=None,
               sampling=True,
               shuffle=False,
               repeat=1,
               num_cls=None,
               normalize='zero-mean',
               cache=False,
               aversion='base'):

    
    assert isinstance(dataset, (list, str)), '[ERROR] Invalid format'
    assert batch_size is not None, '[ERROR] Undefined batch size'
    if same_frac is None:
        same_frac = random_frac
        
    print('[INFO] Probed: {:.2f} Random: {:.2f} Same: {:.2f}'.format(probed_frac, random_frac, same_frac))
    print('[INFO] Normalization: ', normalize)
    

    if isinstance(dataset, list):
        # Check if list contains tar.gz files
        if dataset and isinstance(dataset[0], str) and dataset[0].endswith('.tar.gz'):
            print('[INFO] Loading from .tar.gz shards (list)')
            dataset = load_tar_shards(dataset)
        else:
            dataset = load_records_v2(dataset)

    if isinstance(dataset, str):
        if _is_tar_dir(dataset):
            print('[INFO] Loading from .tar.gz shards in {}'.format(dataset))
            dataset = load_tar_shards(dataset)
        else:
            dataset = load_records(records_dir=dataset)
    
    if shuffle:
        SHUFFLE_BUFFER = 10000
        dataset = dataset.shuffle(SHUFFLE_BUFFER)

    # REPEAT LIGHT CURVES
    if repeat > 1:
        print('[INFO] Repeating dataset x{} times'.format(repeat))
        dataset = dataset.repeat(repeat)
    
    dataset = dataset.filter(filter_fn)
    
    
    # CREATE WINDOWS
    dataset = pp.to_windows(dataset,
                         window_size=window_size,
                         sampling=sampling)        
    
    if normalize is None:
        dataset = dataset.map(pp.nothing)
        
    if normalize == 'zero-mean':
        dataset = dataset.map(pp.standardize)
    
    if normalize == 'random-mean':
        dataset = dataset.map(pp.random_mean)

    if normalize == 'minmax':
        dataset = dataset.map(pp.min_max_scaler)
        
    
    dataset, shapes = mask_dataset(dataset,
                           msk_frac=probed_frac,
                           rnd_frac=random_frac,
                           same_frac=same_frac,
                           window_size=window_size)
    
    dataset = dataset.padded_batch(batch_size, padded_shapes=shapes)

    # FORMAT INPUT DICTONARY
    dataset = dataset.map(lambda x: format_inp_astromer(x,
                                                num_cls=num_cls,
                                                aversion=aversion),
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    if cache:
        print('[INFO] Cache activated')
        dataset = dataset.cache()

    #PREFETCH BATCHES
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
