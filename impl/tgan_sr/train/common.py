# based on DeepLTL: https://github.com/reactive-systems/deepltl

from argparse import ArgumentParser
import os.path as path
import sys
import json
import random

import tensorflow as tf
import numpy as np



def argparser():
    parser = ArgumentParser()
    # Meta
    parser.add_argument('--run-name', default='default', help='name of this run, to better find produced data later')
    parser.add_argument('--job-dir', default='runs', help='general job directory to save produced data into')
    parser.add_argument('--data-dir', default='datasets', help='directory of datasets')
    parser.add_argument('--ds-name', default=None, help='Name of the dataset to use')
    do_test = parser.add_mutually_exclusive_group()
    do_test.add_argument('--train', dest='test', action='store_false', default=False, help='Run in training mode, do not perform testing; default')
    do_test.add_argument('--test', dest='test', action='store_true', default=False, help='Run in testing mode, do not train')
    parser.add_argument('--binary-path', default=None, help='Path to binaries, current: aalta')
    parser.add_argument('--no-auto', action='store_true', help="Do not get parameters from params.txt when testing")
    parser.add_argument('--eval-name', default='test', help="Name of log and test files")
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--save-only', type=str, default='last', help='save which checkpoints: all, best, last')
    parser.add_argument('--params-file', type=str, help='load parameters from specified file')
    parser.add_argument('--seed', type=int, help='Global seed for python, numpy, tensorflow. If not specified, generate new one')

    # Typical Hyperparameters
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--initial-epoch', type=int, default=0, help='used to track the epoch number correctly when resuming training')
    parser.add_argument('--samples', type=int, default=None)
    return parser


EXCLUDE_AUTO_ARGS = ['job_dir', 'run_name', 'data_dir', 'binary_path', 'test', 'force_load', 'eval_name', 'load_from', 'load_parts']


def load_params(params_dict, path, exclude_auto=True):
    with tf.io.gfile.GFile(path, 'r') as f:
        d = json.loads(f.read())
    if exclude_auto:
        for exclude in EXCLUDE_AUTO_ARGS:
            if exclude in d:
                d.pop(exclude)
    dropped = []
    for q, qm in map(lambda x: (x, '--' + x.replace('_', '-')),  list(d)):
        if any(arg.startswith(qm) for arg in sys.argv[1:]): # drop if specified on command line
            d.pop(q)
            dropped.append(qm)
    print('Loaded parameters from', path, (', dropped ' + str(dropped) + ' as specified on command line') if dropped else '')
    new =  params_dict.copy()
    new.update(d)
    return new


def setup(**kwargs):
    # If testing, load from params.txt
    if kwargs['test'] and not kwargs['no_auto']:
        if kwargs['params_file'] is not None:
            raise NotImplementedError()
        load_path = path.join(kwargs['job_dir'], kwargs['run_name'], 'params.json')
        kwargs = load_params(kwargs, load_path, exclude_auto=True)
    elif kwargs['params_file'] is not None:
        kwargs = load_params(kwargs, kwargs['params_file'], exclude_auto=False)
    binary_path = kwargs['binary_path']

    # GPU stuff
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('GPUs', gpus)
    if len(gpus) > 1:
        print("More than one GPU specified, I'm scared!")
        sys.exit(1)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Get binaries
    filenames = [] #['aalta']
    if binary_path is not None:
        for filename in filenames:
            try:
                tf.io.gfile.makedirs('bin')
                tf.io.gfile.copy(path.join(binary_path, filename), path.join('bin', filename))
            except tf.errors.AlreadyExistsError:
                pass
    
    # Random stuff
    if kwargs['seed'] is None:
        random.seed()
        kwargs['seed'] = random.randint(0, 2**32 - 1)
        print('Seed not provided, generated new one:', kwargs['seed'])
    random.seed(kwargs['seed'])
    np.random.seed(kwargs['seed'])
    tf.random.set_seed(kwargs['seed'])
    return kwargs


def log_params(job_dir, run_name, _skip=None, **kwargs):
    if _skip is None:
        _skip = []
    logdir = path.join(job_dir, run_name)
    tf.io.gfile.makedirs(logdir)
    d = kwargs.copy()
    d.update({'job_dir' : job_dir, 'run_name' : run_name})
    for _s in _skip:
        if _s in d:
            d.pop(_s)
    with tf.io.gfile.GFile(path.join(logdir, 'params.json'), 'w') as f:
        f.write(json.dumps(d, indent=4) + '\n')


def checkpoint_path(job_dir, run_name, **kwargs):
    return path.join(job_dir, run_name, 'checkpoints')


def checkpoint_callback(job_dir, run_name, save_weights_only=True, save_only='all', **kwargs):
    if save_only == 'all':
        filepath = str(path.join(checkpoint_path(job_dir, run_name), 'cp_')) + 'ep{epoch:02d}_vl{val_loss:.3f}'  # save per epoch
    elif save_only == 'best':
        filepath = str(path.join(checkpoint_path(job_dir, run_name), 'best'))  # save best only
    elif save_only == 'last':
        filepath = str(path.join(checkpoint_path(job_dir, run_name), 'last'))  # save best only
    return tf.keras.callbacks.ModelCheckpoint(filepath, save_weights_only=save_weights_only, save_best_only=save_only=='best')


def get_log_dir(job_dir, run_name, **kwargs):
    return str(path.join(job_dir, run_name))


def tensorboard_callback(job_dir, run_name, **kwargs):
    log_dir = str(path.join(job_dir, run_name))
    return tf.keras.callbacks.TensorBoard(log_dir)


def last_checkpoint(job_dir, run_name, load_from=None, **kwargs):
    if load_from is not None:
        run_name = load_from
    return tf.train.latest_checkpoint(checkpoint_path(job_dir, run_name))
