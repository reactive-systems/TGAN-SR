# based on DeepLTL: https://github.com/reactive-systems/deepltl
# pylint: disable = line-too-long, no-member

"""Train a (W)GAN model. Invoke module from command line (python -m ...)"""


import sys, os
import math
from argparse import Namespace
import multiprocessing as mp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # reduce TF verbosity
import tensorflow as tf
from tensorflow.io import gfile #pylint: disable=import-error

from tgan_sr.train.common import *
from tgan_sr.utils import vocabulary, datasets, math_parser
from tgan_sr.gan.main import TransformerGAN



def run():
    # Argument parsing
    parser = argparser()
    # add specific arguments
    parser.add_argument('--problem', type=str, default='ltl', help='available problems: ltl, math')
    parser.add_argument('--d-embed-enc', type=int, default=128)
    parser.add_argument('--d-ff', type=int, default=512)
    parser.add_argument('--ff-activation', default='relu')
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--warmup-steps', type=int, default=4000)
    parser.add_argument('--force-load', default=False, action='store_true', help='Ensure that weigths from checkpoint are loaded, fail otherwise')
    parser.add_argument('--max-encode-length', type=int, help='filter input length')
    parser.add_argument('--load-from', type=str, help='name of a different run to load from checkpoint')
    parser.add_argument('--load-parts', type=str, default='all', help='load only fragments of the model, e.g. critic, generator')
    parser.add_argument('--objectives', type=str, default='gan', help='Possible objectives: class, gan')
    parser.add_argument('--val-set', type=str, default=None, help='External validation dataset')
    params = parser.parse_args()
    params_dict = setup(**vars(params))

    # Default parameters
    d = {}
    d['gan_generate_confusion'] = False
    d['gan_trainsteps_infer_interval'] = 10
    d['gan_trainsteps_process_interval'] = 5
    d['gan_generator_layers'] = 4
    d['gan_critic_target_fn'] = 'sigmoid-log' # one of: logits, sigmoid, sigmoid-log
    d['gan_critic_target_mode'] = 'one-minus' # one of: direct, one-minus
    d['gan_generator_target_fn'] = 'sigmoid-log' # one of: logits, sigmoid, sigmoid-log
    d['gan_generator_target_mode'] = 'direct' # one of: direct, one-minus
    d['gan_critic_steps'] = 1
    d['gan_gradient_penalty'] = 0.
    d['gan_intgrad_method'] = 'uniform' # one of: uniform, none
    d['gan_intgrad_target'] = 1
    d['gan_sigma_real'] = 0.1
    d['gan_objweight_class'] = 1
    d['gan_objweight_genclass'] = 1 # TODO check how!!
    d['gan_objweight_confusion'] = 0.5
    d['gan_delay_confusion_steps'] = 0
    d['gan_critic_class_layers'] = 0
    d['gan_critic_critic_layers'] = 0
    d['gan_copy_shape_critic'] = True
    d['gan_copy_shape_generator'] = True
    d['gan_force_constant_lr'] = False
    d['gan_class_loss'] = 'crossentropy' # crossentropy / hinge
    d['gan_process_generated_samples'] = False
    d['gan_filter_generated_entropy_threshold'] = 0.0
    d['gan_save_valid_samples'] = False
    d['gan_save_candidate_samples'] = False
    d['gan_solve_valid_samples'] = True
    d['gan_balance_valid_samples'] = True
    d['gan_solve_tool'] = 'aalta'

    d['gan_incremental_learning_mode'] = False
    d['gan_created_buffer_size'] = 100000
    d['gan_created_buffer_method'] = 'reservoir'
    d['gan_incremental_usage_zero_step'] = None
    d['gan_incremental_usage_full_step'] = None
    d['gan_load_buffer_from'] = None
    d['gan_load_buffer_until'] = None
    d['gan_confusion_loss'] = 'entropy' # entropy / mse / mae

    if not params.test:
        for key in ['eval_name', 'no_auto']:
            del params_dict[key]
    else:
        for key in ['epochs', 'initial_epoch']:
            del params_dict[key]
    for k, v in d.items(): # update only non-existent values
        if k not in params_dict:
            params_dict[k] = v
    params = Namespace(**params_dict)


    # Construct vocabulary and datasets
    max_in_length = 50 # hardcoded (!)
    num_variables = 10 # hardcoded (!)
    if params.problem == 'ltl':
        aps = list(map(chr, range(97, 97 + num_variables)))
        consts = ['0', '1']
        input_vocab = vocabulary.LTLVocabulary(aps=aps, consts=consts + ['%'],  ops=['!', 'X', 'F', 'G', '&', '|', '>', '=', 'U', 'W'], eos=True, start=False, mask=True)
        target_vocab = vocabulary.TraceVocabulary(aps=[], consts=consts, ops=[], special=[], eos=False, start=False, pad=True)
        dataset = datasets.LTLTracesDataset(params.ds_name, input_vocab, target_vocab, data_dir=params.data_dir, reduce_witness_to_sat=True, only_sat=False)
        if params.val_set is not None:
            external_dataset = datasets.LTLTracesDataset(params.val_set, input_vocab, target_vocab, data_dir=None, reduce_witness_to_sat=True, only_sat=False)
    elif params.problem == 'math':
        consts = list(map(str, range(0,10)))
        input_vocab = vocabulary.LTLVocabulary(aps=[], consts=consts, ops=math_parser.token_dict.keys(), eos=True, start=False, mask=False)
        target_vocab = vocabulary.TraceVocabulary(aps=[], consts=['0', '1'], ops=[], special=[], eos=False, start=False, pad=True) # just a dummy
        dataset = datasets.MathDataset(params.ds_name, input_vocab, data_dir=params.data_dir)
        assert params.val_set is None
    additional_in_tokens = 1
    max_encode_length = max_in_length + additional_in_tokens
    max_decode_length = 1
    if params.max_encode_length is None:
        base_2 = math.ceil(math.log(max_encode_length, 2))
        params.max_encode_length = 2**base_2 if max_encode_length / 2**base_2 >= 0.8 else max_encode_length # use next power of 2
    params.input_vocab_size = input_vocab.vocab_size()
    params.input_pad_id = input_vocab.pad_id
    params.input_eos_id = input_vocab.eos_id
    params.input_start_id = input_vocab.start_id
    params.target_vocab_size = target_vocab.vocab_size()
    assert params.target_vocab_size == 3
    assert target_vocab.vocab[2] == '1'
    params.dtype = 'float32'

    print('Specified dimension of encoder embedding:', params.d_embed_enc)
    params.d_embed_enc -= params.d_embed_enc % params.num_heads  # round down
    print('Parameters:')
    params_dict = vars(params)
    for key, val in params_dict.items():
        print('{:25} : {}'.format(key, val))


    # Load datasets
    if not params.test: # train mode
        splits_from_source = ['train'] 
        if params.val_set is None:
            splits_from_source.append('val')
        splits = dataset.get_dataset(splits_from_source, max_length_formula = params.max_encode_length-additional_in_tokens,
            max_length_trace=max_decode_length, prepend_start_token=False)
        if params.val_set is None:
            train_dataset, val_dataset = splits
        else:
            train_dataset = splits[0]
            val_dataset, = external_dataset.get_dataset(['val'], max_length_formula = params.max_encode_length-additional_in_tokens,
                max_length_trace=max_decode_length, prepend_start_token=False)
            print('using external validation set')
        if params.samples is not None:
            train_dataset = train_dataset.take(int(params.samples))
            val_dataset = val_dataset.take(int(params.samples/10))
        train_dataset = datasets.prepare_dataset_no_tf(train_dataset, params.batch_size, params.d_embed_enc, shuffle=True, in_length=params.max_encode_length)
        val_dataset = datasets.prepare_dataset_no_tf(val_dataset, params.batch_size, params.d_embed_enc, shuffle=False, in_length=params.max_encode_length)
    else:  # test mode
        val_dataset, = dataset.get_dataset(['val'], max_length_formula = params.max_encode_length-additional_in_tokens,
            max_length_trace=max_decode_length, prepend_start_token=False)
        val_dataset = datasets.prepare_dataset_no_tf(val_dataset, params.batch_size, params.d_embed_enc, shuffle=False)


    # Model & Training specification
    if not params.test:  # train mode
        model = TransformerGAN(params_dict, input_vocab, get_log_dir(**params_dict))
        model.compile(run_eagerly=True)
        latest_checkpoint = last_checkpoint(**params_dict)
        if latest_checkpoint:
            if params.load_parts == 'all':
                model.load_weights(latest_checkpoint) #.expect_partial()
                print(f'Loaded model weights from checkpoint {latest_checkpoint}')
            else:
                parts_to_load = params.load_parts.strip().split(',')
                loading_dict = { q : getattr(model, q) for q in parts_to_load }
                loading_model = tf.train.Checkpoint(**loading_dict)
                loading_model.restore(latest_checkpoint).expect_partial()
                print('Loaded weights for parts', ','.join(parts_to_load), f'from checkpoint {latest_checkpoint}')
        elif params.force_load:
            sys.exit('Failed to load weights, no checkpoint found!')
        else:
            print('No checkpoint found, creating fresh parameters')

        # Load GAN generated data buffer
        if params.gan_load_buffer_from is not None:
            if params.gan_load_buffer_from.endswith('.txt'):
                load_dir = os.path.dirname(params.gan_load_buffer_from)
                load_name = os.path.basename(params.gan_load_buffer_from)[:-4]
            else:
                load_dir = params.gan_load_buffer_from
                load_name = 'all'
            buffer_dataset = datasets.LTLTracesDataset(load_dir, input_vocab, target_vocab, data_dir=None, reduce_witness_to_sat=True, only_sat=False, step_limit=params.gan_load_buffer_until)
            buffer_dataset, *_ = buffer_dataset.get_dataset(splits=[load_name], max_length_formula = params.max_encode_length-additional_in_tokens, prepend_start_token=False)
            buffer_dataset = datasets.prepare_dataset_buffer(buffer_dataset, params.batch_size, in_length=params.max_encode_length, out_length=1)
            num_batches = 0
            for x,y in buffer_dataset:
                batch = tf.concat([x,y], axis=1) # add label to input
                model.created_buffer.update(batch)
                num_batches += 1
            print('Loaded', num_batches, 'batches into GAN buffer!')
            
        sys.stdout.flush()
        mp.set_start_method('forkserver')

        callbacks = []
        if not params.no_save:
            callbacks.append(checkpoint_callback(save_weights_only=True, **params_dict))
        callbacks.append(ProxyCallback())
        np.set_printoptions(threshold=np.inf, linewidth=300)

        # TRAIN!
        log_params(**params_dict)
        model.fit(train_dataset, epochs=params.epochs, validation_data=val_dataset, validation_freq=1, callbacks=callbacks, initial_epoch=params.initial_epoch, verbose=2, shuffle=False)

    else:  # test mode
        model = TransformerGAN(params_dict, input_vocab, get_log_dir(**params_dict))
        model.compile(run_eagerly=True)
        latest_checkpoint = last_checkpoint(**params_dict)
        if latest_checkpoint:
            model.load_weights(latest_checkpoint).expect_partial()
            print(f'Loaded weights from checkpoint {latest_checkpoint}')
        else:
            sys.exit('Failed to load weights, no checkpoint found!')
        sys.stdout.flush()
        np.set_printoptions(threshold=np.inf, linewidth=300)
        model.evaluate(val_dataset)


if __name__ == "__main__":
    run()


class ProxyCallback(tf.keras.callbacks.Callback):
    """redirects epoch callbacks to the model"""
    def on_epoch_begin(self, epoch, logs=None):
        self.model.on_epoch_begin(epoch, logs)
    def on_epoch_end(self, epoch, logs=None):
        self.model.on_epoch_end(epoch, logs)    
