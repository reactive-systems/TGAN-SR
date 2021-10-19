# based on DeepLTL: https://github.com/reactive-systems/deepltl
# pylint: disable = line-too-long

import os.path as path
import tensorflow as tf
from tgan_sr.utils import ltl_parser, math_parser
from tgan_sr.utils.vocabulary import LTLVocabulary, TraceVocabulary


class LTLTracesDataset():
    """Dataset that consists of pairs of a LTL formula and a satisfying trace."""

    def __init__(self, name, ltl_vocab: LTLVocabulary, trace_vocab: TraceVocabulary, data_dir=None, reduce_witness_to_sat=False, only_sat=True, step_limit=None):
        r"""Given the name of the dataset tries to automatically determine data dir. Expects data file to have formula\nsolution\n format"""
        self.dataset_dir = path.join(data_dir, name) if data_dir is not None else name
        if not tf.io.gfile.exists(self.dataset_dir):
            raise FileNotFoundError('Cannot access dataset directory ' + str(self.dataset_dir))
        self.ltl_vocab = ltl_vocab
        self.trace_vocab = trace_vocab
        self.reduce_witness_to_sat = reduce_witness_to_sat
        self.only_sat = only_sat
        self.step_limit = step_limit
        self.targets = ['train', 'val', 'test']

    def get_dataset(self, splits=None, dtype=tf.int32, max_length_formula=-1, max_length_trace=-1, prepend_start_token=True):
        """Returns the requested splits of the dataset or a dict containing the default ones."""
        if splits is not None:
            self.targets = splits
        res = {}
        for id, split in enumerate(self.targets):
            res[split] = tf.data.Dataset.from_generator(self._generator, (dtype, dtype), args=(id, max_length_formula, max_length_trace, prepend_start_token))
        if splits is not None:
            res = [res[split] for split in splits]
        return res

    def _generator(self, split_id, max_length_formula, max_length_trace, prepend_start_token):
        target_file = path.join(self.dataset_dir, self.targets[split_id] + '.txt')
        with tf.io.gfile.GFile(target_file, 'r') as file:  # expect formula\ntrace\n format
            for line_in in file:
                if line_in == '\n':
                    return
                if line_in.startswith('#'):
                    if self.step_limit is not None and line_in.startswith('#step '):
                        step = int(line_in.strip()[6:])
                        if step > self.step_limit:
                            print('Read dataset until step', step)
                            return
                    continue
                line_in = line_in.split('#')[0].strip()
                line_out = next(file).strip()  # get second line
                if self.reduce_witness_to_sat and line_out and line_out != '0':
                    line_out = '1'
                elif self.only_sat and line_out == '0':
                    continue
                if max_length_formula >= 0 and len(line_in) > max_length_formula:
                    continue
                if max_length_trace >= 0 and len(line_out) > max_length_trace:
                    continue
                formula = ltl_parser.ltl_formula(line_in, 'network-polish')
                encoded_in = self.ltl_vocab.encode(formula.to_str('network-polish', spacing='all ops').split(' '))
                encoded_out = self.trace_vocab.encode(line_out, prepend_start_token=prepend_start_token)
                yield (tf.constant(encoded_in), tf.constant(encoded_out))


class MathDataset():
    """Dataset that consists of mathematical formulas"""
    def __init__(self, name, ltl_vocab: LTLVocabulary, data_dir=None):
        self.dataset_dir = path.join(data_dir, name) if data_dir is not None else name
        if not tf.io.gfile.exists(self.dataset_dir):
            raise FileNotFoundError('Cannot access dataset directory ' + str(self.dataset_dir))
        self.ltl_vocab = ltl_vocab
        self.targets = ['train', 'val', 'test']

    def get_dataset(self, splits=None, dtype=tf.int32, max_length_formula=-1, prepend_start_token=False, **kwargs):
        """Returns the requested splits of the dataset or a dict containing the default ones."""
        if splits is not None:
            self.targets = splits
        res = {}
        for id, split in enumerate(self.targets):
            res[split] = tf.data.Dataset.from_generator(self._generator, (dtype, dtype), args=(id, max_length_formula, prepend_start_token))
        if splits is not None:
            res = [res[split] for split in splits]
        return res

    def _generator(self, split_id, max_length_formula, prepend_start_token):
        target_file = path.join(self.dataset_dir, self.targets[split_id] + '.txt')
        with tf.io.gfile.GFile(target_file, 'r') as file:
            for line_in in file:
                if line_in == '\n':
                    return
                line_in = line_in.strip()
                try:
                    formula = math_parser.parse(line_in)
                except ltl_parser.ParseError:
                    continue
                as_list = formula.as_list()
                if max_length_formula >= 0 and len(as_list) > max_length_formula:
                    continue
                encoded_in = self.ltl_vocab.encode(as_list)
                yield (tf.constant(encoded_in), tf.constant(0, shape=(1,)))


def prepare_dataset_no_tf(dataset, batch_size, d_embedding, shuffle=True, pos_enc=False, in_length=None, out_length=None):
    """dataset pipeline preprocessing"""
    def shape_dataset(x, y):
        return ((x, y), y)

    def shape_pos_enc_dataset(x, y, z):
        return ((x, y, z), z)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(shape_pos_enc_dataset if pos_enc else shape_dataset)
    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(100000, reshuffle_each_iteration=True)
        # dataset = dataset.shuffle(10000, reshuffle_each_iteration=True)
    padded_shapes = (([in_length], [in_length, d_embedding], [out_length]), [out_length]) if pos_enc else (([in_length], [out_length]), [out_length])
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=True)
    return dataset


def prepare_dataset_buffer(dataset, batch_size, in_length=None, out_length=None):
    """GAN buffer loading pipeline"""
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    padded_shapes = ([in_length], [out_length])
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=True)
    return dataset
