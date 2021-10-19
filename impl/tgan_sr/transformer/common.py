# based on DeepLTL: https://github.com/reactive-systems/deepltl

import tensorflow as tf


def create_padding_mask(indata, pad_id, dtype=tf.float32):
    """
        indata: int tensor with shape (batch_size, input_length)
        pad_id: int, encodes the padding token
        dtype: tf.dtypes.Dtype(), data type of padding mask
    Returns:
        padding mask with shape (batch_size, 1, 1, input_length) that indicates padding with 1 and 0 everywhere else
    """
    mask = tf.cast(tf.math.equal(indata, pad_id), dtype)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size, dtype=tf.float32):
    """
    creates a look ahead mask that masks future positions in a sequence, e.g., [[[[0, 1, 1], [0, 0, 1], [0, 0, 0]]]] for size 3
    Args:
        size: int, specifies the size of the look ahead mask
        dtype: tf.dtypes.Dtype(), data type of look ahead mask
    Returns:
        look ahead mask with shape (1, 1, size, size) that indicates masking with 1 and 0 everywhere else
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size), dtype), -1, 0)
    return tf.reshape(mask, [1, 1, size, size])
