# based on DeepLTL: https://github.com/reactive-systems/deepltl

import tensorflow as tf

from tgan_sr.transformer import attention
from tgan_sr.transformer import positional_encoding as pe
from tgan_sr.transformer.common import create_padding_mask, create_look_ahead_mask


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, params):
        """
            params: hyperparameter dictionary containing the following keys:
                d_embed_enc: int, dimension of encoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                dropout: float, percentage of droped out units
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
        """
        super(TransformerEncoderLayer, self).__init__()
        self.__dict__['params'] = params

        self.multi_head_attn = attention.MultiHeadAttention(params['d_embed_enc'], params['num_heads'], dtype=params['dtype'])

        if 'leaky_relu' in params['ff_activation']:
            alpha = float(params['ff_activation'].split('$')[1])
            ff_activation = lambda x: tf.nn.leaky_relu(x, alpha=alpha)
        else:
            ff_activation = params['ff_activation']
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(params['d_ff'], activation=ff_activation),
            tf.keras.layers.Dense(params['d_embed_enc'])
        ])

        self.norm_attn = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_ff = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout_attn = tf.keras.layers.Dropout(params['dropout'])
        self.dropout_ff = tf.keras.layers.Dropout(params['dropout'])

    def call(self, input, mask, training):
        """
        Args:
            input: float tensor with shape (batch_size, input_length, d_embed_dec)
            mask: float tensor with shape (batch_size, 1, 1, input_length)
            training: bool, whether layer is called in training mode or not
        """
        attn, _ = self.multi_head_attn(input, input, input, mask)
        attn = self.dropout_attn(attn, training=training)
        norm_attn = self.norm_attn(attn + input) # res connection

        ff_out = self.ff(norm_attn)
        ff_out = self.dropout_ff(ff_out, training=training)
        norm_ff_out = self.norm_ff(ff_out + norm_attn)

        return norm_ff_out


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, params):
        """
                d_embed_enc: int, dimension of encoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                dropout: float, percentage of droped out units
                ff_activation: string, activation function used in feed-forward networks
                input_vocab_size: int, size of input vocabulary
                num_heads: int, number of attention heads
                num_layers: int, number of encoder / decoder layers
        """
        super(TransformerEncoder, self).__init__()
        self.__dict__['params'] = params
        self.enc_layers = [TransformerEncoderLayer(params) for _ in range(params['num_layers'])]

    def call(self, x, padding_mask, training):
        for i in range(self.params['num_layers']):
            x = self.enc_layers[i](x, padding_mask, training)
        return x

