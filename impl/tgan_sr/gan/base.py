"""Components of the GAN architecture"""

import tensorflow as tf
import numpy as np

from tgan_sr.transformer.base import TransformerEncoder
from tgan_sr.transformer import positional_encoding



class TransformerCritic(tf.keras.layers.Layer):
    """(W)GAN critic built with a Transformer encoder"""

    def __init__(self, params, sigmoid=False):
        """
        params : hyperparameters for the architecture, specified in train/gan.py
        sigmoid : whether to internally apply a sigmoid on the output or not
        """
        super().__init__()
        params = params.copy()
        self.params = params
        self.transformer_encoder = TransformerEncoder(params)
        if params['gan_critic_class_layers'] > 0: # split upper layers for internal classifier only
            class_enc_params = params.copy()
            class_enc_params['num_layers'] = params['gan_critic_class_layers']
            self.class_encoder = TransformerEncoder(class_enc_params)
        else:
            self.class_encoder = None
        if params['gan_critic_critic_layers'] > 0: # split upper layers for critic part only when used with internal classifier
            critic_enc_params = params.copy()
            critic_enc_params['num_layers'] = params['gan_critic_critic_layers']
            self.critic_encoder = TransformerEncoder(class_enc_params)
        else:
            self.critic_encoder = None
        self.final_projection = tf.keras.layers.Dense(3, activation=('sigmoid' if sigmoid else None))
        self.embed = tf.keras.layers.Dense(params['d_embed_enc'], kernel_initializer='uniform') # first operation on input, maps |V| to d_emb
        self.stdpe = positional_encoding.positional_encoding(params['max_encode_length'], params['d_embed_enc'], dtype=params['dtype']) # standard positional encoding
        self.dropout = tf.keras.layers.Dropout(params['dropout'])


    def call(self, x, positive_mask, training=False):
        """
        x : bs × l × |V| tensor
        positive mask : bs × l boolean tensor with False at masked (padded) positions
        returns bs × 2 tensor (class_output, gan_output; mostly only second position relevant)
        """
        batch_size, seq_len, d_v = tf.shape(x)
        padding_mask = tf.reshape(tf.cast(tf.logical_not(positive_mask), tf.float32), [batch_size, 1, 1, seq_len])
        pe = self.stdpe[:, :seq_len, :]

        x = self.embed(x)
        x *= tf.math.sqrt(tf.cast(self.params['d_embed_enc'], self.params['dtype'])) # magical scaling
        x += pe
        x = self.dropout(x, training=training)

        x = self.transformer_encoder(x, padding_mask, training=training)
        if self.class_encoder is not None: # split upper layers, only with internal classifier
            x_class = self.class_encoder(x, padding_mask, training=training)
        else:
            x_class = x
        if self.critic_encoder is not None: # split upper layers, only with internal classifier
            x_critic = self.critic_encoder(x, padding_mask, training=training)
        else:
            x_critic = x
        y_class = self.final_projection(x_class)
        y_critic = self.final_projection(x_critic)
        y = tf.concat([y_class[:, :, :1], y_critic[:, :, 1:]], axis=-1)
        y = tf.reduce_mean(y, axis=1)
        return y



class TransformerGenerator(tf.keras.layers.Layer):
    """(W)GAN generator built with a Transformer encoder"""

    def __init__(self, params, proc_logits_fn=None):
        """
        params : hyperparameters for the architecture, specified in train/gan.py
        proc_logits_fn : function to process output logits as last step, e.g. softmax with custom temperature
        """
        super().__init__()
        params = params.copy()
        params['num_layers'] = params['gan_generator_layers']
        self.params = params
        self.proc_logits_fn = proc_logits_fn
        self.transformer_encoder = TransformerEncoder(params)
        self.stdpe = positional_encoding.positional_encoding(params['max_encode_length'], params['d_embed_enc'], dtype=params['dtype'])
        self.dropout = tf.keras.layers.Dropout(params['dropout'])
        self.embedz = tf.keras.layers.Dense(params['d_embed_enc'], kernel_initializer='glorot_uniform') # initial embedding from scalar to d_emb
        self.final_proj = tf.keras.layers.Dense(params['input_vocab_size'])


    def call(self, z, positive_mask, training=False):
        """
        z : bs × l × 1 tensor of random noise
        positive mask : bs × l boolean tensor with False at masked (padded) positions
        returns bs × l × |V| tensor
        """
        batch_size, seq_len, d_z = tf.shape(z)
        padding_mask = tf.reshape(tf.cast(tf.logical_not(positive_mask), tf.float32), [batch_size, 1, 1, seq_len])

        x = self.embedz(z)
        x *= tf.math.sqrt(tf.cast(self.params['d_embed_enc'], self.params['dtype'])) # magical scaling
        pe = self.stdpe[:, :seq_len, :]

        x += pe
        x = self.dropout(x, training=True) # keep always true

        x = self.transformer_encoder(x, padding_mask, training=True) # keep always true
        y = self.final_proj(x)
        y = self.proc_logits_fn(y)
        return y
