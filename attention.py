#!/usr/bin/env python3

import tensorflow as tf


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

'''
The cross attention layer:

---> At the literal center for the Transformer is the cross-attention layer. This layer connects the
     encoder and decoder. This layer is the most straight-forward use of attention in the model

---> To implement this you pass the target sequence x as the query and the context sequence as the
     key/value when calling the mha layer
'''

class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x

'''
The global self attention layer:

---> The global self attention layer on the other hand lets every sequence element directly access every
     other sequence element, with only a few operations, and all the outputs can be computed in parallel
---> To implement thi layer your just need to pass the target sequence, x, as both the query, and value
     arguments to the mha layer
'''

class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

'''
The causal self attention layer:

---> To build a causal self attention layer, you need to use an appropriate mask when computing scores
     and summing the attention value S
---> This is taken care of automatically if you pass `use_causal_mask = True` to the MultiHeadAttention layer when it is called
---> Ther causal mask ensure that each location only has access to the locations that come before it
'''

class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x



class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])

        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x
