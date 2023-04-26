#!/usr/bin/env python3

import tensorflow as tf
from attention import CausalSelfAttention, CrossAttention
from embedding_pos_encode import PositionalEmbedding
from attention import FeedForward

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.cross_attention0 = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.cross_attention1 = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context0, context1):
        # gotta build two cross attention mechanisms, one for guidance and one for source doc
        x = self.causal_self_attention(x=x)

        guidance = self.cross_attention0(x=x, context=context0)

        source = self.cross_attention1(x=guidance, context=context1)


        # Cache the last attention scores for plotting later
        #self.last_attn_scores = self.cross_attention.last_attn_scores

        final = self.ffn(source)  # Shape `(batch_size, seq_len, d_model)`.
        return final # instead of x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
                        for _ in range(num_layers)]
        self.last_attn_scores = None

    def call(self, x, context0, context1):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x) # (batch_size, target_seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context0, context1)

        #self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model)
        return x
