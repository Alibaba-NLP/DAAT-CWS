import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.crf as crf
import os 
import numpy as np

class Embedding_layer():
    def __init__(self, vocab_size, emb_dim, emb_project=False, scope="char_emb"):
        self.scope = scope 
        self.emb_project = emb_project
        with tf.variable_scope(self.scope):
            
            self.embeddings = tf.get_variable(name="embeddings", shape=[vocab_size, emb_dim], dtype=tf.float32,
                                        trainable=True)

            if self.emb_project:
                self.dense = tf.layers.Dense(units=emb_dim, use_bias=True, _reuse=tf.AUTO_REUSE, name="emb_project")

    def __call__(self, char_ids):
        with tf.variable_scope(self.scope):
            char_emb = tf.nn.embedding_lookup(self.embeddings, char_ids)
            if self.emb_project:
                char_emb = self.dense(char_emb)
        return char_emb


class GCNN_layer():
    def __init__(self, hidden_layers, kernel_size, channels, dropout_emb, dropout_hidden, use_wn=True, reuse=None, scope='gcnn'):
        self.hidden_layers = hidden_layers
        self.kernel_size = kernel_size
        self.channels = channels
        self.dropout_emb = dropout_emb
        self.dropout_hidden = dropout_hidden
        self.use_wn = use_wn
        self.reuse = reuse
        self.scope = scope

                 
    def __call__(self, inputs, seq_lengths, is_train):
        # Define the encoder
        with tf.variable_scope(self.scope, reuse=self.reuse, initializer=tf.uniform_unit_scaling_initializer()):
            masks = tf.cast(tf.sequence_mask(seq_lengths), tf.float32)
            # Dropout on embedding output.
            if self.dropout_emb:
                inputs = tf.cond(is_train,
                             lambda: tf.nn.dropout(inputs, 1 - self.dropout_emb),
                             lambda: inputs)

            hidden_output = inputs
            pre_channels = inputs.get_shape()[-1].value
            for i in xrange(self.hidden_layers):
                k = self.kernel_size
                cur_channels = self.channels[i]
                filter_w = tf.get_variable('filter_w_%d' % i, shape=[k, pre_channels, cur_channels], dtype=tf.float32)
                filter_v = tf.get_variable('filter_v_%d' % i, shape=[k, pre_channels, cur_channels], dtype=tf.float32)
                bias_b = tf.get_variable('bias_b_%d' % i, shape=[cur_channels],
                                         initializer=tf.zeros_initializer(dtype=tf.float32))
                bias_c = tf.get_variable('bias_c_%d' % i, shape=[cur_channels],
                                         initializer=tf.zeros_initializer(dtype=tf.float32))

                # Weight normalization.
                if self.use_wn:
                    epsilon = 1e-12
                    g_w = tf.get_variable('g_w_%d' % i, shape=[k, 1, cur_channels], dtype=tf.float32)
                    g_v = tf.get_variable('g_v_%d' % i, shape=[k, 1, cur_channels], dtype=tf.float32)
                    filter_w = g_w * filter_w / (tf.sqrt(tf.reduce_sum(filter_w ** 2, 1, keepdims=True)) + epsilon)
                    filter_v = g_v * filter_v / (tf.sqrt(tf.reduce_sum(filter_v ** 2, 1, keepdims=True)) + epsilon)

                w = tf.nn.conv1d(hidden_output, filter_w, 1, 'SAME') + bias_b
                v = tf.nn.conv1d(hidden_output, filter_v, 1, 'SAME') + bias_c

                hidden_output = w * tf.nn.sigmoid(v)

                hidden_output = hidden_output * tf.expand_dims(masks, -1)

                if self.dropout_hidden:
                    hidden_output = tf.cond(is_train,
                                            lambda: tf.nn.dropout(hidden_output, 1 - self.dropout_hidden),
                                            lambda: hidden_output
                                            )

                pre_channels = cur_channels

            hidden_output = hidden_output
            return hidden_output


class CRF_layer():
    def __init__(self, num_tags, reuse=None, scope="crf"):
        self.num_tags = num_tags
        self.reuse = reuse
        self.scope = scope
        
    def __call__(self, inputs, stag_ids, seq_lengths):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            scores = layers.fully_connected(inputs, self.num_tags, tf.identity)
            cost, transitions = crf.crf_log_likelihood(inputs=scores, tag_indices=stag_ids, sequence_lengths=seq_lengths)
            return scores, tf.reduce_mean(-cost)


class TextCNN_layer():
    def __init__(self, emb_size, num_filters, filter_sizes, reuse=None, scope='textcnn'):
        self.emb_size = emb_size
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.num_filters_total = num_filters * len(filter_sizes)
        self.reuse = reuse
        self.scope = scope

    def __call__(self, inputs, is_train):
        # Text cnn model
        inputs_expanded=tf.expand_dims(inputs, -1)
        pooled_outputs = []
        with tf.variable_scope(self.scope, reuse=self.reuse):
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.variable_scope("conv-%s" % filter_size):
                    w = tf.get_variable("filter-%s" % filter_size, [filter_size, self.emb_size, 1, self.num_filters], initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
                    conv = tf.nn.conv2d(inputs_expanded, w, strides=[1, 1, 1, 1], padding="VALID",name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.reduce_max(h, axis=1, keepdims=True)
                    pooled_outputs.append(pooled)
            h_pool = tf.concat(pooled_outputs,3)
            h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total])
            h_full_conn = tf.layers.dense(h_pool_flat, 1, activation=None, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
            h_drop = tf.nn.dropout(h_full_conn, keep_prob=0.5)
        return h_drop
