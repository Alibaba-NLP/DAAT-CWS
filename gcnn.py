import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.crf as crf
import os 
import numpy as np

INT_TYPE = np.int32
FLOAT_TYPE = np.float32

class gcnn():
    def __init__(self, inputs, seq_lengths, stag_ids, vocab_size, emb_size, scope='gcnn', is_train=True, reuse=False):
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.inputs = inputs
        self.seq_lengths = seq_lengths
        self.stag_ids = stag_ids
        self.scope = scope
        self.num_tags = 4
        self.is_train = is_train
        self.reuse=reuse
        self.encoder()
                 
    def encoder(self, is_training=False, hidden_layers=5, kernel_size=3, channels=[200]*5, dropout_emb=0.2, dropout_hidden=0.2, use_wn=True, use_bn=False):
        # Define the encoder
        # embeddings = tf.get_variable('embeddings', [self.vocab_size, self.emb_size])
        with tf.variable_scope(self.scope, reuse=self.reuse, initializer=tf.uniform_unit_scaling_initializer()):
            masks = tf.cast(tf.sequence_mask(self.seq_lengths, maxlen=64), FLOAT_TYPE)
            # Dropout on embedding output.
            if dropout_emb:
                self.inputs = tf.cond(self.is_train,
                                 lambda: tf.nn.dropout(self.inputs, 1 - dropout_emb),
                                 lambda: self.inputs)
            hidden_output = self.inputs
            pre_channels = self.inputs.get_shape()[-1].value
            for i in xrange(hidden_layers):
                k = kernel_size
                cur_channels = channels[i]
                filter_w = tf.get_variable('filter_w_%d' % i, shape=[k, pre_channels, cur_channels], dtype=FLOAT_TYPE)
                filter_v = tf.get_variable('filter_v_%d' % i, shape=[k, pre_channels, cur_channels], dtype=FLOAT_TYPE)
                bias_b = tf.get_variable('bias_b_%d' % i, shape=[cur_channels],
                                         initializer=tf.zeros_initializer(dtype=FLOAT_TYPE))
                bias_c = tf.get_variable('bias_c_%d' % i, shape=[cur_channels],
                                         initializer=tf.zeros_initializer(dtype=FLOAT_TYPE))

                # Weight normalization.
                if use_wn:
                    epsilon = 1e-12
                    g_w = tf.get_variable('g_w_%d' % i, shape=[k, 1, cur_channels], dtype=FLOAT_TYPE)
                    g_v = tf.get_variable('g_v_%d' % i, shape=[k, 1, cur_channels], dtype=FLOAT_TYPE)
                    # Perform wn
                    filter_w = g_w * filter_w / (tf.sqrt(tf.reduce_sum(filter_w ** 2, 1, keep_dims=True)) + epsilon)
                    filter_v = g_v * filter_v / (tf.sqrt(tf.reduce_sum(filter_v ** 2, 1, keep_dims=True)) + epsilon)

                w = tf.nn.conv1d(hidden_output, filter_w, 1, 'SAME') + bias_b
                v = tf.nn.conv1d(hidden_output, filter_v, 1, 'SAME') + bias_c

                if use_bn:
                    w = layers.batch_norm(inputs=v, decay=0.9, is_training=self.is_train, center=True, scale=True,
                                          scope='BatchNorm_w_%d' % i)
                    v = layers.batch_norm(inputs=w, decay=0.9, is_training=self.is_train, center=True, scale=True,
                                          scope='BatchNorm_v_%d' % i)

                hidden_output = w * tf.nn.sigmoid(v)

                # Mask paddings.
                hidden_output = hidden_output * tf.expand_dims(masks, -1)
                # Dropout on hidden output.
                if dropout_hidden:
                    hidden_output = tf.cond(self.is_train,
                                            lambda: tf.nn.dropout(hidden_output, 1 - dropout_hidden),
                                            lambda: hidden_output
                                            )

                pre_channels = cur_channels

            hidden_output = hidden_output
            self.fc1 = hidden_output

class Embedding_layer():
    def __init__(self, scope, vocab_size, emb_size, reuse=False):
        self.scope = scope
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.reuse = reuse
        self.init_embeddings()

    def init_embeddings(self):
        with tf.variable_scope(self.scope, reuse=self.reuse, initializer=tf.uniform_unit_scaling_initializer()):
            self.embeddings = tf.get_variable('embeddings', [self.vocab_size, self.emb_size])
    
    def get_embeddings(self, seq_ids):      
        inputs = tf.nn.embedding_lookup(self.embeddings, seq_ids)
        return inputs 


class CRF_Layer():
    def __init__(self, num_tags=4, scope=None, reuse=False):
        self.num_tags = num_tags
        self.scope = scope
        self.reuse = reuse
    
    def tagger(self, inputs, stag_ids, seq_lengths):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            scores = layers.fully_connected(inputs, self.num_tags, tf.identity)
            #self.fc2 = scores
            cost, transitions = crf.crf_log_likelihood(inputs=scores, tag_indices=stag_ids, sequence_lengths=seq_lengths)
            cost = -tf.reduce_mean(cost)
            trans = transitions
            cost = cost
            return scores, cost

class GAN():
    def __init__(self, scope, source_domain_labels, target_domain_labels):
        # self.inpdduts = inputs
        self.scope = scope
        self.initializer=tf.random_normal_initializer(stddev=0.1, seed=2018)
        self.source_domain_labels = source_domain_labels
        self.target_domain_labels = target_domain_labels
        self.embed_size = 200
        self.num_filters = 200
        self.filter_sizes = [3,4,5]
        self.num_filters_total=self.num_filters * len(self.filter_sizes)

    def discriminator(self, inputs, reuse=False, trainable=True):
        # Text cnn model
        inputs_expanded=tf.expand_dims(inputs,-1)
        pooled_outputs = []
        with tf.variable_scope(self.scope, reuse=reuse):
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.variable_scope("convolution-pooling-%s" % filter_size):
                    filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters],initializer=self.initializer)
                    conv = tf.nn.conv2d(inputs_expanded, filter, strides=[1, 1, 1, 1], padding="VALID",name="conv")
                    conv = tf.contrib.layers.batch_norm(conv, is_training=True, scope='cnn_bn_')
                    b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
                    # h = tf.nn.leaky_relu(tf.nn.bias_add(conv, b), name="leaky_relu")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.nn.max_pool(h, ksize=[1, 64 - filter_size + 1, 1, 1],strides=[1, 1, 1, 1], padding='VALID',name="pool")
                    #pooled = tf.reduce_max(h, axis=1, keep_dims=True)
                    pooled_outputs.append(pooled)
            h_pool = tf.concat(pooled_outputs,3)
            h_pool_flat = tf.reshape(h_pool, [-1,self.num_filters_total])
            h_drop = tf.nn.dropout(h_pool_flat, keep_prob=0.5)
            fc3 = tf.layers.dense(h_drop, 2, activation=None, use_bias=True)
            return fc3

    def build_ad_loss(self,disc_s, disc_t):
        #source_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=disc_s, labels=self.source_domain_labels)
        #target_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=disc_t, labels=self.target_domain_labels)
        #target_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_t,labels=tf.ones_like(disc_t)*(1-0.6)+0.3)
        #target_loss = tf.reduce_mean(target_loss)
        #source_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_s,labels=tf.ones_like(disc_s)*(1-0.6)+0.3)
        #source_loss = tf.reduce_mean(source_loss)
        source_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_s, labels=tf.ones_like(disc_s)-0.2))
        source_dis_loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_s, labels=tf.zeros_like(disc_s)+0.2))
        target_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_t, labels=tf.ones_like(disc_t)-0.2))
        target_dis_loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_t, labels=tf.zeros_like(disc_t)+0.2))
        #d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_s,labels=tf.ones_like(disc_s)*(1-0.6)+0.3))+tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_t, labels=tf.zeros_like(disc_t)*(1-0.6)+0.3))
        #g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_s,labels=tf.zeros_like(disc_s)*(1-0.6)+0.3))+tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_t, labels=tf.ones_like(disc_t)*(1-0.6)+0.3))
        return source_dis_loss, source_dis_loss_2, target_dis_loss, target_dis_loss_2
