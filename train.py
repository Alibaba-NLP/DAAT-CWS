from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.crf as crf
import os
import cws
import codecs
import time
import random
import numpy as np
from itertools import izip
from utils import *
from layer import *
from argparse import ArgumentParser

class DAATNet():
    def __init__(self, src_train_path, src_test_path, tgt_train_path, tgt_test_path, emb_file, num_tags, batch_size, lr, epochs, emb_size, hidden_layers, 
        kernel_size, channels, dropout_emb, dropout_hidden, use_wn, use_crf, share_crf, use_src_crf, num_filters, filter_sizes):
        
        self.src_train_path = src_train_path
        self.src_test_path = src_test_path
        self.tgt_train_path = tgt_train_path
        self.tgt_test_path = tgt_test_path
        self.pre_trained_emb_path = emb_file

        self.num_tags = num_tags
        self.batch_size = batch_size
        self.lr = lr
        self.emb_size = emb_size
        self.epochs = epochs
         
        self.hidden_layers = hidden_layers
        self.kernel_size = kernel_size
        self.channels = channels
        self.dropout_emb = dropout_emb
        self.dropout_hidden = dropout_hidden
        self.use_wn = use_wn
        self.use_crf = use_crf
        self.share_crf = share_crf
        self.use_src_crf = use_src_crf

        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
          
        self.src_train_data = cws.read_train_file(codecs.open(self.src_train_path, 'r', 'utf8'))
        self.src_test_data = cws.read_train_file(codecs.open(self.src_test_path, 'r', 'utf8'))
        self.tgt_train_data = cws.read_train_file(codecs.open(self.tgt_train_path, 'r', 'utf8'))
        self.tgt_test_data = cws.read_train_file(codecs.open(self.tgt_test_path, 'r', 'utf8'))

        self.add_placeholders()
        self.read_data()
        self.build_model()

    def add_placeholders(self):
        self.src_seq_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='src_seq_ids')
        self.src_stag_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='src_stag_ids')
        self.src_seq_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='src_seq_lengths')
        self.tgt_seq_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='tgt_seq_ids')
        self.tgt_stag_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='tgt_stag_ids')
        self.tgt_seq_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='tgt_seq_lengths')
        self.is_train = tf.placeholder(dtype=tf.bool, shape=[], name='is_train')

    def read_data(self):
        # load character embeddings
        pre_trained = {}
        if self.pre_trained_emb_path and os.path.isfile(self.pre_trained_emb_path):
            for l in codecs.open(self.pre_trained_emb_path, 'r', 'utf8'):
                we = l.split()
                if len(we) == self.emb_size + 1:
                    w, e = we[0], np.array(map(float, we[1:]))
                    pre_trained[w] = e
        self.pre_trained = pre_trained
        
        # Load or create mappings.
        item2id, id2item = create_mapping(create_dic(self.src_train_data[0]+self.tgt_train_data[0], add_unk=True, add_pad=True))
        tag2id, id2tag = create_mapping(create_dic(self.src_train_data[-1]))
    
        self.item2id = item2id
        self.id2item = id2item
        self.tag2id = tag2id
        self.id2tag = id2tag

        self.src_train_data_ids = data_to_ids(self.src_train_data, [self.item2id] + [self.tag2id])
        self.tgt_train_data_ids = data_to_ids(self.tgt_train_data, [self.item2id] + [self.tag2id])
        print ('Finishing loading the dataset!!!', end='')

    def inference_src(self, scores, sequence_lengths=None):
        if not self.use_crf:
            return np.argmax(scores, 2)
        else:
            with tf.variable_scope(self.scope_src_crf, reuse=True):
                transitions = tf.get_variable('transitions').eval(session=self.sess)
            paths = np.zeros(scores.shape[:2], dtype=np.int32)
            for i in xrange(scores.shape[0]):
                tag_score, length = scores[i], sequence_lengths[i]
                if length == 0:
                    continue
                path, _ = crf.viterbi_decode(tag_score[:length], transitions)
                paths[i, :length] = path
            return paths

    def inference_tgt(self, scores, sequence_lengths=None):
        if not self.use_crf:
            return np.argmax(scores, 2)
        else:
            with tf.variable_scope(self.scope_tgt_crf, reuse=True):
                transitions = tf.get_variable('transitions').eval(session=self.sess)
            paths = np.zeros(scores.shape[:2], dtype=np.int32)
            for i in xrange(scores.shape[0]):
                tag_score, length = scores[i], sequence_lengths[i]
                if length == 0:
                    continue
                path, _ = crf.viterbi_decode(tag_score[:length], transitions)
                paths[i, :length] = path
            return paths

    def tag_sequence(self, data, labels):
        assert len(data) == len(labels)
        results = []
        tmp = []
        for i, label in enumerate(labels):
            if label == 'S':
                results.append(data[i])
            elif label == 'B':
                tmp.append(data[i])
            elif label == 'M':
                tmp.append(data[i])
            else:
                tmp.append(data[i])
                results.append(''.join(tmp))
                tmp = []
        return ' '.join(results).encode('utf8')

    def build_model(self):
        # embedding layer
        src_embedding_layer = Embedding_layer(vocab_size=len(self.item2id), emb_dim=self.emb_size, scope='src_char_emb')
        tgt_embedding_layer = Embedding_layer(vocab_size=len(self.item2id), emb_dim=self.emb_size, scope='tgt_char_emb')

        src_input = src_embedding_layer(self.src_seq_ids)
        tgt_input = tgt_embedding_layer(self.tgt_seq_ids)

        # gcnn encoder
        src_gcnn = GCNN_layer(hidden_layers=self.hidden_layers, kernel_size=self.kernel_size, channels=self.channels, dropout_emb=self.dropout_emb, 
            dropout_hidden=self.dropout_hidden, use_wn=self.use_wn, reuse=tf.AUTO_REUSE, scope='src_gcnn')
        tgt_gcnn = GCNN_layer(hidden_layers=self.hidden_layers, kernel_size=self.kernel_size, channels=self.channels, dropout_emb=self.dropout_emb,
            dropout_hidden=self.dropout_hidden, use_wn=self.use_wn, reuse=tf.AUTO_REUSE, scope='tgt_gcnn')
        share_gcnn = GCNN_layer(hidden_layers=self.hidden_layers, kernel_size=self.kernel_size, channels=self.channels, dropout_emb=self.dropout_emb,
            dropout_hidden=self.dropout_hidden, use_wn=self.use_wn, reuse=tf.AUTO_REUSE, scope='share_gcnn')

        # discriminator
        textCNN = TextCNN_layer(emb_size=self.emb_size, num_filters=self.num_filters, filter_sizes=self.filter_sizes, reuse=tf.AUTO_REUSE, scope='textcnn')
        
        if self.share_crf:
            self.scope_src_crf = self.scope_tgt_crf = 'crf'
        else:
            self.scope_src_crf = 'src_crf'
            self.scope_tgt_crf = 'tgt_crf' 
        
        # crf layer
        src_crf = CRF_layer(num_tags=self.num_tags, reuse=tf.AUTO_REUSE, scope=self.scope_src_crf)
        tgt_crf = CRF_layer(num_tags=self.num_tags, reuse=tf.AUTO_REUSE, scope=self.scope_tgt_crf)

        # output of gcnn encoder
        src_hidden = src_gcnn(inputs=src_input, seq_lengths=self.src_seq_lengths,is_train=self.is_train)
        tgt_hidden = src_gcnn(inputs=tgt_input, seq_lengths=self.tgt_seq_lengths, is_train=self.is_train)
        src_hidden_share = share_gcnn(inputs=src_input, seq_lengths=self.src_seq_lengths, is_train=self.is_train)
        tgt_hidden_share = share_gcnn(inputs=tgt_input, seq_lengths=self.tgt_seq_lengths, is_train=self.is_train)

        
        src_textcnn = textCNN(src_hidden_share, is_train=self.is_train)
        tgt_textcnn = textCNN(tgt_hidden_share, is_train=self.is_train)

        src_hidden_concat = tf.concat([src_hidden, src_hidden_share], axis=-1)
        tgt_hidden_concat = tf.concat([tgt_hidden, tgt_hidden_share], axis=-1)

        self.src_scores, self.src_loss = src_crf(src_hidden_concat, self.src_stag_ids, self.src_seq_lengths)
        self.tgt_scores, self.tgt_loss = tgt_crf(tgt_hidden_concat, self.tgt_stag_ids, self.tgt_seq_lengths)

        self.src_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=src_textcnn, labels=tf.ones_like(src_textcnn)))
        self.src_c_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=src_textcnn, labels=tf.zeros_like(src_textcnn)))
        self.tgt_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tgt_textcnn, labels=tf.ones_like(tgt_textcnn)))
        self.tgt_c_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tgt_textcnn, labels=tf.zeros_like(tgt_textcnn)))

        self.loss1 = self.src_loss + self.tgt_loss + self.src_d_loss + self.tgt_c_loss
        self.loss2 = self.src_loss + self.tgt_loss + self.src_c_loss + self.tgt_d_loss
        
        src_optimizer = tf.train.AdamOptimizer(self.lr)
        grads_and_vars_src = src_optimizer.compute_gradients(loss=self.loss1, var_list=tf.trainable_variables())
        grads_and_vars_src = [(g, v) for g, v in grads_and_vars_src if g is not None]

        grads_summary_op_src = tf.summary.histogram('grads_src', tf.concat([tf.reshape(g, [-1]) for g, _ in grads_and_vars_src], 0))
        grads_norm_src = tf.sqrt(sum([tf.reduce_sum(tf.pow(g, 2)) for g, _ in grads_and_vars_src]))
        grads_and_vars_src = [(g / (tf.reduce_max([grads_norm_src, 5]) / 5), v) for g, v in grads_and_vars_src]

        self.train_op_src = src_optimizer.apply_gradients(grads_and_vars_src)

        tgt_optimizer = tf.train.AdamOptimizer(self.lr)
        grads_and_vars_tgt = tgt_optimizer.compute_gradients(loss=self.loss2, var_list=tf.trainable_variables())
        grads_and_vars_tgt = [(g, v) for g, v in grads_and_vars_tgt if g is not None]

        grads_summary_op_tgt = tf.summary.histogram('grads_tgt', tf.concat([tf.reshape(g, [-1]) for g, _ in grads_and_vars_tgt], 0))
        grads_norm_tgt = tf.sqrt(sum([tf.reduce_sum(tf.pow(g, 2)) for g, _ in grads_and_vars_tgt]))
        grads_and_vars_tgt = [(g / (tf.reduce_max([grads_norm_tgt, 5]) / 5), v) for g, v in grads_and_vars_tgt]

        self.train_op_tgt = tgt_optimizer.apply_gradients(grads_and_vars_tgt)

    def train(self):
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        print('Finished.')
        print('Start training the network...')
        self.sess.run(init_op)
        start_time_begin = time.time()
        with tf.variable_scope('src_char_emb', reuse=True):
            embeddings = tf.get_variable('embeddings')
            value = self.sess.run(embeddings)
            count = 0
            for item in self.item2id:
                item_id = self.item2id[item]
                if item in self.pre_trained:
                    value[item_id] = self.pre_trained[item]
                    count += 1
            self.sess.run(embeddings.assign(value))

        with tf.variable_scope('tgt_char_emb', reuse=True):
            embeddings = tf.get_variable('embeddings')
            value = self.sess.run(embeddings)
            count = 0
            for item in self.item2id:
                item_id = self.item2id[item]
                if item in self.pre_trained:
                    value[item_id] = self.pre_trained[item]
                    count += 1
            self.sess.run(embeddings.assign(value))

        print('%d of %d character embeddings were loaded from pre-trained.' % (count, len(self.item2id)))

        global_step = 0
        for epoch in range(1, self.epochs + 1):
            print('Starting training network epoch %d...' % epoch)
            start_time = time.time()
            loss_ep = 0
            n_step = 0
            src_iterator = data_iterator(self.src_train_data_ids, self.batch_size, shuffle=True)
            tgt_iterator = data_iterator(self.tgt_train_data_ids, self.batch_size, shuffle=True)
            src_seq_ids_all = []
            src_stag_ids_all = []
            src_seq_lengths_all =[]
            tgt_seq_ids_all = []
            tgt_stag_ids_all = []
            tgt_seq_lengths_all =[]

            for batch in src_iterator:
                batch = create_input(batch)
                seq_ids, seq_other_ids_list, stag_ids, seq_lengths = batch[0], batch[1: -2], batch[-2], batch[-1]
                src_seq_ids_all.append(seq_ids)
                src_stag_ids_all.append(stag_ids)
                src_seq_lengths_all.append(seq_lengths)

            for batch in tgt_iterator:
                batch = create_input(batch)
                seq_ids, seq_other_ids_list, stag_ids, seq_lengths = batch[0], batch[1: -2], batch[-2], batch[-1]
                tgt_seq_ids_all.append(seq_ids)
                tgt_stag_ids_all.append(stag_ids)
                tgt_seq_lengths_all.append(seq_lengths)

            eval_batch_size=1024
            for i in range(min(len(src_seq_ids_all), len(tgt_seq_ids_all))):
                feed_dict = {self.src_seq_ids: src_seq_ids_all[i].astype(np.int32),
                             self.src_seq_lengths: src_seq_lengths_all[i].astype(np.int32),
                             self.src_stag_ids: src_stag_ids_all[i].astype(np.int32),
                             self.tgt_seq_ids: tgt_seq_ids_all[i].astype(np.int32),
                             self.tgt_seq_lengths: tgt_seq_lengths_all[i].astype(np.int32),
                             self.tgt_stag_ids: tgt_stag_ids_all[i].astype(np.int32),
                             self.is_train: True}
                
                n_step += 1
                global_step +=1
                if i % 2 == 0:
                    _, src_loss, tgt_loss = self.sess.run([self.train_op_src, self.src_loss, self.tgt_loss], feed_dict=feed_dict)
                else:
                    _, src_loss, tgt_loss = self.sess.run([self.train_op_tgt, self.src_loss, self.tgt_loss], feed_dict=feed_dict) 
                
                if global_step % 100 == 0:
                    print('Step %d, src_loss %.6f, tgt_loss %.6f' % (global_step, src_loss, tgt_loss))
            
            if self.use_src_crf:
                t_test_pre, t_test_rec, t_test_f1 = \
                    cws.evaluator((self.tgt_test_data[0], self.tgt_test_data[-1], self.tag_all_src(self.tgt_test_data[:-1], eval_batch_size)[1]), 
                         'tgt_test', epoch)
            else:
                t_test_pre, t_test_rec, t_test_f1 = \
                    cws.evaluator((self.tgt_test_data[0], self.tgt_test_data[-1], self.tag_all_tgt(self.tgt_test_data[:-1], eval_batch_size)[1]),
                         'tgt_test', epoch)

            print("Target domain test precision / recall / f1 score: %.2f / %.2f / %.2f" %
                  (t_test_pre * 100, t_test_rec * 100, t_test_f1 * 100))
        self.sess.close()


    def tag_src(self, data_iter):
        output = []
        for data in data_iter:
            batch = data_to_ids(data, [self.item2id])
            batch = create_input(batch)
            seq_ids, seq_other_ids_list, seq_lengths = batch[0], batch[1: -1], batch[-1]
            feed_dict = {self.src_seq_ids: seq_ids.astype(np.int32),
                         self.src_seq_lengths: seq_lengths.astype(np.int32),
                         self.is_train: False}
            scores = self.sess.run(self.src_scores, feed_dict)
            stag_ids = self.inference_src(scores, seq_lengths)
            for seq, stag_id, length in izip(data[0], stag_ids, seq_lengths):
                output.append((seq, [self.id2tag[t] for t in stag_id[:length]]))
            yield zip(*output)
            output = []

    def tag_all_src(self, data, batch_size):
        data_iter = data_iterator(data, batch_size=batch_size, shuffle=False)
        output = []
        for b in self.tag_src(data_iter):
            output.extend(zip(*b))
        return zip(*output)

    def tag_tgt(self, data_iter):
        output = []
        for data in data_iter:
            batch = data_to_ids(data, [self.item2id])
            batch = create_input(batch)
            seq_ids, seq_other_ids_list, seq_lengths = batch[0], batch[1: -1], batch[-1]
            feed_dict = {self.tgt_seq_ids: seq_ids.astype(np.int32),
                         self.tgt_seq_lengths: seq_lengths.astype(np.int32),
                         self.is_train: False}
            scores = self.sess.run(self.tgt_scores, feed_dict)
            stag_ids = self.inference_tgt(scores, seq_lengths)
            for seq, stag_id, length in izip(data[0], stag_ids, seq_lengths):
                output.append((seq, [self.id2tag[t] for t in stag_id[:length]]))
            yield zip(*output)
            output = []

    def tag_all_tgt(self, data, batch_size):
        data_iter = data_iterator(data, batch_size=batch_size, shuffle=False)
        output = []
        for b in self.tag_tgt(data_iter):
            output.extend(zip(*b))
        return zip(*output)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src_train_path', type=str, default='data/datasets/pku/train.txt', help='source domain train data')
    parser.add_argument('--src_test_path', type=str, default='data/datasets/pku/test.txt', help='source domain test data')
    parser.add_argument('--tgt_train_path', type=str, default='data/datasets/dm/train.txt', help='target domain train data')
    parser.add_argument('--tgt_test_path', type=str, default='data/datasets/dm/test.txt', help='target domain test data')
    parser.add_argument('--num_tags', type=int, default=4, help='number of tags BMES')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='max training epochs')
    parser.add_argument('--emb_size', type=int, default=200, help='character embedding size')
    parser.add_argument('--emb_file', type=str, default='data/embeddings/char.vec', help='pre-trained character embedding file')
    parser.add_argument('--hidden_layers', type=int, default=5, help='number of gcnn layers')
    parser.add_argument('--kernel_size', type=int, default=3, help='kernel_size of gcnn layer')
    parser.add_argument('--channels', type=list, default=[200]*5, help='output dimension of gcnn layer')
    parser.add_argument('--dropout_emb', type=float, default=0.2, help='dropout rate for embedding layer')
    parser.add_argument('--dropout_hidden', type=float, default=0.3, help='dropout rate for gcnn layer')
    parser.add_argument('--use_wn', type=bool, default=True, help='using weight normalisation in gcnn layer')
    parser.add_argument('--use_crf', type=bool, default=True, help='use crf as decoder')
    parser.add_argument('--share_crf', type=bool, default=True, help='share crf of source and target domain')
    parser.add_argument('--use_src_crf', type=bool, default=False, help='using source domain crf for test')
    parser.add_argument('--num_filters', type=int, default=200, help='number of filters in textcnn')
    parser.add_argument('--filter_sizes', type=list, default=[3,4,5], help='number of individual filter size in textcnn')
    args = parser.parse_args()
    print(args)

    runner = DAATNet(src_train_path = args.src_train_path,
                     src_test_path = args.src_test_path,
                     tgt_train_path = args.tgt_train_path,
                     tgt_test_path = args.tgt_test_path,
                     emb_file = args.emb_file,
                     num_tags = args.num_tags,
                     batch_size = args.batch_size,
                     lr = args.lr,
                     epochs = args.epochs,
                     emb_size = args.emb_size,
                     hidden_layers = args.hidden_layers, 
                     kernel_size  = args.kernel_size, 
                     channels = args.channels,
                     dropout_emb = args.dropout_emb,
                     dropout_hidden = args.dropout_hidden, 
                     use_wn = args.use_wn,
                     use_crf = args.use_crf,
                     share_crf = args.share_crf,
                     use_src_crf =  args.use_src_crf,
                     num_filters = args.num_filters, 
                     filter_sizes = args.filter_sizes)
    runner.train() 
