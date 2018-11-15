import os
import random

import numpy as np
import tensorflow as tf

from utils import BatchDatasets, PRF, print_metrics, eval_reranker
from tqdm import tqdm
from layers.BiGRU import BiGRU

EPSILON = 1e-7


class CQAModel:

    def __init__(self, embedding_matrix, args, char_num=128, seed=1):
        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        seed = random.randint(1, 600)
        print(seed)
        tf.set_random_seed(seed)

        # hyper parameters and neccesary info
        self._is_train = tf.get_variable('is_train', shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable('word_mat',
                                        initializer=tf.constant(embedding_matrix, dtype=tf.float32),
                                        trainable=args.word_trainable)
        self._global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                            initializer=tf.constant_initializer(0), trainable=False)
        self.dropout_keep_prob = tf.get_variable('dropout', shape=[], dtype=tf.float32,
                                                 initializer=tf.constant_initializer(1), trainable=False)
        self.margin = tf.get_variable('margin', shape=[], dtype=tf.float32,
                                      initializer=tf.constant_initializer(1))
        self._lr = tf.get_variable('lr', shape=[], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.001), trainable=False)
        self._cweight = tf.get_variable('cweight', dtype=tf.float32,
                                        initializer=tf.constant([1. for _ in range(args.categories_num)]),
                                        trainable=False)
        self.args = args
        self.char_num = char_num

        # batch input
        self.inputs = self.Q, self.CText_pos, self.CText_neg,\
            self.cQ, self.cC_pos, self.cC_neg, self.Qcategory = self.create_input()
        self.inputs = self.inputs[:-1]

        # build model
        score_pos = self.network(self.Q, self.CText_pos, self.cQ, self.cC_pos, self.Qcategory)
        score_neg = self.network(self.Q, self.CText_neg, self.cQ, self.cC_neg, self.Qcategory)

        # computing loss
        with tf.variable_scope('predict'):
            self.cost = tf.maximum(0., self.margin + score_neg - score_pos)
            self.loss = tf.reduce_mean(self.cost)
            if self.args.l2_weight != 0:
                for v in tf.trainable_variables():
                    self.loss += self.args.l2_weight * tf.nn.l2_loss(v)

        # counting parameters
        self.count()

        # getting ready for training
        # tf.train.AdagradOptimizer(learning_rate=self._lr)
        self.opt = tf.train.AdamOptimizer(learning_rate=self._lr, epsilon=1e-6)
        grads = self.opt.compute_gradients(self.loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(gradients, clip_norm=5.0)
        self.train_op = self.opt.apply_gradients(zip(capped_grads, variables),
                                                 global_step=self._global_step)

        self.train_input_placeholder = [inp for inp in self.inputs if inp is not None] + [self.Qcategory]
        self.test_input_placeholder = [self.Q, self.CText_pos, self.cQ, self.cC_pos, self.Qcategory]

    def network(self, Q, C, cQ=None, cC=None, Qcategory=None):
        # preparing mask and length info
        Q_mask = tf.cast(tf.cast(Q, tf.bool), tf.float32)
        C_mask = tf.cast(tf.cast(C, tf.bool), tf.float32)

        Q_len = tf.reduce_sum(tf.cast(Q_mask, tf.int32), axis=1)
        C_len = tf.reduce_sum(tf.cast(C_mask, tf.int32), axis=1)
        Q_maxlen = tf.reduce_max(Q_len)
        C_maxlen = tf.reduce_max(C_len)

        N = Q_mask.get_shape()[0]

        if self.args.use_char_level:
            cQ_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(cQ, tf.bool), tf.int32), axis=2), [-1])
            cC_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(cC, tf.bool), tf.int32), axis=2), [-1])
        else:
            cQ_len = None
            cC_len = None

        # embedding word vector and char vector
        Q, C = self.embedding(Q, C, Q_maxlen, C_maxlen, cQ, cC, cQ_len, cC_len, N)

        # building model
        with tf.variable_scope('build_model'):
            return self.build_model(Q, C, Q_mask, C_mask, Q_len, C_len)

    def weight_cross_entropy_with_logits(self):
        # alpha = tf.multiply(tf.cast(self.label, tf.float32), tf.expand_dims(self._cweight, axis=0))
        # n_alpha = tf.multiply(tf.cast(1-self.label, tf.float32), tf.expand_dims(self._cweight, axis=0))
        # Focal Loss
        prob = tf.clip_by_value(self.predict_prob, EPSILON, 1-EPSILON)
        info = tf.cast(self.label, tf.float32) * tf.log(prob)
        return -tf.reduce_sum(info, axis=1)

    def create_input(self):
        q = tf.placeholder(tf.int32, [None, None])
        c_pos = tf.placeholder(tf.int32, [None, None])
        c_neg = tf.placeholder(tf.int32, [None, None])
        inputs = [q, c_pos, c_neg]

        if self.args.use_char_level:
            cq = tf.placeholder(tf.int32, [None, None, None])
            cc_pos = tf.placeholder(tf.int32, [None, None, None])
            cc_neg = tf.placeholder(tf.int32, [None, None, None])
            inputs += [cq, cc_pos, cc_neg]
        else:
            inputs += [None] * 3

        category = tf.placeholder(tf.int32, [None])

        return inputs + [category]

    def embedding(self, Q, C, Q_maxlen, C_maxlen, cQ, cC, cQ_len, cC_len, N):
        # word embedding
        with tf.variable_scope('emb'):
            Q = tf.nn.embedding_lookup(self.word_mat, Q)
            C = tf.nn.embedding_lookup(self.word_mat, C)

            embedded = [Q, C]

            if self.args.use_char_level:
                with tf.variable_scope('char', initializer=tf.glorot_uniform_initializer(), reuse=True) as scope:
                    char_mat = tf.get_variable('char_mat', shape=(self.char_num + 1, self.args.char_dim),
                                               initializer=tf.glorot_uniform_initializer())

                    cQ = tf.reshape(tf.nn.embedding_lookup(char_mat, cQ),
                                     [N * Q_maxlen, self.args.char_max_len, self.args.char_dim])
                    cC = tf.reshape(tf.nn.embedding_lookup(char_mat, cC),
                                    [N * C_maxlen, self.args.char_max_len, self.args.char_dim])

                    char_hidden = 8
                    cell_fw = tf.nn.rnn_cell.GRUCell(char_hidden, name='Qcell_fw', reuse=True)
                    cell_bw = tf.nn.rnn_cell.GRUCell(char_hidden, name='Qcell_bw', reuse=True)

                    _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, cQ, cQ_len, dtype=tf.float32, scope=scope)
                    cQ_emb = tf.reshape(tf.concat([state_fw, state_bw], axis=1),
                                        [N, Q_maxlen, 2 * char_hidden])

                    cell_fw = tf.nn.rnn_cell.GRUCell(char_hidden, name='Ccell_fw', reuse=True)
                    cell_bw = tf.nn.rnn_cell.GRUCell(char_hidden, name='Ccell_bw', reuse=True)
                    _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, cC, cC_len, dtype=tf.float32, scope=scope)
                    cC_emb = tf.reshape(tf.concat([state_fw, state_bw], axis=1),
                                        [N, C_maxlen, 2 * char_hidden])

                    char_embedded = [cQ_emb, cC_emb]
                    embedded = [tf.concat([a, b], axis=2)
                                for a, b in zip(embedded, char_embedded)]

            return embedded

    @property
    def cweight(self):
        return self.sess.run(self._cweight)

    @cweight.setter
    def cweight(self, value):
        self.sess.run(tf.assign(self._cweight, tf.constant(value, dtype=tf.float32)))

    @property
    def lr(self):
        return self.sess.run(self._lr)

    @lr.setter
    def lr(self, value):
        assert isinstance(value, float)
        self.sess.run(tf.assign(self._lr, tf.constant(value, dtype=tf.float32)))

    @property
    def dropout(self):
        return 1 - self.sess.run(self.dropout_keep_prob)

    @dropout.setter
    def dropout(self, value):
        assert isinstance(value, float)
        self.sess.run(tf.assign(self.dropout_keep_prob, tf.constant(1-value, dtype=tf.float32)))

    @property
    def global_step(self):
        return self.sess.run(self._global_step)

    @global_step.setter
    def global_step(self, value):
        assert isinstance(value, int)
        self.sess.run(tf.assign(self._global_step, tf.constant(value, dtype=tf.int32)))

    @property
    def is_train(self):
        return self.sess.run(self._is_train)

    @is_train.setter
    def is_train(self, value):
        assert isinstance(value, bool)
        self.sess.run(tf.assign(self._is_train, tf.constant(value, dtype=tf.bool)))

    def build_model(self, Q, C, Q_mask, C_mask, Q_len, C_len):
        raise NotImplementedError

    def evaluate(self, eva_data, steps_num, eva_type, eva_ID=None):
        label = []
        predict = []
        loss = []
        with tqdm(total=steps_num, ncols=70) as tbar:
            for batch_eva_data in eva_data:
                batch_label = batch_eva_data[-2]
                feed_dict = {inv: array for inv, array in zip(self.input_placeholder, batch_eva_data)}
                batch_loss, batch_predict = self.sess.run([self.loss, self.predict_prob], feed_dict=feed_dict)
                label.append(batch_label.argmax(axis=1))
                loss.append(batch_loss * batch_label.shape[0])
                predict.append(batch_predict)

                tbar.update(batch_label.shape[0])

        label = np.concatenate(label, axis=0)
        predict = np.concatenate(predict, axis=0)
        loss = sum(loss) / steps_num
        metrics = PRF(label, predict.argmax(axis=1))
        metrics['loss'] = loss

        if eva_ID is not None:
            eval_reranker(eva_ID, label, predict[:, 0], metrics['matrix'], categories_num=self.args.categories_num)

        loss_summ = tf.Summary(value=[tf.Summary.Value(
            tag="{}/loss".format(eva_type), simple_value=metrics['loss']), ])
        macro_F_summ = tf.Summary(value=[tf.Summary.Value(
            tag="{}/f1".format(eva_type), simple_value=metrics['macro_prf'][2]), ])
        acc = tf.Summary(value=[tf.Summary.Value(
            tag="{}/acc".format(eva_type), simple_value=metrics['acc']), ])
        return metrics, [loss_summ, macro_F_summ, acc]

    def one_train(self, batch_dataset, saver, writer, config, fold_num=None):
        loss_save = 100
        patience = 0
        self.lr = config.lr
        self.dropout = config.dropout

        print('---------------------------------------------')
        print('process train data')
        train_data = [batch for batch in batch_dataset.batch_train_data(fold_num=fold_num)]
        train_steps = batch_dataset.train_steps_num
        print('---------------------------------------------')

        print('class weight: ', batch_dataset.cweight)
        # self.cweight = [.9, 5., 1.1]

        print('\n---------------------------------------------')
        print('process dev data')
        dev_data = [batch for batch in batch_dataset.batch_dev_data()]
        dev_steps = batch_dataset.dev_steps_num
        dev_id = batch_dataset.dev_cID
        print('----------------------------------------------\n')
        # print(list(zip(dev_data[0][0], dev_data[0][-1])))

        for epoch in range(1, config.epochs + 1):
            print('---------------------------------------')
            print('EPOCH %d' % epoch)

            print('the number of samples: %d\n' % train_steps)

            print('training model')
            self.is_train = True

            with tqdm(total=train_steps, ncols=70) as tbar:
                for batch_train_data in train_data:
                    feed_dict = {inv: array for inv, array in zip(self.input_placeholder, batch_train_data)}
                    loss, train_op = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)

                    if self.global_step % config.period == 0:
                        loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=loss), ])
                        writer.add_summary(loss_sum, self.global_step)

                    tbar.update(batch_dataset.batch_size)

            print('\n---------------------------------------')
            print('\nevaluating model\n')
            self.is_train = False
            # metrics, summ = self.evaluate(train_data, train_steps, 'train')
            # metrics['epoch'] = epoch
            #
            # for s in summ:
            #     writer.add_summary(s, self.global_step)

            val_metrics, summ = self.evaluate(dev_data, dev_steps, 'dev', dev_id)
            val_metrics['epoch'] = epoch

            if val_metrics['loss'] < loss_save:
                loss_save = val_metrics['loss']
                patience = 0
            else:
                patience += 1

            if patience >= config.patience:
                self.lr = self.lr / 2.0
                loss_save = val_metrics['loss']
                patience = 0

            for s in summ:
                writer.add_summary(s, self.global_step)
            writer.flush()

            path = os.path.join(config.model_dir, self.__class__.__name__)
            if not os.path.exists(path):
                os.mkdir(path)
            if fold_num is not None:
                path = os.path.join(path, 'fold_%d' % fold_num)
                if not os.path.exists(path):
                    os.mkdir(path)
            filename = os.path.join(path, "epoch{0}_acc{1:.4f}_fscore{2:.4f}.model"
                                    .format(epoch, val_metrics['acc'], val_metrics['macro_prf'][2]))

            # print_metrics(metrics, 'train', path, categories_num=self.args.categories_num)
            print_metrics(val_metrics, 'val', path, categories_num=self.args.categories_num)
            saver.save(self.sess, filename)

    def train(self, batch_dataset: BatchDatasets, config):
        with self.sess:

            saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())

            path = os.path.join(config.model_dir, self.__class__.__name__)
            if not os.path.exists(config.model_dir):
                os.mkdir(config.model_dir)

            if config.k_fold > 1:
                for i in range(config.k_fold):
                    if config.load_best_model and os.path.exists(path):
                        print('------------  load model  ------------')
                        print(tf.train.latest_checkpoint(path + '/fold_{}'.format(i)))
                        saver.restore(self.sess, tf.train.latest_checkpoint(path + '/fold_{}'.format(i)))
                    path = os.path.join(path, 'fold_%d' % i)
                    if not os.path.exists(path):
                        os.mkdir(path)
                    writer = tf.summary.FileWriter(path)
                    self.one_train(batch_dataset, saver, writer, config, i)
                    tf.reset_default_graph()
            else:
                if config.load_best_model and os.path.exists(path):
                    print('------------  load model  ------------')
                    print('epoch5_acc0.7289_fscore0.7043.model')
                    saver.restore(self.sess, os.path.join(path, 'epoch5_acc0.7289_fscore0.7043.model'))
                if not os.path.exists(path):
                    os.mkdir(path)
                writer = tf.summary.FileWriter(path)
                self.one_train(batch_dataset, saver, writer, config)

    def one_test(self, batch_dataset, config):
        test_data = [batch for batch in batch_dataset.batch_test_data(2 * config.batch_size)]
        steps = batch_dataset.test_steps_num
        cID = batch_dataset.test_cID
        self.is_train = False
        self.cweight = [1., 1., 1.]
        test_metrics, _ = self.evaluate(test_data, steps, 'test', cID)
        print_metrics(test_metrics, 'test', categories_num=self.args.categories_num)

    def test(self, batch_dataset: BatchDatasets, config):
        with self.sess:
            self.sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            if config.k_fold > 1:
                sub_dir = os.listdir(config.model_dir)
                for name in sub_dir:
                    path = os.path.join(config.model_dir, name)
                    print(tf.train.latest_checkpoint(config.model_dir))
                    saver.restore(self.sess, tf.train.latest_checkpoint(path))
                    self.one_test(batch_dataset, config)
            else:
                print(tf.train.latest_checkpoint(config.model_dir))
                saver.restore(self.sess, tf.train.latest_checkpoint(config.model_dir))
                self.one_test(batch_dataset, config)

    def count(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            # print(shape)
            # print(len(shape))
            variable_parameters = 1
            for dim in shape:
                # print(dim)
                variable_parameters *= dim.value
            # print(variable_parameters)
            total_parameters += variable_parameters
        print('\n\n------------------------------------------------')
        print('total_parameters: ', total_parameters)
        print('------------------------------------------------\n\n')









