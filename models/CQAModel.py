import os
import random

import numpy as np
import tensorflow as tf

from utils import BatchDatasets, PRF, print_metrics, eval_reranker
from tqdm import tqdm


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
                                        trainable=False)
        self._global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                            initializer=tf.constant_initializer(0), trainable=False)
        self.dropout_keep_prob = tf.get_variable('dropout', shape=[], dtype=tf.float32,
                                                 initializer=tf.constant_initializer(1), trainable=False)
        self._lr = tf.get_variable('lr', shape=[], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.001), trainable=False)
        # self._cweight = tf.get_variable('cweight', dtype=tf.float32,
        #                                 initializer=tf.constant([1. for _ in range(args.categories_num)]),
        #                                 trainable=False)
        self.args = args
        self.char_num = char_num

        # batch input
        self.inputs = self.QSubject, self.QBody, self.CText, self.cQS, self.cQB, self.cC = self.create_input()

        # preparing mask and length info
        self.QS_mask = tf.cast(tf.cast(self.QSubject, tf.bool), tf.float32)
        self.QB_mask = tf.cast(tf.cast(self.QBody, tf.bool), tf.float32)
        self.CT_mask = tf.cast(tf.cast(self.CText, tf.bool), tf.float32)

        self.QS_len = tf.reduce_sum(tf.cast(self.QS_mask, tf.int32), axis=1)
        self.QB_len = tf.reduce_sum(tf.cast(self.QB_mask, tf.int32), axis=1)
        self.CT_len = tf.reduce_sum(tf.cast(self.CT_mask, tf.int32), axis=1)
        self.QS_maxlen = tf.reduce_max(self.QS_len)
        self.QB_maxlen = tf.reduce_max(self.QB_len)
        self.CT_maxlen = tf.reduce_max(self.CT_len)

        if self.args.use_char_level:
            self.cQS_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.cQS, tf.bool), tf.int32), axis=2), [-1])
            self.cQB_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.cQB, tf.bool), tf.int32), axis=2), [-1])
            self.cC_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.cC, tf.bool), tf.int32), axis=2), [-1])

        # embedding word vector and char vector
        self.QS, self.QB, self.CT = self.embedding()

        # building model
        self.output = self.build_model()

        # getting label
        self.label = self.create_label()

        # computing loss
        with tf.variable_scope('predict'):
            # self.label = tf.multiply(tf.cast(self.label, tf.float32), tf.expand_dims(self._cweight, axis=0))

            self.predict_prob = tf.nn.softmax(self.output, axis=1)
            # self.predict = tf.argmax(self.predict_prob, axis=1)
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output,
                                                                labels=tf.stop_gradient(self.label))
            # losses = self.weight_cross_entropy_with_logits(self.predict_prob, self.label)
            self.loss = tf.reduce_mean(losses)
            if self.args.l2_weight != 0:
                for v in tf.trainable_variables():
                    self.loss += self.args.l2_weight * tf.nn.l2_loss(v)

        # getting ready for training
        self.opt = tf.train.AdamOptimizer(learning_rate=self._lr, epsilon=1e-6)
        grads = self.opt.compute_gradients(self.loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(gradients, clip_norm=5.0)
        self.train_op = self.opt.apply_gradients(zip(capped_grads, variables),
                                                 global_step=self._global_step)

        self.input_placeholder = [inp for inp in self.inputs if inp is not None] + [self.label]

    def weight_cross_entropy_with_logits(self, logits, labels, weights=[1, 5, 1]):
        # labels = tf.stop_gradient(labels)
        labels = tf.cast(labels, dtype=tf.float32)
        self.mul = labels * tf.log(logits)
        self.mul2 = tf.multiply(self.mul, weights)
        return -tf.reduce_sum(self.mul2, reduction_indices=[1])

    def create_input(self):
        qs = tf.placeholder(tf.int32, [None, None])
        qb = tf.placeholder(tf.int32, [None, None])
        c = tf.placeholder(tf.int32, [None, None])
        inputs = [qs, qb, c]

        if self.args.use_char_level:
            cqs = tf.placeholder(tf.int32, [None, None, None])
            cqb = tf.placeholder(tf.int32, [None, None, None])
            cc = tf.placeholder(tf.int32, [None, None, None])
            inputs += [cqs, cqb, cc]
        else:
            inputs += [None] * 3
        return inputs

    def embedding(self):
        # word embedding
        with tf.variable_scope('emb'):
            QS = tf.nn.embedding_lookup(self.word_mat, self.QSubject)
            QB = tf.nn.embedding_lookup(self.word_mat, self.QBody)
            CT = tf.nn.embedding_lookup(self.word_mat, self.CText)

            embedded = [QS, QB, CT]

            if self.args.use_char_level:
                with tf.variable_scope('char', initializer=tf.glorot_uniform_initializer()):
                    char_mat = tf.get_variable('char_mat', shape=(self.char_num + 1, self.args.char_dim),
                                               initializer=tf.glorot_uniform_initializer())
                    N = tf.shape(self.cQS)[0]
                    cQS = tf.reshape(tf.nn.embedding_lookup(char_mat, self.cQS),
                                     [N*self.QS_maxlen, self.args.char_max_len, self.args.char_dim])
                    cQB = tf.reshape(tf.nn.embedding_lookup(char_mat, self.cQB),
                                     [N * self.QB_maxlen, self.args.char_max_len, self.args.char_dim])
                    cC = tf.reshape(tf.nn.embedding_lookup(char_mat, self.cC),
                                    [N * self.CT_maxlen, self.args.char_max_len, self.args.char_dim])

                    char_hidden = 8
                    cell_fw = tf.nn.rnn_cell.GRUCell(char_hidden)
                    cell_bw = tf.nn.rnn_cell.GRUCell(char_hidden)

                    _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, cQS, self.cQS_len, dtype=tf.float32)
                    cQS_emb = tf.reshape(tf.concat([state_fw, state_bw], axis=1), [N, self.QS_maxlen, 2 * char_hidden])

                    _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, cQB, self.cQB_len, dtype=tf.float32)
                    cQB_emb = tf.reshape(tf.concat([state_fw, state_bw], axis=1), [N, self.QB_maxlen, 2 * char_hidden])

                    _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, cC, self.cC_len, dtype=tf.float32)
                    cC_emb = tf.reshape(tf.concat([state_fw, state_bw], axis=1), [N, self.CT_maxlen, 2 * char_hidden])

                    char_embedded = [cQS_emb, cQB_emb, cC_emb]
                    embedded = [tf.concat([a, b], axis=2) for a, b in zip(embedded, char_embedded)]

            return embedded

    def create_label(self):
        return tf.placeholder(tf.int32, [None, self.args.categories_num])

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

    def build_model(self):
        raise NotImplementedError

    def evaluate(self, eva_data, steps_num, eva_type, eva_ID=None):
        label = []
        predict = []
        loss = []
        with tqdm(total=steps_num, ncols=70) as tbar:
            for batch_eva_data in eva_data:
                batch_label = batch_eva_data[-1]
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

        print('\n---------------------------------------------')
        print('process dev data')
        dev_data = [batch for batch in batch_dataset.batch_dev_data(dev_batch_size=2*config.batch_size)]
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
            # self.cweight = batch_dataset.cweight
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
            # self.cweight = [1., 1., 1.]
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
            writer = tf.summary.FileWriter(config.log_dir)
            saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())

            path = os.path.join(config.model_dir, self.__class__.__name__)
            if config.load_best_model and os.path.exists(path):
                print('------------  load model  ------------')
                saver.restore(self.sess, tf.train.latest_checkpoint(path))

            if config.k_fold > 1:
                for i in range(config.k_fold):
                    self.one_train(batch_dataset, saver, writer, config, i)
                    tf.reset_default_graph()
            else:
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
                    saver.restore(self.sess, tf.train.latest_checkpoint(path))
                    self.one_test(batch_dataset, config)
            else:
                saver.restore(self.sess, tf.train.latest_checkpoint(config.model_dir))
                self.one_test(batch_dataset, config)









