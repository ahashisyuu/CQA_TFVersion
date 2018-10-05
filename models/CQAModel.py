import os
import numpy as np
import tensorflow as tf

from utils import BatchDatasets, PRF, print_metrics, eval_reranker
from tqdm import tqdm


class CQAModel:

    def __init__(self, embedding_matrix, categories_num=3, char_embed=None):
        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        # hyper parameters and neccesary info
        self.is_train = tf.get_variable('is_train', shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable('word_mat',
                                        initializer=tf.constant(embedding_matrix, dtype=tf.float32),
                                        trainable=False)
        self.char_mat = tf.get_variable('char_mat',
                                        initializer=tf.constant(char_embed, dtype=tf.float32),
                                        trainable=True)
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.dropout_keep_prob = tf.get_variable('dropout', shape=[], dtype=tf.float32,
                                                 initializer=tf.constant_initializer(1), trainable=False)
        self.lr = tf.get_variable('lr', shape=[], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.001), trainable=False)
        self.categories_num = categories_num

        # batch input
        self.QSubject, self.QBody, self.CText, self.cQS, self.cQB, self.cC = self.create_input()

        # preparing mask and length info
        self.QS_mask = tf.cast(self.QSubject, tf.bool)
        self.QB_mask = tf.cast(self.QBody, tf.bool)
        self.QS_len = tf.reduce_sum(tf.cast(self.QS_mask, tf.int32), axis=1)
        self.QB_len = tf.reduce_sum(tf.cast(self.QB_mask, tf.int32), axis=1)
        self.cQS_len = tf.reduce_sum(tf.cast(tf.cast(self.cQS, tf.bool), tf.int32), axis=2)
        self.cQB_len = tf.reduce_sum(tf.cast(tf.cast(self.cQB, tf.bool), tf.int32), axis=2)
        self.cC_len = tf.reduce_sum(tf.cast(tf.cast(self.cC, tf.bool), tf.int32), axis=2)

        # building model
        self.output = self.build_model()

        # getting label
        self.label = self.create_label()

        # computing loss
        with tf.variable_scope('predict'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output,
                                                                labels=tf.stop_gradient(self.label))
            self.loss = tf.reduce_mean(losses)
            self.predict_prob = tf.nn.softmax(self.output, axis=1)
            self.predict = tf.argmax(self.predict_prob, axis=1)

        # getting ready for training
        self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr, epsilon=1e-6)
        grads = self.opt.compute_gradients(self.loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(gradients, clip_norm=5.0)
        self.train_op = self.opt.apply_gradients(zip(capped_grads, variables),
                                                 global_step=self.global_step)

        self.input_placeholder = [self.QSubject, self.QBody, self.CText, self.cQS, self.cQB, self.cC, self.label]

    def create_input(self):
        qs = tf.placeholder(tf.int32, [None, None])
        qb = tf.placeholder(tf.int32, [None, None])
        c = tf.placeholder(tf.int32, [None, None])

        cqs = tf.placeholder(tf.int32, [None, None, None])
        cqb = tf.placeholder(tf.int32, [None, None, None])
        cc = tf.placeholder(tf.int32, [None, None, None])

        return qs, qb, c, cqs, cqb, cc

    def create_label(self):
        return tf.placeholder(tf.int32, [None, self.categories_num])

    @property
    def lr(self):
        return self.sess.run(self.lr)

    @lr.setter
    def lr(self, value):
        assert isinstance(value, float)
        self.sess.run(tf.assign(self.lr, tf.constant(value, dtype=tf.float32)))

    def dropout(self, value):
        assert isinstance(value, float)
        self.sess.run(tf.assign(self.dropout_keep_prob, tf.constant(1-value, dtype=tf.float32)))

    @property
    def global_step(self):
        return self.sess.run(self.global_step)

    @global_step.setter
    def global_step(self, value):
        assert isinstance(value, int)
        self.sess.run(tf.assign(self.global_step, tf.constant(value, dtype=tf.int32)))

    @property
    def is_train(self):
        return self.sess.run(self.is_train)

    @is_train.setter
    def is_train(self, value):
        assert isinstance(value, bool)
        self.sess.run(tf.assign(self.is_train, tf.constant(value, dtype=tf.bool)))

    def build_model(self):
        raise NotImplementedError

    def evaluate(self, eva_data, steps_num, eva_type, eva_ID=None):
        label = []
        predict = []
        loss = []
        with tqdm(total=steps_num) as tbar:
            for batch_eva_data in tqdm(eva_data):
                batch_label = batch_eva_data[-1]

                feed_dict = {inv: array for inv, array in zip(self.input_placeholder, batch_eva_data)}
                batch_loss, batch_predict = self.sess.run([self.loss, self.predict_prob], feed_dict=feed_dict)

                label.append(batch_label.argmax(axis=1))
                loss.append(batch_loss*batch_label.shape[0])
                predict.append(batch_predict)

                tbar.update(batch_label.shape[0])

        label = np.concatenate(label, axis=0)
        predict = np.concatenate(predict, axis=0)
        loss = np.concatenate(loss)

        loss = loss.sum() / steps_num
        metrics = PRF(label, predict.argmax(axis=1))
        metrics['loss'] = loss

        if eva_ID is not None:
            eval_reranker(eva_ID, label, predict[:, 0], metrics['matrix'])

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
        for epoch in range(1, config.epochs + 1):
            print('---------------------------------------')
            print('EPOCH %d' % epoch)

            train_data = batch_dataset.batch_train_data(fold_num=fold_num)
            train_steps = batch_dataset.train_steps_num

            print('the number of samples: %d\n' % train_steps)

            print('training model')
            self.is_train = True
            with tqdm(total=train_steps) as tbar:
                for batch_train_data in tqdm(train_data):
                    feed_dict = {inv: array for inv, array in zip(self.input_placeholder, batch_train_data)}
                    loss, train_op = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)

                    if self.global_step % config.period == 0:
                        loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=loss), ])
                        writer.add_summary(loss_sum, self.global_step)

                    tbar.update(batch_dataset.batch_size)

            print('---------------------------------------')
            print('\nevaluating model\n')
            self.is_train = False
            metrics, summ = self.evaluate(train_data, train_steps, 'train')
            metrics['epoch'] = epoch

            for s in summ:
                writer.add_summary(s, self.global_step)

            dev_data = batch_dataset.batch_dev_data(dev_batch_size=2*config.batch_size)
            dev_steps = batch_dataset.dev_steps_num
            dev_id = batch_dataset.dev_cID
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

            print_metrics(metrics, 'train', path)
            print_metrics(val_metrics, 'val', path)

            saver.save(self.sess, filename)

    def train(self, batch_dataset: BatchDatasets, config):
        with self.sess:
            writer = tf.summary.FileWriter(config.log_dir)
            saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())
            self.lr = config.lr
            if config.k_fold > 1:
                for i in range(config.k_fold):
                    self.one_train(batch_dataset, saver, writer, config, i)
            else:
                self.one_train(batch_dataset, saver, writer, config)

    def one_test(self, batch_dataset, config):
        test_data = batch_dataset.batch_test_data()
        steps = batch_dataset.test_steps_num
        cID = batch_dataset.test_cID
        self.is_train = False
        test_metrics, _ = self.evaluate(test_data, steps, 'test', cID)
        print_metrics(test_metrics, 'test')

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









