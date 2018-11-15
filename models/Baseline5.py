import tensorflow as tf

from .CQAModel import CQAModel
from layers.BiGRU import NativeGRU as BiGRU
import keras.losses


class Baseline5(CQAModel):
    def build_model(self):
        with tf.variable_scope('baseline', initializer=tf.glorot_uniform_initializer()):
            units = 300
            Q, C = self.QS, self.CT
            Q_len, C_len = self.QS_len, self.CT_len

            with tf.variable_scope('encode'):
                Q_sequence = tf.layers.conv1d(Q, filters=200, kernel_size=3, padding='same')
                C_sequence = tf.layers.conv1d(C, filters=200, kernel_size=3, padding='same')

            with tf.variable_scope('interaction'):
                Q_ = tf.expand_dims(Q_sequence, axis=2)  # (B, L1, 1, dim)
                C_ = tf.expand_dims(C_sequence, axis=1)  # (B, 1, L2, dim)
                hQ = tf.tile(Q_, [1, 1, self.CT_maxlen, 1])
                hC = tf.tile(C_, [1, self.QS_maxlen, 1, 1])
                H = tf.concat([hQ, hC], axis=-1)
                A = tf.layers.dense(H, units=200, activation=tf.tanh)  # (B, L1, L2, dim)

                rQ = tf.reduce_max(A, axis=2)
                rC = tf.reduce_max(A, axis=1)

            with tf.variable_scope('attention'):
                # concate
                cate_f_ = tf.expand_dims(self.cate_f, axis=1)
                Q_m = tf.concat([Q_sequence, rQ, tf.tile(cate_f_, [1, self.QS_maxlen, 1])], axis=-1)
                C_m = tf.concat([C_sequence, rC, tf.tile(cate_f_, [1, self.CT_maxlen, 1])], axis=-1)

                Q_m = tf.layers.dense(Q_m, units=300, activation=tf.tanh, name='fw1')
                C_m = tf.layers.dense(C_m, units=300, activation=tf.tanh, name='fw1', reuse=True)

                Q_m = tf.layers.dense(Q_m, units=1, activation=tf.tanh, name='fw2')
                C_m = tf.layers.dense(C_m, units=1, activation=tf.tanh, name='fw2', reuse=True)

                Q_m -= (1 - tf.expand_dims(self.QS_mask, axis=-1)) * 1e30
                C_m -= (1 - tf.expand_dims(self.CT_mask, axis=-1)) * 1e30
                Qalpha = tf.nn.softmax(Q_m, axis=1)
                Calpha = tf.nn.softmax(C_m, axis=1)

                Q_vec = tf.reduce_sum(Qalpha * rQ, axis=1)
                C_vec = tf.reduce_sum(Calpha * rC, axis=1)

            info = tf.concat([Q_vec, C_vec], axis=1)
            median = tf.layers.dense(info, 300, activation=tf.tanh)
            output = tf.layers.dense(median, self.args.categories_num, activation=tf.identity)

            return output

