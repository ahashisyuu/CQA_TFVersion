import tensorflow as tf

from .CQAModel import CQAModel
from layers.BiGRU import NativeGRU as BiGRU


class Baseline(CQAModel):

    def transformer(self, QS, QB, trans_type=0):
        # QS: (B, L1, dim), QB: (B, L2, dim)
        with tf.variable_scope('transformer'):
            QS_exp = tf.tile(tf.expand_dims(QS, axis=2), [1, 1, self.QB_maxlen, 1])  # (B, L1, L2, dim)
            QB_exp = tf.tile(tf.expand_dims(QB, axis=1), [1, self.QS_maxlen, 1, 1])  # (B, L1, L2, dim)

            infomation = tf.concat([QS_exp, QB_exp, QS_exp - QB_exp, QS_exp * QB_exp], axis=3)
            infomation = tf.nn.dropout(infomation, keep_prob=self.dropout_keep_prob)
            score_matrix = tf.layers.dense(infomation, 1, activation=tf.tanh, use_bias=False)  # (B, L1, L2, 1)

            if trans_type == 0:
                mask = tf.expand_dims(tf.expand_dims(self.QB_mask, axis=1), axis=3)
                score_matrix -= (1 - mask) * 1e30
                alpha = tf.nn.softmax(score_matrix, axis=2)  # L2
                newQ = tf.reduce_sum(alpha * QB_exp, axis=2)  # (B, L1, dim)
            else:
                mask = tf.expand_dims(tf.expand_dims(self.QS_mask, axis=2), axis=3)
                score_matrix -= (1 - mask) * 1e30
                alpha = tf.nn.softmax(score_matrix, axis=1)  # L2
                newQ = tf.reduce_sum(alpha * QB_exp, axis=1)  # (B, L2, dim)

            return newQ

    def build_model(self):
        with tf.variable_scope('baseline', initializer=tf.glorot_uniform_initializer()):
            units = 300
            Q, C = self.QBody, self.CT
            Q_len, C_len = self.QB_len, self.CT_len

            return output

