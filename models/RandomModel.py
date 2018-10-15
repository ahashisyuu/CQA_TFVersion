import tensorflow as tf

from .CQAModel import CQAModel


class RandomModel(CQAModel):
    def build_model(self):
        with tf.variable_scope('CT_random'):
            # score = tf.layers.dense(self.CT, 1, activation=tf.tanh)
            # score -= (1-tf.expand_dims(self.CT_mask, -1))*1e30
            # alpha = tf.nn.softmax(score)
            # r = tf.reduce_sum(alpha * self.CT, axis=1)

            return tf.random_uniform((60, 3))

