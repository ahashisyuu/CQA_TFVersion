import tensorflow as tf

from .CQAModel import CQAModel


class Baseline(CQAModel):
    def build_model(self):
        with tf.variable_scope('baseline', initializer=tf.glorot_uniform_initializer()):

            return None

