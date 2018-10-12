import tensorflow as tf

from .CQAModel import CQAModel
from layers.QCNLayer import *


class QCN(CQAModel):
    def embedding(self):
        with tf.variable_scope('emb'):
            QS = tf.nn.embedding_lookup(self.word_mat, self.QSubject)
            QB = tf.nn.embedding_lookup(self.word_mat, self.QBody)
            CT = tf.nn.embedding_lookup(self.word_mat, self.CText)

            char_mat = tf.get_variable('char_mat',
                                       initializer=tf.random_normal((self.char_num + 1, self.args.char_dim)))

            ps_fchar = tf.reshape(self.cQS, [-1, tf.shape(self.cQS)[2]])  # (bm,w)
            pb_fchar = tf.reshape(self.cQB, [-1, tf.shape(self.cQB)[2]])
            qt_fchar = tf.reshape(self.cC, [-1, tf.shape(self.cC)[2]])

            ps_cemb = tf.nn.embedding_lookup(char_mat, ps_fchar)     # (bm,w,d)->(bm,d)->(b,m,d)
            pb_cemb = tf.nn.embedding_lookup(char_mat, pb_fchar)
            qt_cemb = tf.nn.embedding_lookup(char_mat, qt_fchar)

            # ps_cmask = tf.expand_dims(tf.sequence_mask(self.cQS_len, tf.shape(ps_fchar)[1], tf.float32), -1)
            # pb_cmask = tf.expand_dims(tf.sequence_mask(self.cQB_len, tf.shape(pb_fchar)[1], tf.float32), -1)
            # qt_cmask = tf.expand_dims(tf.sequence_mask(self.cC_len, tf.shape(qt_fchar)[1], tf.float32), -1)

            with tf.variable_scope("char") as scope:
                # ps_cinp = source2token(ps_cemb, ps_cmask, self.dropout_keep_prob, 'char')
                # scope.reuse_variables()
                # pb_cinp = source2token(pb_cemb, pb_cmask, self.dropout_keep_prob, 'char')
                # qt_cinp = source2token(qt_cemb, qt_cmask, self.dropout_keep_prob, 'char')
                ps_cinp = text_cnn(ps_cemb, [2, 3, 4, 5], 25)
                pb_cinp = text_cnn(pb_cemb, [2, 3, 4, 5], 25)
                qt_cinp = text_cnn(qt_cemb, [2, 3, 4, 5], 25)

            with tf.variable_scope("input") as scope:
                QS = tf.concat([QS, tf.reshape(ps_cinp, [tf.shape(self.cQS)[0], -1, 100])], -1)
                QB = tf.concat([QB, tf.reshape(pb_cinp, [tf.shape(self.cQB)[0], -1, 100])], -1)
                CT = tf.concat([CT, tf.reshape(qt_cinp, [tf.shape(self.cC)[0], -1, 100])], -1)
                QS = multilayer_highway(QS, 300, 1, tf.nn.elu, self.dropout_keep_prob, 'ps_input')
                QB = multilayer_highway(QB, 300, 1, tf.nn.elu, self.dropout_keep_prob, 'pb_input')

                # sigma = dense(CT, 300, tf.nn.tanh, self.dropout_keep_prob, 'qi_input_sigma')
                CT_ = dense(CT, 300, tf.nn.tanh, self.dropout_keep_prob, 'qt_input')
        return [QS, QB, CT_]

    def build_model(self):
        with tf.variable_scope('QC_interaction'):
            self.QS_mask = tf.expand_dims(self.QS_mask, -1)
            self.QB_mask = tf.expand_dims(self.QB_mask, -1)
            self.CT_mask = tf.expand_dims(self.CT_mask, -1)

            para, orth = ps_pb_interaction(self.QS, self.QB, self.QS_mask, self.QB_mask, self.dropout_keep_prob, 'parallel')
            p = tf.concat([para, orth], -1)
            q = tf.layers.dense(self.CT, 2 * 300, tf.nn.tanh, name='qt_tanh') * tf.layers.dense(
                self.CT, 2 * 300, tf.nn.sigmoid, name='qt_sigmoid')
            p_inter, q_inter = pq_interaction(p, q, self.QS_mask, self.CT_mask, self.dropout_keep_prob, 'p_q')
            p_vec = source2token(p_inter, self.QS_mask, self.dropout_keep_prob, 'p_vec')
            q_vec = source2token(q_inter, self.CT_mask, self.dropout_keep_prob, 'q_vec')

            l0 = tf.concat([p_vec, q_vec], 1)
            l1 = tf.layers.dense(l0, 300, tf.nn.elu, name='l1')
            l2 = tf.layers.dense(l1, 300, tf.nn.elu, name='l2')
            logits = tf.layers.dense(l2, self.args.categories_num, tf.identity, name='logits')
        return logits
