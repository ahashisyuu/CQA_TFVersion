# -*- utf-8 -*-
import json
import os
# import gensim
import pickle as pkl
import numpy as np
import tensorflow as tf

path = 'E:/tensorflow_workspace/QATAR_word2vec/' \
       'qatarliving_qc_size200_win5_mincnt1_rpl_skip3_phrFalse_2016_02_25.word2vec'

filename = os.path.join(path, 'qatarliving_qc_size200_win5_mincnt1_rpl_skip3'
                              '_phrFalse_2016_02_25.word2vec.bin.syn0.npy')

# model = gensim.models.Word2Vec.load(filename)
# print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))
#
# print(model['qatar'])
print(np.array([0., 100.23456789101112]))
data = np.load(filename)
print(data)

