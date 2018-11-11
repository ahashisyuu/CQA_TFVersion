import json
import os
import tensorflow as tf

# path = 'E:/tensorflow_workspace/QATAR_word2vec/semeval-2016_2017-task3-subtaskA-unannotated-english.json'
#
# with open(os.path.join(path, 'semeval-2016_2017-task3-subtaskA-unannotated-english.json')) as fr, \
#          open('view.json', 'w') as fw:
#     data = json.load(fr)
#     for line in data[:1]:
#         json.dump(line, fw)
#         fw.write('\n')

sess = tf.Session()
a = tf.constant([[0.3, 0.5, 0.2], [0.1, 0.4, 0.5]])
b, indices = tf.nn.top_k(a, 2)
c = tf.gather(a, indices, axis=1)

print(sess.run(b))
print(sess.run(indices))
print(sess.run(c))




