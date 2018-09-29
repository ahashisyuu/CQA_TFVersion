import os
import random
import numpy as np

from prettytable import PrettyTable
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.sequence import pad_sequences


class BatchDatasets:
    def __init__(self, train_samples: list, dev_samples: list, max_len, char_max_len,
                 need_shuffle=False, batch_size=64, k_fold=0, test_samples=None):
        self.train_samples = self.processing_sample(train_samples)
        self.dev_samples = self.processing_sample(dev_samples)
        self.test_samples = self.processing_sample(test_samples)
        self.max_len = max_len
        self.char_max_len = char_max_len
        self.need_shuffle = need_shuffle
        self.batch_size = batch_size
        self.train_samples_num = len(self.train_samples)
        self.dev_samples_num = len(self.dev_samples)
        self.k_fold = k_fold
        self.train_steps_num = 0
        self.dev_steps_num = 0
        self.test_steps_num = 0
        self.train_data = None
        self.train_label = None
        self.dev_data = None
        self.dev_label = None

        if k_fold > 1:  # merge train data and dev data
            self.train_samples += self.dev_samples
            self.train_samples = [trans for trans in zip(*self.train_samples)]
            self.data = [np.asarray(trans) for trans in self.train_samples[1:-1]]
            self.label = np.asarray(self.train_samples[-1])

            skf = StratifiedKFold(n_splits=k_fold, random_state=0)
            self.index_list = [index for index in skf.split(self.data[0], self.label)]

        else:
            if self.need_shuffle:
                self.shuffle()
            self.train_samples = [trans for trans in zip(*self.train_samples)]
            self.train_data = self.train_samples[1:-1]
            self.train_label = self.train_samples[-1]

            self.dev_samples = [trans for trans in zip(*self.dev_samples)]
            self.dev_data = self.dev_samples[1:-1]
            self.dev_label = np.asarray(self.dev_samples[-1])

        if test_samples is not None:
            self.test_samples = [trans for trans in zip(*self.test_samples)]
            self.test_data = self.dev_samples[1:-1]
            self.test_label = np.asarray(self.dev_samples[-1])

    @staticmethod
    def processing_sample(samples_list):
        if samples_list is None:
            return None

        new_samples = []
        for samples in samples_list:
            new_samples += samples

        return new_samples

    def shuffle(self):
        random.shuffle(self.train_samples)

    def get_len(self, e):
        return min(len(max(e, key=len)), self.max_len)

    def pad_sentence(self, e, maxlen):
        return pad_sequences(e, padding='post', truncating='post', maxlen=maxlen)

    def padding(self, batch_data):
        assert len(batch_data) == 6
        cur_max_len = [self.get_len(e) for e in batch_data[0:3]]*2

        return [self.pad_sentence(e, maxlen=l) for e, l in zip(batch_data, cur_max_len)]

    def mini_batch_data(self, data, label, batch_size):
        data_size = label.shape[0]
        for batch_start in np.arange(0, data_size, batch_size):
            batch_train_data = [e[batch_start:batch_start+batch_size]
                                for e in data]
            batch_train_label = label[batch_start:batch_start+batch_size]
            yield [self.padding(batch_train_data), batch_train_label]

    def batch_train_data(self, fold_num=None):
        if self.k_fold > 1:
            train_index, dev_index = self.index_list[fold_num]
            self.train_data = [list(element[train_index]) for element in self.data]
            self.train_label = self.label[train_index]
            self.dev_data = [list(element[dev_index]) for element in self.data]
            self.dev_data = self.label[dev_index]

        self.train_steps_num = self.train_label.shape[0]

        return self.mini_batch_data(self.train_data, self.train_label, self.batch_size)

    def batch_dev_data(self, dev_batch_size=None):
        if dev_batch_size is None:
            dev_batch_size = self.batch_size

        self.dev_steps_num = self.dev_label.shape[0]

        return self.mini_batch_data(self.dev_data, self.dev_label, dev_batch_size)

    def batch_test_data(self, test_batch_size=None):
        assert self.test_samples is not None
        if test_batch_size is None:
            test_batch_size = self.batch_size

        self.test_steps_num = self.test_label.shape[0]

        return self.mini_batch_data(self.test_data, self.test_label, test_batch_size)


def PRF(label: np.ndarray, predict: np.ndarray):
    matrix = np.zeros((3, 3), dtype=np.int32)

    label_array = [(label == i).astype(np.int32) for i in range(3)]
    predict_array = [(predict == i).astype(np.int32) for i in range(3)]

    for i in range(3):
        for j in range(3):
            matrix[i, j] = label_array[i][predict_array[j] == 1].sum()

    # (1) confusion matrix
    label_sum = matrix.sum(axis=1, keepdims=True)  # shape: (3, 1)
    matrix = np.concatenate([matrix, label_sum], axis=1)  # or: matrix = np.c_[matrix, label_sum]
    predict_sum = matrix.sum(axis=0, keepdims=True)  # shape: (1, 4)
    matrix = np.concatenate([matrix, predict_sum], axis=0)  # or: matrix = np.r_[matrix, predict_sum]

    # (2) accuracy
    temp = 0
    for i in range(3):
        temp += matrix[i, i]
    accuracy = temp / matrix[3, 3]

    # (3) precision (P), recall (R), and F1-score for each label
    P = np.zeros((3,))
    R = np.zeros((3,))
    F = np.zeros((3,))

    for i in range(3):
        P[i] = matrix[i, i] / matrix[3, i]
        R[i] = matrix[i, i] / matrix[i, 3]
        F[i] = 2 * P[i] * R[i] / (P[i] + R[i])

    # # (4) micro-averaged P, R, F1
    # micro_P = micro_R = micro_F = accuracy

    # (5) macro-averaged P, R, F1
    macro_P = P.mean()
    macro_R = R.mean()
    macro_F = 2 * macro_P * macro_R / (macro_P + macro_R)

    return {'matrix': matrix, 'acc': accuracy,
            'each_prf': [P, R, F], 'macro_prf': [macro_P, macro_R, macro_F]}


def print_metrics(metrics, metrics_type, save_dir=None):
    matrix = metrics['matrix']
    acc = metrics['acc']
    each_prf = [[v * 100 for v in prf] for prf in zip(*metrics['each_prf'])]
    macro_prf = [v * 100 for v in metrics['macro_prf']]
    loss = metrics['loss']
    epoch = metrics['epoch']

    lines = ['------------  Epoch {0}, loss {1:.4f}  -----------'.format(epoch, loss),
             'Confusion matrix:',
             '{0:>6}|{1:>6}|{2:>6}|{3:>6}|<-- classified as'.format(' ', 'Good', 'Pot.', 'Bad'),
             '------|--------------------|{0:>6}'.format('-SUM-'),
             '{0:>6}|{1:>6}|{2:>6}|{3:>6}|{4:>6}'.format('Good', *matrix[0].tolist()),
             '{0:>6}|{1:>6}|{2:>6}|{3:>6}|{4:>6}'.format('Pot.', *matrix[1].tolist()),
             '{0:>6}|{1:>6}|{2:>6}|{3:>6}|{4:>6}'.format('Bad', *matrix[2].tolist()),
             '------|--------------------|------',
             '{0:>6}|{1:>6}|{2:>6}|{3:>6}|{4:>6}'.format('-SUM-', *matrix[3].tolist()),
             '\nAccuracy = {0:6.2f}%\n'.format(acc * 100),
             'Results for the individual labels:',
             '\t{0:>6}: P ={1:>6.2f}%, R ={2:>6.2f}%, F ={3:>6.2f}%'.format('Good', *each_prf[0]),
             '\t{0:>6}: P ={1:>6.2f}%, R ={2:>6.2f}%, F ={3:>6.2f}%'.format('Pot.', *each_prf[1]),
             '\t{0:>6}: P ={1:>6.2f}%, R ={2:>6.2f}%, F ={3:>6.2f}%'.format('Bad', *each_prf[2]),
             '\n<<Official Score>>Macro-averaged result:',
             'P ={0:>6.2f}%, R ={1:>6.2f}%, F ={2:>6.2f}%'.format(*macro_prf),
             '--------------------------------------------------\n']
    [print(line) for line in lines]

    if save_dir is not None:
        with open(os.path.join(save_dir, "{}_logs.log".format(metrics_type)), 'a') as fw:
            [fw.write(line + '\n') for line in lines]



