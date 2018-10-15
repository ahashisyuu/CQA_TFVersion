import os
import random
import numpy as np

from prettytable import PrettyTable
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.sequence import pad_sequences
from OfficialScorer import metrics


class BatchDatasets:
    def __init__(self,  max_len, char_max_len,
                 need_shuffle=False, batch_size=64, k_fold=0, categories_num=3,
                 train_samples: list=None, dev_samples: list=None, test_samples=None):
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
        self.categories_num = categories_num
        self.train_steps_num = 0
        self.dev_steps_num = 0
        self.test_steps_num = 0
        self.train_data = None
        self.train_label = None
        self.cweight = []
        self.dev_data = None
        self.dev_label = None
        self.dev_cID = None

        self.temp_array = np.asarray([[1, 0], [0, 1]])

        if train_samples is not None and dev_samples is not None:
            if k_fold > 1:  # merge train data and dev data
                self.train_samples += self.dev_samples
                self.train_samples = [trans for trans in zip(*self.train_samples)]
                self.data_cID = np.asarray(self.train_samples[0])
                self.data = [np.asarray(trans) for trans in self.train_samples[1:-1]]
                self.label = self.label_tranformer(np.asarray(self.train_samples[-1]))
                skf = StratifiedKFold(n_splits=k_fold, random_state=0)
                self.index_list = [index for index in skf.split(self.data[0], self.label.argmax(axis=1))]

            else:
                if self.need_shuffle:
                    self.shuffle()
                self.train_samples = [trans for trans in zip(*self.train_samples)]
                self.train_data = self.train_samples[1:-1]
                self.train_label = self.label_tranformer(np.asarray(self.train_samples[-1]))

                self.dev_samples = [trans for trans in zip(*self.dev_samples)]
                self.dev_cID = self.dev_samples[0]
                self.dev_data = self.dev_samples[1:-1]
                self.dev_label = self.label_tranformer(np.asarray(self.dev_samples[-1]))

        if test_samples is not None:
            self.test_samples = [trans for trans in zip(*self.test_samples)]
            self.test_cID = self.test_samples[0]
            self.test_data = self.test_samples[1:-1]
            self.test_label = self.label_tranformer(np.asarray(self.test_samples[-1]))

    @staticmethod
    def processing_sample(samples_list):
        if samples_list is None or len(samples_list) == 0:
            return None

        new_samples = []
        for samples in samples_list:
            new_samples += samples

        return new_samples

    def shuffle(self):
        random.shuffle(self.train_samples)

    def get_len(self, e):
        return min(len(max(e, key=len)), self.max_len)

    @staticmethod
    def pad_sentence(e, maxlen):
        return pad_sequences(e, padding='post', truncating='post', maxlen=maxlen)

    def padding(self, batch_data):
        assert len(batch_data) == 6
        cur_max_len = [self.get_len(e) for e in batch_data[0:3]]*2
        return [self.pad_sentence(e, maxlen=l) for e, l in zip(batch_data, cur_max_len)]

    def mini_batch_data(self, data, label, batch_size):
        data_size = label.shape[0]
        for batch_start in np.arange(0, data_size, batch_size):
            batch_data = [e[batch_start:batch_start+batch_size]
                          for e in data]
            batch_label = label[batch_start:batch_start+batch_size]

            yield self.padding(batch_data) + [batch_label]

    def compute_class_weight(self, train_label):
        label = train_label.argmax(axis=1)
        number = [(label == i).astype('int32').sum() for i in range(self.categories_num)]

        max_num = max(number)
        min_num = min(number)
        median = max_num
        for n in number:
            if n != max_num and n != min_num:
                median = n

        return [median/n for n in number]

    def batch_train_data(self, batch_size=None, fold_num=None):
        if self.k_fold > 1:
            train_index, dev_index = self.index_list[fold_num]
            self.train_data = [list(element[train_index]) for element in self.data]
            self.train_label = self.label[train_index]
            self.dev_data = [list(element[dev_index]) for element in self.data]
            self.dev_label = self.label[dev_index]
            self.dev_cID = self.data_cID[dev_index].tolist()

        self.train_steps_num = self.train_label.shape[0]
        self.cweight = self.compute_class_weight(self.train_label)

        batch_size = self.batch_size if batch_size is None else batch_size
        return self.mini_batch_data(self.train_data, self.train_label, batch_size)

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

    def label_tranformer(self, batch_label: np.ndarray):
        # label 0 stands for 'Good', label 1 stands for 'Bad'
        if self.categories_num == 3:
            return batch_label
        new_label = (batch_label.argmax(axis=1) != 0).astype('int32')
        return self.temp_array[new_label]


def PRF(label: np.ndarray, predict: np.ndarray):
    categories_num = label.max() + 1
    matrix = np.zeros((categories_num, categories_num), dtype=np.int32)

    label_array = [(label == i).astype(np.int32) for i in range(categories_num)]
    predict_array = [(predict == i).astype(np.int32) for i in range(categories_num)]

    for i in range(categories_num):
        for j in range(categories_num):
            matrix[i, j] = label_array[i][predict_array[j] == 1].sum()

    # (1) confusion matrix
    label_sum = matrix.sum(axis=1, keepdims=True)  # shape: (ca_num, 1)
    matrix = np.concatenate([matrix, label_sum], axis=1)  # or: matrix = np.c_[matrix, label_sum]
    predict_sum = matrix.sum(axis=0, keepdims=True)  # shape: (1, ca_num+1)
    matrix = np.concatenate([matrix, predict_sum], axis=0)  # or: matrix = np.r_[matrix, predict_sum]

    # (2) accuracy
    temp = 0
    for i in range(categories_num):
        temp += matrix[i, i]
    accuracy = temp / matrix[categories_num, categories_num]

    # (3) precision (P), recall (R), and F1-score for each label
    P = np.zeros((categories_num,))
    R = np.zeros((categories_num,))
    F = np.zeros((categories_num,))

    for i in range(categories_num):
        P[i] = matrix[i, i] / matrix[categories_num, i]
        R[i] = matrix[i, i] / matrix[i, categories_num]
        F[i] = 2 * P[i] * R[i] / (P[i] + R[i]) if P[i] + R[i] > 0 else 0

    # # (4) micro-averaged P, R, F1
    # micro_P = micro_R = micro_F = accuracy

    # (5) macro-averaged P, R, F1
    macro_P = P.mean()
    macro_R = R.mean()
    macro_F = 2 * macro_P * macro_R / (macro_P + macro_R) if macro_P + macro_R else 0

    return {'matrix': matrix, 'acc': accuracy,
            'each_prf': [P, R, F], 'macro_prf': [macro_P, macro_R, macro_F]}


def print_metrics(metrics, metrics_type, save_dir=None, categories_num=3):
    matrix = metrics['matrix']
    acc = metrics['acc']
    each_prf = [[v * 100 for v in prf] for prf in zip(*metrics['each_prf'])]
    macro_prf = [v * 100 for v in metrics['macro_prf']]
    loss = metrics['loss']
    epoch = metrics['epoch']
    if categories_num == 3:
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
    else:
        lines = ['------------  Epoch {0}, loss {1:.4f}  -----------'.format(epoch, loss),
                 'Confusion matrix:',
                 '{0:>6}|{1:>6}|{2:>6}|<-- classified as'.format(' ', 'Good', 'Bad'),
                 '------|-------------|{0:>6}'.format('-SUM-'),
                 '{0:>6}|{1:>6}|{2:>6}|{3:>6}'.format('Good', *matrix[0].tolist()),
                 '{0:>6}|{1:>6}|{2:>6}|{3:>6}'.format('Bad', *matrix[1].tolist()),
                 '------|-------------|------',
                 '{0:>6}|{1:>6}|{2:>6}|{3:>6}'.format('-SUM-', *matrix[2].tolist()),
                 '\nAccuracy = {0:6.2f}%\n'.format(acc * 100),
                 'Results for the individual labels:',
                 '\t{0:>6}: P ={1:>6.2f}%, R ={2:>6.2f}%, F ={3:>6.2f}%'.format('Good', *each_prf[0]),
                 '\t{0:>6}: P ={1:>6.2f}%, R ={2:>6.2f}%, F ={3:>6.2f}%'.format('Bad', *each_prf[1]),
                 '\n<<Official Score>>Macro-averaged result:',
                 'P ={0:>6.2f}%, R ={1:>6.2f}%, F ={2:>6.2f}%'.format(*macro_prf),
                 '--------------------------------------------------\n']

    [print(line) for line in lines]

    if save_dir is not None:
        with open(os.path.join(save_dir, "{}_logs.log".format(metrics_type)), 'a') as fw:
            [fw.write(line + '\n') for line in lines]


def transferring(matrix: np.ndarray, categories_num=3):
    conf_matrix = {'true': {'true': {}, 'false': {}}, 'false': {'true': {}, 'false': {}}}
    if categories_num == 3:
        conf_matrix['true']['true'] = matrix[0, 0]
        conf_matrix['true']['false'] = matrix[0, 1] + matrix[0, 2]
        conf_matrix['false']['true'] = matrix[1, 0] + matrix[2, 0]
        conf_matrix['false']['false'] = matrix[1, 1] + matrix[1, 2] + matrix[2, 1] + matrix[2, 2]
    else:
        conf_matrix['true']['true'] = matrix[0, 0]
        conf_matrix['true']['false'] = matrix[0, 1]
        conf_matrix['false']['true'] = matrix[1, 0]
        conf_matrix['false']['false'] = matrix[1, 1]
    return conf_matrix


def get_pre(eval_id, label, score, reranking_th, ignore_noanswer):

    model_pre = {}
    for cID, relevant, s in zip(eval_id, label, score):
        relevant = 'true' if 0 == relevant else 'false'
        # Process the line from the res file.
        qid = '_'.join(cID.split('_')[0:-1])
        if qid not in model_pre:
            model_pre[qid] = []
        model_pre[qid].append((relevant, s, cID))

    # Remove questions that contain no correct answer
    if ignore_noanswer:
        for qid in model_pre.keys():
            candidates = model_pre[qid]
            if all(relevant == "false" for relevant, _, _ in candidates):
                del model_pre[qid]

    for qid in model_pre:
        # Sort by model prediction score.
        pre_sorted = model_pre[qid]
        max_score = max([score for rel, score, cid in pre_sorted])
        if max_score >= reranking_th:
            pre_sorted = sorted(pre_sorted, key=lambda x: x[1], reverse=True)

        model_pre[qid] = [rel for rel, score, aid in pre_sorted]

    return model_pre


def eval_reranker(eval_id, label, score, matrix,
                  th=10,
                  reranking_th=0.0,
                  ignore_noanswer=False,
                  categories_num=3):
    conf_matrix = transferring(matrix, categories_num)
    model_pre = get_pre(eval_id, label, score,
                        reranking_th=reranking_th, ignore_noanswer=ignore_noanswer)

    # Calculate standard P, R, F1, Acc
    acc = 1.0 * (conf_matrix['true']['true'] + conf_matrix['false']['false']) / (
                conf_matrix['true']['true'] + conf_matrix['false']['false'] + conf_matrix['true']['false'] +
                conf_matrix['false']['true'])
    p = 0
    if (conf_matrix['true']['true'] + conf_matrix['false']['true']) > 0:
        p = 1.0 * (conf_matrix['true']['true']) / (conf_matrix['true']['true'] + conf_matrix['false']['true'])
    r = 0
    if (conf_matrix['true']['true'] + conf_matrix['true']['false']) > 0:
        r = 1.0 * (conf_matrix['true']['true']) / (conf_matrix['true']['true'] + conf_matrix['true']['false'])
    f1 = 0
    if (p + r) > 0:
        f1 = 2.0 * p * r / (p + r)

    # evaluate SVM
    prec_model = metrics.recall_of_1(model_pre, th)
    acc_model = metrics.accuracy(model_pre, th)
    acc_model1 = metrics.accuracy1(model_pre, th)
    acc_model2 = metrics.accuracy2(model_pre, th)

    mrr_model = metrics.mrr(model_pre, th)
    map_model = metrics.map(model_pre, th)

    avg_acc1_model = metrics.avg_acc1(model_pre, th)

    print("")
    print("*** Official score (MAP for SYS): %5.4f" % map_model)
    print("")
    print("******************************")
    print("*** Classification results ***")
    print("******************************")
    print("")
    print("Acc = %5.4f" % acc)
    print("P   = %5.4f" % p)
    print("R   = %5.4f" % r)
    print("F1  = %5.4f" % f1)
    print("")
    print("********************************")
    print("*** Detailed ranking results ***")
    print("********************************")
    print("")
    print("SYS -- Score for the output of the tested system.")
    print("")
    print("%13s" % "SYS")
    print("MAP   : %5.4f" % map_model)
    print("AvgRec: %5.4f" % avg_acc1_model)
    print("MRR   : %6.2f" % mrr_model)

    for i, (p_model, a_model, a_model1, a_model2) in enumerate(
            zip(prec_model, acc_model, acc_model1, acc_model2), 1):
        print("REC-1@%02d: %6.2f  ACC@%02d: %6.2f  AC1@%02d: %6.2f  AC2@%02d: %4.0f" % (
              i, p_model, i, a_model, i, a_model1, i, a_model2))
    print('----------------------------------------------------------------')
    print('----------------------------------------------------------------\n')


