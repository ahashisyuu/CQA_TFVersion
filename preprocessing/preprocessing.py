import os
import spacy
import json
import numpy as np
import pickle as pkl
import xml.etree.ElementTree as ET

from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences

nlp = spacy.load('en')
relevance2label = {'Good': 0, 'PotentiallyUseful': 1, 'Bad': 2}


def load_glove(filename):
    print('\nload word dictionary starting!')
    word_dic = {}
    with open(filename, encoding='utf-8') as fr:
        lines = [line for line in fr]
        for line in lines:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_dic[word] = coefs

    print('load word dictionary ending!\n')

    return word_dic


def get_samples(filename):
    root = ET.parse(filename).getroot()

    for thread in root.findall('Thread'):
        question = thread.find('RelQuestion')
        qsubject = question.find('RelQSubject').text
        qbody = question.find('RelQBody').text
        for relcomment in thread.findall('RelComment'):
            cID = relcomment.get('RELC_ID')
            Relevance = relcomment.get('RELC_RELEVANCE2RELQ')
            cTEXT = relcomment.find('RelCText').text
            if qsubject is None:
                print('----------  qsubject None, cID: %s  ------- ' % cID)
                continue
            if qbody is None:
                print('----------  qbody None, cID: %s  ------- ' % cID)
                continue
            if cTEXT is None:
                print('----------  cTEXT None, cID: %s  ------- ' % cID)
                continue
            if Relevance is None:
                print('----------  Relevance None, cID: %s  ------- ' % cID)
                continue
            assert qsubject is not None
            assert qbody is not None
            assert cTEXT is not None
            assert Relevance is not None
            yield [cID, qsubject, qbody, cTEXT, Relevance]


def tokenizer(text, need_punct=False):
    if need_punct:
        return [word.orth_ for word in nlp(text)]
    else:
        return [word.orth_ for word in nlp(text) if not word.is_punct | word.is_space]


def char_tokenizer(text):
    return [np.fromstring(token, dtype=np.uint8).astype(dtype=np.int32) for token in text]


def count_word_number(word_count, text):
    for token in text:
        if token in word_count:
            word_count[token] += 1
        else:
            word_count[token] = 1


def check_word(text):
    for word in text:
        if len(word.split(" ")) != 1:
            print('\t', word)


def process_sample(sample, word_count, char_max_len):
    """
    样本形式为 ‘[cID, qsubject, qbody, cTEXT, Relevance]’ 的列表
    :param sample:
    :return:
    """
    cID, qsubject, qbody, cTEXT, Relevanc = sample
    qsubject = tokenizer(qsubject)
    qbody = tokenizer(qbody)
    cTEXT = tokenizer(cTEXT)

    check_word(qsubject)
    check_word(qbody)
    check_word(cTEXT)

    count_word_number(word_count, qsubject)
    count_word_number(word_count, qbody)
    count_word_number(word_count, cTEXT)

    char_qsubject = char_tokenizer(qsubject)
    char_qbody = char_tokenizer(qbody)
    char_cTEXT = char_tokenizer(cTEXT)

    char_qsubject = pad_sequences(char_qsubject, maxlen=char_max_len, padding='post', truncating='post')
    char_qbody = pad_sequences(char_qbody, maxlen=char_max_len, padding='post', truncating='post')
    char_cTEXT = pad_sequences(char_cTEXT, maxlen=char_max_len, padding='post', truncating='post')

    return cID, qsubject, qbody, cTEXT, char_qsubject, char_qbody, char_cTEXT, Relevanc


def replace2index(sample, word2index):
    cID, qsubject, qbody, cTEXT, char_qsubject, char_qbody, char_cTEXT, Relevanc = sample
    qsubject = [word2index[token] for token in qsubject]
    qbody = [word2index[token] for token in qbody]
    cTEXT = [word2index[token] for token in cTEXT]
    label = [0, 0, 0]
    label[relevance2label[Relevanc]] = 1

    return cID, qsubject, qbody, cTEXT, char_qsubject, char_qbody, char_cTEXT, np.asarray(label)


def preprocessing(filepath, savepath, char_max_len):
    listname = os.listdir(filepath)
    word_count = {}
    all_samples = {}
    print('-------------------------------------')
    print('\t开始处理样本并产生计数词典')
    for name in listname:
        print('\n\t\t处理文件：%s\n' % name)
        filename = os.path.join(filepath, name)
        samples = get_samples(filename)
        samples = [process_sample(sample, word_count, char_max_len) for sample in tqdm(samples)]
        all_samples[name] = samples

    print('\n--------------------------------------')
    print('\t获得word2index并保存')
    word_count_ = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    word2index = {word: index + 1 for index, (word, _) in enumerate(word_count_)}
    print('\t\t总词数为：%d' % len(word2index))
    with open(os.path.join(savepath, 'word2index.json'), 'w') as fw:
        json.dump(word2index, fw)

    print('\n--------------------------------------')
    print('\t作嵌入矩阵并保存\n')
    word_vector = load_glove(os.path.join(savepath, 'glove.6B.300d.txt'))
    embedding_matrix = np.zeros((len(word2index) + 1, 300))
    for word, index in word2index.items():
        if word in word_vector:
            embedding_matrix[index] = word_vector[word]
    with open(os.path.join(savepath, 'embedding_matrix.pkl'), 'wb') as fw:
        pkl.dump(embedding_matrix, fw)

    print('--------------------------------------')
    print('\t用word2index替换所有样本里的单词\n')
    listname = os.listdir(filepath)
    for name in listname:
        print('\n\t\t处理数据集名称：%s\n' % name)
        samples = all_samples[name]
        all_samples[name] = [replace2index(sample, word2index) for sample in tqdm(samples)]

    with open(os.path.join(savepath, 'dataset.pkl'), 'wb') as fw:
        pkl.dump(all_samples, fw)


if __name__ == '__main__':
    preprocessing('../rawData', '../data', 16)




