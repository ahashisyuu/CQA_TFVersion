import os
import re
import string
import spacy
import json
import numpy as np
import pickle as pkl
import xml.etree.ElementTree as ET

from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences

nlp = spacy.load('en')
relevance2label = {'Good': 0, 'PotentiallyUseful': 1, 'Bad': 2}
char2index = {key: value+1 for value, key in enumerate(string.ascii_letters + string.digits + string.punctuation)}


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


def SemEval15_sample(root):
    for question in root.findall('Question'):
        qsubject = question.find('QSubject').text
        qbody = question.find('QBody').text
        for relcomment in question.findall('Comment'):
            cID = relcomment.get('CID')
            Relevance = relcomment.get('CGOLD')
            cTEXT = relcomment.find('CBody').text
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
            yield [cID, qsubject, qbody, cTEXT, Relevance]


def SemEval16or17_sample(root):
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
            yield [cID, qsubject, qbody, cTEXT, Relevance]


def get_samples(filename):
    root = ET.parse(filename).getroot()

    if '15' in os.path.basename(filename):
        return SemEval15_sample(root)
    else:
        return SemEval16or17_sample(root)


def tokenizer(text, need_punct=False):
    if need_punct:
        return [word.orth_ for word in nlp(text) if not word.is_space]
    else:
        return [word.orth_ for word in nlp(text) if not word.is_punct or not word.is_space]


def char_tokenizer(text):
    char_text = []
    for token in text:
        token_ = []
        for c in token:
            if c not in char2index:
                char2index[c] = len(char2index)+1
            token_.append(char2index[c])
        char_text.append(token_)
    return char_text


def count_word_number(word_count, text):
    for token in text:
        if token in word_count:
            word_count[token] += 1
        else:
            word_count[token] = 1


def is_website(w):
    if ('http' in w and '/' in w) or ('.com' in w and '@' not in w):
        return True
    else:
        return False


def is_email(w):
    if '@' in w and ('.com' in w or '.cn' in w):
        return True
    else:
        return False


number_li = [str(i) for i in range(10)]
atperson_re = re.compile(r'^@\w+$')


def is_number(w):
    if re.search(r'[^\d, ]', w):
        return False
    return True


def is_time(w):
    if re.search(r'[^\d\.]', w):
        return False
    return True


def is_atperson(w):
    if atperson_re.match(w):
        return True
    else:
        return False


def check_word(text, word_vector_keys):
    new_text = []
    for word in text:
        w = word.lower().strip()

        if is_website(w):
            new_text.append('<url>')
        elif is_email(w):
            new_text.append('<email>')
        elif is_number(w):
            new_text.append('<number>')
        elif is_time(w):
            new_text.append('<time>')
        elif is_atperson(w):
            new_text.append('<atperson>')
        elif w in word_vector_keys:
            new_text.append(w)
        else:
            # # remove punctuation
            # w = remove_punct(w)
            # fine_text = []
            # for i, w_fine in enumerate(w):
            #     if is_number(w_fine):
            #         fine_text.append('<number>')
            #     elif w_fine in word_vector_keys:
            #         fine_text.append(w_fine)
            #     elif i>0 and fine_text[i-1] is '<unk>':
            #         fine_text.append('<unk>')
            new_text.append('<unk>')
    if len(new_text) == 0:
        return ['<blank>']
    return new_text


def process_sample(sample, word_count, char_max_len, word_vector_keys, need_punct=False):
    """
    样本形式为 ‘[cID, qsubject, qbody, cTEXT, Relevance]’ 的列表
    :param sample:
    :return:
    """
    cID, qsubject, qbody, cTEXT, Relevance = sample
    qsubject = tokenizer(qsubject, need_punct=need_punct)
    qbody = tokenizer(qbody, need_punct=need_punct)
    cTEXT = tokenizer(cTEXT, need_punct=need_punct)

    qsubject_sent = check_word(qsubject, word_vector_keys)
    qbody_sent = check_word(qbody, word_vector_keys)
    cTEXT_sent = check_word(cTEXT, word_vector_keys)

    count_word_number(word_count, qsubject_sent)
    count_word_number(word_count, qbody_sent)
    count_word_number(word_count, cTEXT_sent)

    char_qsubject = char_tokenizer(qsubject)
    char_qbody = char_tokenizer(qbody)
    char_cTEXT = char_tokenizer(cTEXT)

    char_qsubject = pad_sequences(char_qsubject, maxlen=char_max_len, padding='post', truncating='post')
    char_qbody = pad_sequences(char_qbody, maxlen=char_max_len, padding='post', truncating='post')
    char_cTEXT = pad_sequences(char_cTEXT, maxlen=char_max_len, padding='post', truncating='post')

    return cID, qsubject_sent, qbody_sent, cTEXT_sent, char_qsubject, char_qbody, char_cTEXT, Relevance


def replace2index(sample, word2index):
    cID, qsubject, qbody, cTEXT, char_qsubject, char_qbody, char_cTEXT, Relevance = sample
    qsubject = [word2index[token] for token in qsubject]
    qbody = [word2index[token] for token in qbody]
    cTEXT = [word2index[token] for token in cTEXT]
    label = [0, 0, 0]
    label[relevance2label[Relevance]] = 1

    return cID, qsubject, qbody, cTEXT, char_qsubject, char_qbody, char_cTEXT, np.asarray(label)


def preprocessing(filepath, savepath, char_max_len, need_punct=False, glove_filename='glove.6B.300d.txt'):
    listname = os.listdir(filepath)
    word_count = {}
    all_samples = {}
    print('-------------------------------------')
    print('\t开始处理样本并产生计数词典')
    glove_vector_file = os.path.join(savepath, glove_filename)
    word_vector = load_glove(glove_vector_file)
    for name in listname:
        print('\n\t\t处理文件：%s\n' % name)
        filename = os.path.join(filepath, name)
        samples = get_samples(filename)
        samples = [process_sample(sample, word_count,
                                  char_max_len, word_vector.keys(),
                                  need_punct=need_punct)
                   for sample in tqdm(samples)]
        all_samples[name] = samples

    print('\n--------------------------------------')
    print('\tsave char2index')
    print('\t\tthe number of charactor：%d' % len(char2index))
    with open(os.path.join(savepath, 'char2index.json'), 'w') as fw:
        json.dump(char2index, fw)

    print('\n--------------------------------------')
    print('\t获得word2index并保存')
    word_count_ = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    with open(os.path.join(savepath, 'wordcount.json'), 'w') as fw:
        json.dump(word_count_, fw)

    word2index = {word: index + 1 for index, (word, _) in enumerate(word_count_)}
    print('\t\t总词数为：%d' % len(word2index))
    with open(os.path.join(savepath, 'word2index.json'), 'w') as fw:
        json.dump(word2index, fw)

    print('\n--------------------------------------')
    print('\t作嵌入矩阵并保存\n')
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
    preprocessing('../rawData', '../data', 16, need_punct=True, glove_filename='glove.6B.300d.txt')




