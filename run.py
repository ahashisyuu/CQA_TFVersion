import argparse
import json
import pickle as pkl

from config import Config
from preprocessing.preprocessing import preprocessing
from utils import BatchDatasets
from models.QCN import QCN


config = Config()
parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--train_list', type=list, default=['15train'])
parser.add_argument('--dev_list', type=list, default=['15dev'])
parser.add_argument('--test_list', type=list, default=['15test'])

parser.add_argument('--lr', type=float, default=config.lr)
parser.add_argument('--dropout', type=float, default=config.dropout)
parser.add_argument('--max_len', type=int, default=config.max_len)
parser.add_argument('--char_max_len', type=int, default=config.char_max_len)
parser.add_argument('--epochs', type=int, default=config.epochs)
parser.add_argument('--batch_size', type=int, default=config.batch_size)
parser.add_argument('--char_dim', type=int, default=config.char_dim)
parser.add_argument('--l2_weight', type=int, default=config.l2_weight)

parser.add_argument('--patience', type=int, default=config.patience)
parser.add_argument('--k_fold', type=int, default=config.k_fold)
parser.add_argument('--categories_num', type=int, default=config.categories_num)
parser.add_argument('--period', type=int, default=config.period)

parser.add_argument('--need_punct', type=bool, default=config.need_punct)
parser.add_argument('--need_shuffle', type=bool, default=config.need_shuffle)
parser.add_argument('--use_char_level', type=bool, default=config.use_char_level)
parser.add_argument('--load_best_model', type=bool, default=config.load_best_model)

parser.add_argument('--model_dir', type=str, default=config.model_dir)
parser.add_argument('--log_dir', type=str, default=config.log_dir)
parser.add_argument('--glove_file', type=str, default=config.glove_filename)


def run(args):
    if args.mode == 'prepare':
        preprocessing('./rawData', './data', need_punct=args.need_punct,
                      char_max_len=args.char_max_len, glove_filename=args.glove_file)
    else:
        # loading preprocessed data
        with open('./data/dataset.pkl', 'rb') as fr, \
             open('./data/embedding_matrix.pkl', 'rb') as fr_embed, \
             open('./data/char2index.json', 'r') as fr_char:
            data = pkl.load(fr)
            embedding_matrix = pkl.load(fr_embed)
            char2index = json.load(fr_char)

        train_samples = [data[k + '.xml'] for k in args.train_list]
        dev_samples = [data[k + '.xml'] for k in args.dev_list]
        test_samples = [data[k + '.xml'] for k in args.test_list]

        all_data = BatchDatasets(args.max_len, args.char_max_len, need_shuffle=args.need_shuffle,
                                 batch_size=args.batch_size, k_fold=args.k_fold, categories_num=args.categories_num,
                                 train_samples=train_samples, dev_samples=dev_samples, test_samples=test_samples)

        model = QCN(embedding_matrix=embedding_matrix, args=args, char_num=len(char2index))

        if args.mode == 'train':
            model.train(all_data, args)
        elif args.mode == 'test':
            model.test(all_data, args)


if __name__ == '__main__':
    run(parser.parse_args())

