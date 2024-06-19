import os
import pickle

script = {"CHINESE": ['CASIA_CHINESE', 'Chinese_content.pkl'],
          'JAPANESE': ['TUATHANDS_JAPANESE', 'Japanese_content.pkl'],
          "ENGLISH": ['CASIA_ENGLISH', 'English_content.pkl']
          }

root = '../data'
dataset = 'CHINESE'


def create_test_pkl():
    pass


def create_train_pkl():
    pass


if __name__ == '__main__':
    data_path = os.path.join(root, script[dataset][0])
    all_writer = pickle.load(open(os.path.join(data_path, 'writer_dict.pkl'), 'rb'))
    train_writer_pkl_name = []
    test_writer_pkl_name = []
    for pkl_name in all_writer['train_writer']:
        train_writer_pkl_name.append(pkl_name.split('.')[0])
    for pkl_name in all_writer['test_writer']:
        test_writer_pkl_name.append(pkl_name.split('.')[0])
    print(test_writer_pkl_name)
