import os
import pickle
import lmdb

script = {"CHINESE": ['CASIA_CHINESE', 'Chinese_content.pkl'],
          'JAPANESE': ['TUATHANDS_JAPANESE', 'Japanese_content.pkl'],
          "ENGLISH": ['CASIA_ENGLISH', 'English_content.pkl']
          }

root = '../data'
dataset = 'CHINESE'
num_img = 15

if __name__ == '__main__':
    data_path = os.path.join(root, script[dataset][0])
    content = pickle.load(open(os.path.join(data_path, script[dataset][1]), 'rb'))
    # print(len(content))
    # print(content['列'])
    char_dict = pickle.load(open(os.path.join(data_path, 'character_dict.pkl'), 'rb'))
    # print(len(char_dict))
    # print(char_dict[99])
    all_writer = pickle.load(open(os.path.join(data_path, 'writer_dict.pkl'), 'rb'))
    writer_dict = all_writer['test_writer']
    # print(all_writer)
    lmdb_path = os.path.join(data_path, 'test')
    img_path = os.path.join(data_path, 'test_style_samples')
    lmdb = lmdb.open(lmdb_path, max_readers=8, readonly=True, lock=False, readahead=False, meminit=False)
    print(lmdb.begin(write=False).get('num_sample'.encode('utf-8')).decode())
