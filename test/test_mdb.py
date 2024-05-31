import os
import lmdb
import pickle

from data_loader.loader import ScriptDataset

script = {"CHINESE": ['CASIA_CHINESE', 'Chinese_content.pkl'],
          'JAPANESE': ['TUATHANDS_JAPANESE', 'Japanese_content.pkl'],
          "ENGLISH": ['CASIA_ENGLISH', 'English_content.pkl']
          }

root = '../data'
dataset = 'CHINESE'
num_img = 15
index = 1000
max_len = 150
if __name__ == '__main__':
    data_path = os.path.join(root, script[dataset][0])
    lmdb_path = os.path.join(data_path, 'test')
    print(lmdb_path)
    lmdb = lmdb.open(lmdb_path, max_readers=8, readonly=True, lock=False, readahead=False, meminit=False)
    print(lmdb.begin(write=False).get('num_sample'.encode('utf-8')).decode())
    with lmdb.begin(write=False) as txn:
        num_sample = int(txn.get('num_sample'.encode('utf-8')).decode())

        indexes = list(range(0, num_sample))
        # print("01:", indexes)

        # indexes2 = []
        # for i in range(num_sample):
        #     data_id = str(i).encode('utf-8')
        #     data_byte = txn.get(data_id)
        #     coords = pickle.loads(data_byte)['coordinates']
        #     if len(coords) < max_len:
        #         indexes2.append(i)
        # print("02:", indexes2)

        index = indexes[index]
        data = pickle.loads(txn.get(str(index).encode('utf-8')))
        tag_char, coords, fname = data['tag_char'], data['coordinates'], data['fname']
        print("tag_char: {}\ncoords_shape: {}\nfname: {}".format(tag_char, coords.shape, fname))
        # print("tag_char, coords, fname:", tag_char, coords, fname)

    sd = ScriptDataset(root='../data', dataset='CHINESE_TEST', is_train=False)
    print(sd.__getitem__(55))
