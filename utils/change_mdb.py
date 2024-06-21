from utils.util import writeCache
import os
import lmdb
import pickle

script = {"CHINESE": ['CASIA_CHINESE', 'Chinese_content.pkl'],
          'JAPANESE': ['TUATHANDS_JAPANESE', 'Japanese_content.pkl'],
          "ENGLISH": ['CASIA_ENGLISH', 'English_content.pkl']
          }

root = '../data'
dataset = 'CHINESE'
index = 10001

if __name__ == '__main__':
    data_path = os.path.join(root, script[dataset][0])
    lmdb_path = os.path.join(data_path, 'test')
    lmdb = lmdb.open(lmdb_path, max_readers=8, readonly=True, lock=False, readahead=False, meminit=False)
    num_sample = lmdb.begin(write=False).get('num_sample'.encode('utf-8')).decode()
    print('num_sample:', lmdb.begin(write=False).get('num_sample'.encode('utf-8')).decode())
    with lmdb.begin(write=False) as txn:
        for i in range(int(num_sample)):
            data = pickle.loads(txn.get(str(i).encode('utf-8')))
            tag_char, coords, fname = data['tag_char'], data['coordinates'], data['fname']
            print("tag_char: {}\ncoords_shape: {}\nfname: {}".format(tag_char, coords.shape, fname))
            print("coords:\n", coords)
            break
    test_cache = {}
    # data = {'coordinates': pred, 'writer_id': writer_id[i].item(),
    #         'character_id': character_id[i].item(), 'coords_gt': coord}
    # test_cache['num_sample'.encode('utf-8')] = str(num_count).encode()
    # data_byte = pickle.dumps(data)
    # writeCache(test_env, test_cache)
