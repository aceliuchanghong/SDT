import os
import pickle
import matplotlib.pyplot as plt
import lmdb

script = {"CHINESE": ['CASIA_CHINESE', 'Chinese_content.pkl'],
          'JAPANESE': ['TUATHANDS_JAPANESE', 'Japanese_content.pkl'],
          "ENGLISH": ['CASIA_ENGLISH', 'English_content.pkl']
          }

root = '../data'
dataset = 'CHINESE'
num_img = 15

if __name__ == '__main__':
    """
    C031-f.pkl 文件结构
    item['img'],item['label']
    """
    data_path = os.path.join(root, script[dataset][0])
    content = pickle.load(open(os.path.join(data_path, script[dataset][1]), 'rb'))
    for _ in content:
        print(_, content[_])
        plt.imshow(content[_], cmap='gray')
        plt.show()
        break
    char_dict = pickle.load(open(os.path.join(data_path, 'character_dict.pkl'), 'rb'))
    print(char_dict)
    all_writer = pickle.load(open(os.path.join(data_path, 'writer_dict.pkl'), 'rb'))
    print(all_writer)
    lmdb_path = os.path.join(data_path, 'test')
    img_path = os.path.join(data_path, 'test_style_samples')
    lmdb = lmdb.open(lmdb_path, max_readers=8, readonly=True, lock=False, readahead=False, meminit=False)
    print(lmdb.begin(write=False).get('num_sample'.encode('utf-8')).decode())

    test_style_samples01 = pickle.load(open(os.path.join(data_path, 'test_style_samples', 'C031-f.pkl'), 'rb'))
    print(len(test_style_samples01))
    i = 0
    for item in test_style_samples01:
        # print(item,item['img'],item['label'])
        """or
        cv2.imshow("aa", item['img'])
        cv2.waitKey(0)
        """
        plt.imshow(item['img'], cmap='gray')
        plt.show()
        i += 1
        if i > 13:
            break
