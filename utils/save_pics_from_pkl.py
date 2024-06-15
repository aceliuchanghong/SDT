import os
from tqdm import tqdm
from utils.check_db import excute_sqlite_sql
from utils.config import table_pkl_path_sql
import pickle
from PIL import Image
import numpy as np

base_dir = '../data/CASIA_CHINESE'
save_suffix = ".png"
save_pics_path = 'suit_pics'
set_nums = 1000


def save_pics_from_pkl(pkl_path, save_pics_path_dir):
    pkl_relative_path = base_dir + '/' + pkl_path
    this_pkl_file = pickle.load(open(pkl_relative_path, 'rb'))
    seed = pkl_path.split("/")[-1].split(".")[-2]
    for item in this_pkl_file:
        img_array = item['img'].astype(np.uint8)
        name = item['label']
        pil_image = Image.fromarray(img_array, mode='L')
        save_path = save_pics_path_dir + "/" + seed
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_pics_path = save_path + "/" + name + save_suffix
        # print(save_path, save_pics_path)
        pil_image.save(save_pics_path)


def get_pkl_path_list():
    picIt = excute_sqlite_sql(table_pkl_path_sql, ('1',), False)
    ans = []
    for _ in picIt:
        name = _[0].split("/")[-1]
        path = _[0].split("/")[-2]
        ans.append(path + "/" + name)
    return ans


if __name__ == '__main__':
    pkl_path = get_pkl_path_list()
    index = 0
    set_num = set_nums if len(pkl_path) > set_nums else len(pkl_path)
    for i in tqdm(pkl_path, desc="Processing", total=set_num):
        save_pics_from_pkl(i, save_pics_path)
        if index >= set_num - 1:
            break
        index += 1
