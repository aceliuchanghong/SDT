import os

from utils.util import write_pkl


def get_files(folder_path, suffix):
    jpg_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(suffix):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                jpg_files.append(file_path)
    return jpg_files


# 示例用法
folder_path = '../style_samples'
suffix = '.jpg'
file_path = '.'
file_name = 'test.pkl'
jpg_files = get_files(folder_path, suffix)
write_pkl(file_path, file_name, jpg_files, 2)
