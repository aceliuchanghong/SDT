import os
from utils.judge_font import get_files
from utils.util import write_pkl

# 示例用法
folder_path = '../style_samples'
suffix = '.jpg'
file_path = '.'
file_name = 'test.pkl'
jpg_files = get_files(folder_path, suffix)
write_pkl(file_path, file_name, jpg_files, 2)
