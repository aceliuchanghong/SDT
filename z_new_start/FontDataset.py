from torch.utils.data import Dataset
import os
import pickle
import torch
from utils.judge_font import get_files
from z_new_start.FontConfig import new_start_config


class FontDataset(Dataset):
    def __init__(self, is_train=False, is_dev=False, train_percent=0.8):
        print("preparing dataset...")

        if is_dev:
            self.config_set = 'dev'
        else:
            self.config_set = 'test'
        self.config = new_start_config

        self.content = pickle.load(open(self.config[self.config_set]['content_pkl_path'], 'rb'))
        self.char_dict = pickle.load(open(self.config[self.config_set]['character_pkl_path'], 'rb'))
        self.pic_path = self.config[self.config_set]['z_pic_pkl_path']
        self.coordinate_path = self.config[self.config_set]['z_coordinate_pkl_path']

        coors_pkl_list_all = get_files(self.coordinate_path, '.pkl')
        pics_pkl_list_all = get_files(self.pic_path, '.pkl')

        self.can_be_used_font = []
        for i, font_pic_pkl in enumerate(pics_pkl_list_all):
            font_name = os.path.basename(font_pic_pkl).split('.')[0]
            for coors_pkl in coors_pkl_list_all:
                if font_name == os.path.basename(coors_pkl).split('.')[0]:
                    self.can_be_used_font.append(font_name)

        self.font_data = []
        for i, font_name in enumerate(self.can_be_used_font):
            font_pic_pkl = os.path.join(self.pic_path, font_name + '.pkl')
            font_coors_pkl = os.path.join(self.coordinate_path, font_name + '.pkl')

            font_pics_list = pickle.load(open(font_pic_pkl, 'rb'))
            font_coors_list = pickle.load(open(font_coors_pkl, 'rb'))

            for pic in font_pics_list:
                char = pic['label']
                if char in font_coors_list:
                    self.font_data.append(
                        (i, font_name, pic['label'], pic['img'], font_coors_list[char])
                    )

        train_size = int(len(self.font_data) * train_percent)
        if is_train:
            self.font_data = self.font_data[:train_size]
        else:
            self.font_data = self.font_data[train_size:]

        self.num_sample = len(self.font_data)
        print("dataset is ready...")

    def __getitem__(self, idx):
        nums, font_name, label, img, coors = self.font_data[idx]
        return {'nums': nums, 'font_name': font_name, 'label': label, 'image': img, 'coordinates': coors}

    def __len__(self):
        return self.num_sample

    def collate_fn_(self, batch_data):
        # 提取各个字段
        nums = [item['nums'] for item in batch_data]
        font_names = [item['font_name'] for item in batch_data]
        labels = [item['label'] for item in batch_data]
        images = [item['image'] for item in batch_data]
        coordinates = [item['coordinates'] for item in batch_data]

        # 将图像堆叠成一个张量
        images = torch.stack([torch.tensor(img, dtype=torch.float32) for img in images])

        # 将 nums 转换为张量
        nums = torch.tensor(nums, dtype=torch.int64)

        # 将 labels 转换为适当的形式
        labels = [torch.tensor(ord(label), dtype=torch.int64) for label in labels]  # 假设 label 是单个字符
        labels = torch.stack(labels)

        # 将 coordinates 转换为张量列表
        coordinates = [torch.tensor(coord, dtype=torch.float32) for coord in coordinates]

        return {
            'nums': nums,
            'font_name': font_names,
            'label': labels,
            'image': images,
            'coordinates': coordinates
        }


if __name__ == '__main__':
    fontDataset = FontDataset(is_train=False, is_dev=False)
