from torch.utils.data import Dataset
import os
import pickle
import torch
from utils.judge_font import get_files
from z_new_start.FontConfig import new_start_config


class FontDataset(Dataset):
    def __init__(self, is_train=False, is_dev=True, train_percent=0.9):
        """
        is_train 给训练数据集还是测试数据集
        is_dev 在真正环境跑还是测试环境跑
        """
        if is_dev:
            self.config_set = 'dev'
        else:
            self.config_set = 'test'
        # print(f"preparing {self.config_set} env dataset...")
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
                if char in font_coors_list and len(font_coors_list[char]) <= 25:
                    # 文字笔画过多不要了
                    # if len(font_coors_list[char]) > 50:
                    #     print(pic['label'], font_coors_list[char])
                    self.font_data.append(
                        (i, font_name, pic['label'], pic['img'], font_coors_list[char])
                    )

        train_size = int(len(self.font_data) * train_percent)
        if is_train:
            self.font_data = self.font_data[:train_size]
        else:
            self.font_data = self.font_data[train_size:]

        self.num_sample = len(self.font_data)
        # print(f"{self.config_set} dataset is ready...")

    def __getitem__(self, idx):
        nums, font_name, label, img, coors = self.font_data[idx]
        return {'nums': nums, 'font_name': font_name, 'label': label, 'image': img, 'coordinates': coors}

    def __len__(self):
        return self.num_sample

    def collect_function(self, batch_data):
        batch_size = len(batch_data)

        # 提取各个字段
        nums = [item['nums'] for item in batch_data]
        font_names = [item['font_name'] for item in batch_data]
        labels = [item['label'] for item in batch_data]
        images = [item['image'] for item in batch_data]
        coordinates = [item['coordinates'] for item in batch_data]

        # 将图像堆叠成一个张量
        images = torch.stack([torch.tensor(img, dtype=torch.float32) for img in images], dim=0)

        # 将 nums 转换为张量
        nums = torch.tensor(nums, dtype=torch.int64)

        # 将 labels 转换为适当的形式
        labels = [torch.tensor(ord(label), dtype=torch.int64) for label in labels]  # label 是单个字符
        labels = torch.stack(labels)

        # 找到 batch 中最长的序列长度
        max_len = max([coord.shape[0] for coord in coordinates])

        # 初始化 coordinates 张量
        padded_coordinates = torch.zeros((batch_size, max_len, 2), dtype=torch.float32)

        for i, coord in enumerate(coordinates):
            length = coord.shape[0]
            padded_coordinates[i, :length] = torch.tensor(coord, dtype=torch.float32)

        # 构造返回字典
        return {
            'nums': nums,
            'font_name': font_names,
            'label': labels,
            'image': images,
            'coordinates': padded_coordinates
        }


if __name__ == '__main__':
    fontDataset = FontDataset(is_train=False, is_dev=False)
