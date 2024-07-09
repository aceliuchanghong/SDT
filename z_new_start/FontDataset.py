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
        self.config = new_start_config

        self.content = pickle.load(open(self.config[self.config_set]['content_pkl_path'], 'rb'))
        self.char_dict = pickle.load(open(self.config[self.config_set]['character_pkl_path'], 'rb'))
        self.pic_path = self.config[self.config_set]['z_pic_pkl_path']
        self.coordinate_path = self.config[self.config_set]['z_coordinate_pkl_path']
        self.max_stroke = 20
        self.max_per_stroke_point = 200

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
                """
                文字笔画过多不要了,最多self.max_length,某个3个笔画的文字
                [
                    [(429.0, -43.0, 1, 0)
                     (404.0, -53.0, 0, 0)
                     (549.0, 56.0, 0, 1)],
                    [(967.0, 744.0, 1, 0)
                     (831.0, 774.0, 0, 1)],
                    [(704.0, 130.0, 1, 0)
                     (760.0, 692.0, 0, 1)]
                ]
                单个笔画坐标点太多了也不要了
                """
                max_per_stroke_point = max(len(sublist) for sublist in font_coors_list[char])
                if (char in font_coors_list and
                        len(font_coors_list[char]) <= self.max_stroke and
                        max_per_stroke_point <= self.max_per_stroke_point):
                    self.font_data.append(
                        (i, font_name, pic['label'], pic['img'], font_coors_list[char])
                    )
                    # print('文字:', pic['label'],
                    #       '笔画数量:', str(len(font_coors_list[char])),
                    #       '一个笔画最多坐标点数量:', str(max_per_stroke_point))

        train_size = int(len(self.font_data) * train_percent)
        if is_train:
            self.font_data = self.font_data[:train_size]
        else:
            self.font_data = self.font_data[train_size:]

        self.num_sample = len(self.font_data)

    def __getitem__(self, idx):
        font_nums, font_name, label, char_img, coors = self.font_data[idx]
        label_id = self.char_dict.index(label)
        char_img = char_img / 255

        # 添加通道维度 1 * 64 * 64
        char_img_tensor = torch.tensor(char_img, dtype=torch.float32).unsqueeze(0)

        # 对coors进行padding 使其长度一致 20 * 200 * 4
        # 1.每个字符最多包含的笔画数
        # 2.每个笔画最多包含的点数
        padded_coors = torch.zeros((self.max_stroke, self.max_per_stroke_point, 4), dtype=torch.float32)
        for i, stroke in enumerate(coors):
            if i >= self.max_stroke:
                break
            for j, point in enumerate(stroke):
                if j >= self.max_per_stroke_point:
                    break
                padded_coors[i, j] = torch.tensor(point, dtype=torch.float32)
        # print(padded_coors)
        # print(padded_coors.shape)
        output = {
            'label_id': torch.tensor(label_id, dtype=torch.long),
            'char_img': char_img_tensor,
            'coordinates': padded_coors
        }
        return output

    def __len__(self):
        return self.num_sample

    def collect_function(self, batch_data):
        batch_char_imgs = torch.stack([item['char_img'] for item in batch_data])
        batch_coordinates = torch.stack([item['coordinates'] for item in batch_data])
        batch_label_ids = torch.tensor([item['label_id'] for item in batch_data], dtype=torch.long)

        return {
            'char_imgs': batch_char_imgs,
            'coordinates': batch_coordinates,
            'label_ids': batch_label_ids
        }


if __name__ == '__main__':
    fontDataset = FontDataset(is_train=False, is_dev=False)
