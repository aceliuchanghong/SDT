from torch.utils.data import Dataset
import os
import pickle

script = {
    "CHINESE": ['CASIA_CHINESE', 'Chinese_content.pkl', 'character_dict.pkl'],
    "train": 'train_style_samples',
    "test": 'test_style_samples',
}


class FontDataset(Dataset):
    def __init__(self, root='../data', dataset='CHINESE', is_train=True, num_img=15):
        data_path = os.path.join(root, script[dataset][0])
        self.dataset = dataset
        self.content = pickle.load(open(os.path.join(data_path, script[dataset][1]), 'rb'))  # content samples
        self.char_dict = pickle.load(open(os.path.join(data_path, script[dataset][2]), 'rb'))
        self.is_train = is_train
        if self.is_train:
            self.img_path = os.path.join(data_path, script['train'])  # style samples
            self.num_img = num_img * 2
        else:
            self.img_path = os.path.join(data_path, script['test'])  # style samples
            self.num_img = num_img
        self.pkl_file = self._get_files(self.img_path, '.pkl')
        self.font_type = len(self.pkl_file)
        self.indexes = {}
        self.pair_index = []

        # Create indexes and pair_index
        for i in range(self.font_type):
            temp_content = pickle.load(open(self.pkl_file[i], 'rb'))
            base_name = os.path.basename(self.pkl_file[i])
            self.indexes[base_name] = len(temp_content)
            for j in range(len(temp_content)):
                self.pair_index.append((base_name, j))

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError('Index out of range')
        font_name, char_idx = self.pair_index[idx]
        sample_label_img = pickle.load(open(os.path.join(self.img_path, font_name), 'rb'))[char_idx]
        return sample_label_img, font_name

    def __len__(self):
        return len(self.pair_index)

    def _get_files(self, path, suffix):
        files_with_suffix = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(suffix):
                    files_with_suffix.append(os.path.join(root, file).replace("\\", '/'))
        return files_with_suffix


if __name__ == '__main__':
    FontDataset = FontDataset(is_train=False)
