import argparse
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from data_loader.loader import UserDataset


def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()

    """setup data_loader instances"""
    test_dataset = UserDataset(
        '../data', cfg.DATA_LOADER.DATASET, opt.style_path)
    print(len(test_dataset))
    print(test_dataset.__getitem__(10))


if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='../configs/CHINESE_USER.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--dir', dest='save_dir', default='../Generated/Chinese_User',
                        help='target dir for storing the generated characters')
    parser.add_argument('--pretrained_model', dest='pretrained_model',
                        default='../checkpoint_path/checkpoint-iter199999.pth',
                        help='continue train model')
    parser.add_argument('--style_path', dest='style_path', default='../style_samples', help='dir of style samples')
    opt = parser.parse_args()
    main(opt)
