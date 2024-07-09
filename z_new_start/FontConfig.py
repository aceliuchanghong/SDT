new_start_config = {
    'test': {
        'z_coordinate_pkl_path': r'D:\aProject\py\SDT\z_new_start\ABtest\files\AB_coors',
        'z_pic_pkl_path': r'D:\aProject\py\SDT\z_new_start\ABtest\files\AB_pics_pkl',
        'content_pkl_path': r'D:\aProject\py\SDT\z_new_start\ABtest\files\new_chinese_content.pkl',
        'character_pkl_path': r'D:\aProject\py\SDT\z_new_start\ABtest\files\new_character_dict.pkl',
        'save_model_dir': r'D:\aProject\py\SDT\z_new_start\save_model',
        'PER_BATCH': 16,
        'NUM_THREADS': 0,
    },
    'dev': {
        'z_coordinate_pkl_path': '/mnt/data/llch/Chinese-Fonts-Dataset/all_test/AB_coors',
        'z_pic_pkl_path': '/mnt/data/llch/Chinese-Fonts-Dataset/all_test/AB_pics_pkl',
        'content_pkl_path': '/mnt/data/llch/Chinese-Fonts-Dataset/all_test/new_chinese_content.pkl',
        'character_pkl_path': '/mnt/data/llch/Chinese-Fonts-Dataset/all_test/ew_character_dict.pkl',
        'save_model_dir': '/mnt/data/llch/SDT/z_new_start/save_model',
        'PER_BATCH': 64,
        'NUM_THREADS': 8,
    },
    'train': {
        'seed': 2024,
        'num_epochs': 20,
        'LEARNING_RATE': 0.001,
        'SNAPSHOT_BEGIN': 2000,
        'SNAPSHOT_EPOCH': 2000,
        'VALIDATE_BEGIN': 2000,
        'VALIDATE_EPOCH': 2000,
        'IMG_H': 64,
        'IMG_W': 64,
    }
}
