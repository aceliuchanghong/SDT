from parse_config import cfg
from data_loader.loader import ScriptDataset
import torch
from torch.utils.data import DataLoader

train_dataset = ScriptDataset(
    cfg.DATA_LOADER.PATH, cfg.DATA_LOADER.DATASET, cfg.TRAIN.ISTRAIN, cfg.MODEL.NUM_IMGS
)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=cfg.TRAIN.IMS_PER_BATCH,
                                           shuffle=True,
                                           drop_last=False,
                                           collate_fn=train_dataset.collate_fn_,
                                           num_workers=cfg.DATA_LOADER.NUM_THREADS)


