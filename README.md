## style-disentangled Transformer(SDT)

### create a test env

```shell
pip freeze > requirements.txt
conda create -n SDTLog python=3.8
conda activate SDTLog
pip install -r requirements.txt --proxy=127.0.0.1:10809
```

## 📂 Folder Structure
```
SDT/
|
├── README.md
├── evaluate.py
├── parse_config.py
├── requirements.txt
├── test.py
├── train.py
├── user_generate.py
├── checkpoint_path/
├── configs/
│   ├── CHINESE_CASIA.yml
│   ├── CHINESE_USER.yml
│   ├── English_CASIA.yml
│   └── Japanese_TUATHANDS.yml
├── data_loader/
│   └── loader.py
├── model_zoo/
├── models/
│   ├── encoder.py
│   ├── gmm.py
│   ├── loss.py
│   ├── model.py
│   └── transformer.py
├── saved/
│   ├── models/
│   ├── samples/
│   └── tborad/
├── trainer/
│   └── trainer.py
└── utils/
    ├── logger.py
    ├── structure.py
    └── util.py
```


## 🚀 Training & Test
**训练**
- 在中文数据集上训练 SDT:
```
python train.py --cfg configs/CHINESE_CASIA.yml --log Chinese_log
```

- 在日语数据集上训练 SDT:
```
python train.py --cfg configs/Japanese_TUATHANDS.yml --log Japanese_log
```

- 在英语数据集上训练 SDT:
```
python train.py --cfg configs/English_CASIA.yml --log English_log
```

**定性测试**
- 生成中文笔迹:
```
python test.py --pretrained_model checkpoint_path --store_type online --sample_size 500 --dir Generated/Chinese
python test.py --pretrained_model checkpoint_path/checkpoint-iter199999.pth --store_type online --sample_size 500 --dir Generated/Chinese
```
- 生成日语笔迹:
```
python test.py --pretrained_model checkpoint_path --store_type online --sample_size 500 --dir Generated/Japanese
```
- 生成英文笔迹:
```
python test.py --pretrained_model checkpoint_path --store_type online --sample_size 500 --dir Generated/English
```

**定量评估**
- 评估生成的笔迹，需要设置为 data_path 生成的笔迹的路径:
```
python evaluate.py --data_path Generated/Chinese
```


