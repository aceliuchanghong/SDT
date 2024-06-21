## Style-Disentangled Transformer(SDT)

论文总体框架

```
SDT:
- a dual-head style encoder
- a content encoder
- a Transformer decoder

style encoder
    WriterNCE Lwri : 同一作者对于不同字的风格
    GlyphNCE Lgly : 同一个字不同作者的风格

content encoder:
    ResNet18_Weights: 作为CNN的核心,核心学习特征Qmap (h ,w ,c(通道维度))

multi-layer transformer decoder:
    Lpre: 笔迹移动的Loss
    Lcls: 笔迹状态的Loss
```

总体架构

![img.png](paper%2Fpics%2Fimg.png)

总体Loss

![img_1.png](paper%2Fpics%2Fimg_1.png)

论文数据集

```text

tarin:CASIA-OLHWDB (https://nlpr.ia.ac.cn/databases/handwriting/Home.html)

test:ICDAR-2013

pkl 文件结构:
    item['img'] ==>样子 (64, 64)
    item['label'] ==>文字
mdb 文件结构eg:
    num_sample: 224589
    tag_char: 累
    coords.shape: (42, 5)
        (x,y,pen_down,pen_up,pen_end)下笔、提笔、终止
    fname: C004-f.pot
```

需要知道的基本Func

```text
view
permute
mean
stack
reshape
cat
transpose

rearrange
repeat
```

### create a test env

```shell
pip freeze > requirements.txt
conda create -n SDTLog python=3.10
conda activate SDTLog
pip install -r requirements.txt --proxy=127.0.0.1:10809
watch -n 1 nvidia-smi
```

### 📂 Folder Structure

```
SDT/
|
├── README.md
├── evaluate.py
├── parse_config.py
├── requirements.txt
├── sdt.pdf
├── test.py
├── train.py
├── user_generate.py
├── Saved/
│   ├── models/
│   ├── samples/
│   └── tborad/
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
├── style_samples/
├── trainer/
│   └── trainer.py
└── utils/
    ├── change_mdb.py
    ├── check_db.py
    ├── config.py
    ├── create_test_and_train_pkl.py
    ├── cut_pics.py
    ├── deal_before_generate.py
    ├── font_labels.db
    ├── judge_font.py
    ├── logger.py
    ├── pic_bin.py
    ├── pics_with_pkl.py
    ├── remove_comments.py
    ├── structure.py
    ├── test.pkl
    └── util.py
```

### 🚀 Training & Test

**模型训练**

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

- 生成笔迹:

```
python test.py --pretrained_model checkpoint_path/checkpoint-iter199999.pth --store_type online --sample_size 500 --dir Generated/Chinese
```

- 生成图片:

```
python test.py --pretrained_model checkpoint_path/checkpoint-iter199999.pth --store_type img --sample_size 500 --dir Generated/Chinese
```

**定量评估**

- 评估生成的笔迹，需要设置为 data_path 生成的笔迹的路径:

```
python evaluate.py --data_path Generated/Chinese
```

**自己字体**

- 把图片放到文件夹style_samples

```
python user_generate.py --pretrained_model checkpoint_path/checkpoint-iter199999.pth --style_path style_samples
```

**png转ttf**

```
# see here
git clone https://github.com/aceliuchanghong/PngToTTF
```

### Add

- pkl文件解析&生成
- 字体转图片
- 图片转字体
- 多gpu训练
- 项目结构目录修改,配置文件完善
- mdb文件解析
- 各个文件注释添加以及规范
- 标注楷书草书
- 增加论文翻译
- 增加各种辅助函数
- 输入图片标准化

### TODO

- 额外的装饰网络，为SDT生成的均匀笔画的文字增加了笔画宽度和颜色
