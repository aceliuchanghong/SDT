
## 📢 Introduction
- The proposed style-disentangled Transformer (SDT) generates online handwritings with conditional content and style. Existing RNN-based methods mainly focus on capturing a person’s overall writing style, neglecting subtle style inconsistencies between characters written by the same person. In light of this, SDT disentangles the writer-wise and character-wise style representations from individual handwriting samples for enhancing imitation performance.
- We extend SDT and introduce an offline-to-offline framework for improving the generation quality of offline Chinese handwritings.



<div style="display: flex; flex-direction: column; align-items: center; ">
<img src="static/overview_sdt.jpg" style="width: 100%;">
</div>
<p align="center" style="margin-bottom: 10px;">
Overview of our SDT
</p>

<div style="display: flex; justify-content: center;">
<img src="static/duo_loop.gif" style="width: 33.33%;"><img src="static/mo_loop.gif" style="width: 33.33%;"><img src="static/tai_loop.gif" style="width: 33.33%;">
</div>
<p align="center">
Three samples of online characters with writing orders
</p>


## 📺 Handwriting generation results
- **Online Chinese handwriting generation**
![online Chinese](static/online_Chinese.jpg)

- **Applications to various scripts**
![other scripts](static/various_scripts.jpg)
- **Extension on offline Chinese handwriting generation**
![offline Chinese](static/offline_Chinese.jpg)

  
## 📂 Folder Structure
  ```
  SDT/
  │
  ├── train.py - main script to start training
  ├── test.py - generate characters via trained model
  ├── evaluate.py - evaluation of generated samples
  │
  ├── configs/*.yml - holds configuration for training
  ├── parse_config.py - class to handle config file
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── loader.py
  │
  ├── model_zoo/ - pre-trained content encoder model
  │
  ├── data/ - default directory for storing experimental datasets
  │
  ├── model/ - networks, models and losses
  │   ├── encoder.py
  │   ├── gmm.py
  │   ├── loss.py
  │   ├── model.py
  │   └── transformer.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   ├── tborad/ - tensorboard visualization
  │   └── samples/ - visualization samples in the training process
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── logger.py - set log dir for tensorboard and logging output
  ```

## 💿 Datasets

We provide Chinese, Japanese and English datasets in [Google Drive](https://drive.google.com/drive/folders/17Ju2chVwlNvoX7HCKrhJOqySK-Y-hU8K?usp=share_link) | [Baidu Netdisk](https://pan.baidu.com/s/1RNQSRhBAEFPe2kFXsHZfLA) PW:xu9u. Please download these datasets, uzip them and move the extracted files to /data.

## 🍔 Pre-trained model
- We provide the pre-trained content encoder model in [Google Drive](https://drive.google.com/drive/folders/1N-MGRnXEZmxAW-98Hz2f-o80oHrNaN_a?usp=share_link) | [Baidu Netdisk](https://pan.baidu.com/s/1RNQSRhBAEFPe2kFXsHZfLA) PW:xu9u. Please download and put it to the /model_zoo. 
- We provide the well-trained SDT model in [Google Drive](https://drive.google.com/drive/folders/1LendizOwcNXlyY946ThS8HQ4wJX--YL7?usp=sharing) | [Baidu Netdisk](https://pan.baidu.com/s/1RNQSRhBAEFPe2kFXsHZfLA) PW:xu9u, so that users can get rid of retraining one and play it right away.

## 🚀 Training & Test
**Training**
- To train the SDT on the Chinese dataset, run this command:
```
python train.py --cfg configs/CHINESE_CASIA.yml --log Chinese_log
```

- To train the SDT on the Japanese dataset, run this command:
```
python train.py --cfg configs/Japanese_TUATHANDS.yml --log Japanese_log
```

- To train the SDT on the English dataset, run this command:
```
python train.py --cfg configs/English_CASIA.yml --log English_log
```

**Qualitative Test**
- To generate Chinese handwritings with our SDT, run this command:
```
python test.py --pretrained_model checkpoint_path --store_type online --sample_size 500 --dir Generated/Chinese
python test.py --pretrained_model checkpoint_path/checkpoint-iter199999.pth --store_type online --sample_size 500 --dir Generated/Chinese
```

- To generate Japanese handwritings with our SDT, run this command:
```
python test.py --pretrained_model checkpoint_path --store_type online --sample_size 500 --dir Generated/Japanese
```

- To generate English handwritings with our SDT, run this command:
```
python test.py --pretrained_model checkpoint_path --store_type online --sample_size 500 --dir Generated/English
```

**Quantitative Evaluation**
- To evaluate the generated handwritings, you need to set `data_path` to the path of the generated handwritings (e.g., Generated/Chinese), and run this command:
```
python evaluate.py --data_path Generated/Chinese
```

### create a test env

```shell
pip freeze > requirements.txt
conda create -n SDTLog python=3.8
conda activate SDTLog
pip install -r requirements.txt --proxy=127.0.0.1:10809
```
