### 说明

```text
由于我是初学者,我很了解初学者什么都不懂,所以我把我的学习路程展示给大家,希望不要笑话我就可以了,XD
以下是我问gpt-4o的内容
```

---

在做transformer训练模型的时候,我的目录结构如下,我应该按照什么顺序写代码?
```structure
├── FontConfig.py
├── FontDataset.py
├── FontModel.py
└── FontTrainer.py
```
还有缺失的组件吗?

---

帮我完善以下字体训练的模型
```python
class FontModel(nn.Module):
    def __init__(self,
                 d_model=512,
                 num_head=8,
                 num_encoder_layers=2,
                 num_head_layers=1,
                 wri_decoder_layers=2,
                 gly_decoder_layers=2,
                 dim_feedforward=2048,  # 前馈神经网络中隐藏层的大小
                 dropout=0.1,
                 activation="relu",
                 normalize_before=True,  # 应用多头注意力和前馈神经网络之前是否对输入进行层归一化
                 ):
        super(FontModel, self).__init__()

    def forward(self, img):
        pass

    def inference(self):
        pass
```

---

