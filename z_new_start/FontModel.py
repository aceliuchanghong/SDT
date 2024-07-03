import torchvision.models as models
from models.transformer import *
from models.encoder import Content_TR
from einops import rearrange, repeat
from models.gmm import get_seq_from_gmm
from torchvision.models.resnet import ResNet18_Weights


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

        # 输入图像的特征提取卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )

        # Transformer 编码器
        encoder_layers = TransformerEncoderLayer(d_model, num_head, dim_feedforward, dropout, activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        # 书写特征解码器
        wri_decoder_layers = TransformerDecoderLayer(d_model, num_head, dim_feedforward, dropout, activation)
        self.wri_transformer_decoder = TransformerDecoder(wri_decoder_layers, wri_decoder_layers)

        # 字形特征解码器
        gly_decoder_layers = TransformerDecoderLayer(d_model, num_head, dim_feedforward, dropout, activation)
        self.gly_transformer_decoder = TransformerDecoder(gly_decoder_layers, gly_decoder_layers)

        self.fc_out = nn.Linear(d_model, 256)  # 假设输出维度为256，可根据需要调整

        # 位置编码
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, d_model))

    def forward(self, img):
        # 图像特征提取
        img_features = self.conv(img)
        img_features = img_features.flatten(2)  # 展平，以便于输入Transformer
        img_features = img_features + self.positional_encoding[:, :img_features.size(1), :]

        # Transformer 编码器
        encoder_output = self.transformer_encoder(img_features)

        # 书写特征解码器
        wri_output = self.wri_transformer_decoder(encoder_output, encoder_output)

        # 字形特征解码器
        gly_output = self.gly_transformer_decoder(encoder_output, encoder_output)

        # 输出
        output = self.fc_out(gly_output.mean(dim=1))

        return output

    def inference(self, img):
        self.eval()
        with torch.no_grad():
            output = self.forward(img)
        return output
