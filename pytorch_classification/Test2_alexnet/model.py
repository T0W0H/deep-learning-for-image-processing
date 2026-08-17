import torch.nn as nn
import torch


class AlexNet(nn.Module):
    """AlexNet 网络：卷积层提特征，全连接层做分类"""

    def __init__(self, num_classes=1000, init_weights=False):
        # num_classes：分类数量；init_weights：是否初始化权重
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(  # 特征提取部分（卷积 + 池化）
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(  # 分类部分（全连接层）
            nn.Dropout(p=0.5),            # 随机丢弃一半神经元，防止过拟合
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),  # 最终输出每个类别的得分
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)                # 1. 卷积提取特征
        x = torch.flatten(x, start_dim=1)   # 2. 展平成一维，才能接全连接层
        x = self.classifier(x)              # 3. 全连接层分类
        return x

    def _initialize_weights(self):
        # 权重初始化：让训练收敛更快、更稳定
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
