import torch.nn as nn  # 导入PyTorch神经网络模块
import torch  # 导入PyTorch

# 官方预训练权重下载地址
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(nn.Module):  # 定义VGG网络类
    def __init__(self, features, num_classes=1000, init_weights=False):  # 初始化
        super(VGG, self).__init__()  # 调用父类初始化
        self.features = features  # 特征提取部分(卷积+池化)
        self.classifier = nn.Sequential(  # 分类器(全连接层)
            nn.Linear(512*7*7, 4096),  # 将特征展平后接全连接层
            nn.ReLU(True),  # ReLU激活
            nn.Dropout(p=0.5),  # 随机失活50%防过拟合
            nn.Linear(4096, 4096),  # 全连接层
            nn.ReLU(True),  # ReLU激活
            nn.Dropout(p=0.5),  # 随机失活50%
            nn.Linear(4096, num_classes)  # 输出层，节点数=类别数
        )
        if init_weights:  # 是否初始化权重
            self._initialize_weights()  # 调用权重初始化方法

    def forward(self, x):  # 前向传播
        # N x 3 x 224 x 224  (输入张量形状)
        x = self.features(x)  # 经过卷积/池化层
        # N x 512 x 7 x 7  (特征图形状)
        x = torch.flatten(x, start_dim=1)  # 展平除batch外的维度
        # N x 512*7*7
        x = self.classifier(x)  # 经过全连接分类器
        return x  # 返回输出

    def _initialize_weights(self):  # 权重初始化
        for m in self.modules():  # 遍历所有子模块
            if isinstance(m, nn.Conv2d):  # 若是卷积层
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)  # Xavier均匀分布初始化权重
                if m.bias is not None:  # 若存在偏置
                    nn.init.constant_(m.bias, 0)  # 偏置初始化为0
            elif isinstance(m, nn.Linear):  # 若是全连接层
                nn.init.xavier_uniform_(m.weight)  # Xavier初始化权重
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)  # 偏置初始化为0


def make_features(cfg: list):  # 根据配置列表构建卷积部分
    layers = []  # 存放层
    in_channels = 3  # 输入通道数为3(RGB)
    for v in cfg:  # 遍历配置
        if v == "M":  # 若为'M'，表示最大池化
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # 添加2x2最大池化
        else:  # 否则为卷积输出通道数
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)  # 3x3卷积
            layers += [conv2d, nn.ReLU(True)]  # 添加卷积+ReLU
            in_channels = v  # 更新输入通道数
    return nn.Sequential(*layers)  # 打包成Sequential返回


cfgs = {  # 各版本网络结构配置
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", **kwargs):  # 创建VGG模型
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)  # 校验配置存在
    cfg = cfgs[model_name]  # 获取对应配置

    model = VGG(make_features(cfg), **kwargs)  # 构建VGG实例
    return model  # 返回模型