# =============================================================================
# model.py —— 用 TensorFlow 定义 LeNet 网络结构（MNIST 手写数字版）
#
# 【这个文件干什么】
#   定义网络结构。和 PyTorch 版（pytorch_classification/Test1_official_demo）
#   是同一个 LeNet 思路，但框架换成了 TensorFlow。
#
# 【本版和 PyTorch 版的三点关键区别（建议对比着学）】
#   ① 数据集不同：这里用 MNIST 手写数字（28x28 灰度图），不是 CIFAR10 彩色图
#      → 输入只有 1 个通道（黑白），所以形状是 [batch, 28, 28, 1]
#   ② 张量维度顺序不同：
#      - PyTorch 约定： [N, C, H, W]（通道在前）→ 彩色图 [N, 3, 32, 32]
#      - TensorFlow 约定：[batch, H, W, C]（通道在最后）→ 灰度图 [batch, 28, 28, 1]
#      写代码前先搞清楚框架的约定，这是新手最容易踩的坑！
#   ③ softmax 位置不同：
#      - PyTorch 版：模型输出 logits（原始得分），CrossEntropyLoss 内部做 softmax
#      - TensorFlow 版：模型里直接加 activation='softmax' 输出概率，
#        因为 SparseCategoricalCrossentropy 默认认为输入已经是概率（from_logits=False）
#        —— 详细解释见 train.py
#
# 【网络结构（数据流向）】
#   输入 [batch, 28, 28, 1]
#     ↓ Conv2D（32 个 3x3 卷积核，ReLU）
#   [batch, 26, 26, 32]
#     ↓ Flatten（拉平成一长条）
#   [batch, 21632]
#     ↓ Dense（128 个神经元，ReLU）
#   [batch, 128]
#     ↓ Dense（10 个神经元，softmax → 概率）
#   [batch, 10]  ← 10 个数字类别（0~9）各自的概率
# =============================================================================

# 从 TensorFlow 的 Keras 接口里导入需要的"层"（layer）：
#   Conv2D  = 二维卷积层（特征提取）
#   Flatten = 拉平层（把多维数据变一维）
#   Dense   = 全连接层（矩阵乘法 y = Wx + b）
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model  # Model 是所有 Keras 模型的基类


class MyModel(Model):
    # 和 PyTorch 的 nn.Module 类似，tf.keras.Model 帮我们管理参数和训练流程。
    def __init__(self):
        super(MyModel, self).__init__()  # 调用父类的构造函数（固定写法）

        # ------------------------------------------------------------------
        # Conv2D(滤波器个数, 卷积核大小, activation=激活函数)
        #   conv1: 32 个 3x3 卷积核 + ReLU 激活
        #
        # 【和 PyTorch 写法的区别】
        #   PyTorch:  nn.Conv2d(输入通道数, 输出通道数, 核大小)
        #   TF:       Conv2D(输出通道数(=滤波器个数), 核大小)
        #   → TF 不需要写"输入通道数"！它会根据喂进来的数据自动推断。
        #
        # 【每个参数什么意思】
        #   32 个滤波器 = 32 个"特征检测器"（有的看横线、有的看竖线、有的看圆弧…），
        #   所以输出的第 3 维是 32（每个滤波器产生一张"特征图"）。
        #   3x3 卷积核在图上滑动：输出尺寸 = 输入 - 核 + 1 = 28 - 3 + 1 = 26
        #   （没写 padding 就是默认不填充，所以图变小了一圈）
        #   activation='relu' 表示卷积后立刻做 f(x) = max(0, x)，
        #   给网络引入非线性（没有它，多层线性变换叠起来还是线性的）。
        # ------------------------------------------------------------------
        self.conv1 = Conv2D(32, 3, activation='relu')

        # Flatten：把 [batch, 26, 26, 32] 拉平成 [batch, 26*26*32] = [batch, 21632]
        # 全连接层只能吃一维向量（矩阵乘法的要求），所以先拉平。
        self.flatten = Flatten()

        # Dense(神经元个数, activation=激活函数)：全连接层
        # d1: 128 个神经元。每个神经元对输入的 21632 个数做加权和 + 偏置，
        #     再过 ReLU。数学：y = Wx + b（W 形状 [128, 21632]）
        self.d1 = Dense(128, activation='relu')

        # d2: 10 个神经元 = 10 个类别（数字 0~9）
        # activation='softmax'：把 10 个原始得分变成概率分布！
        #   p_i = exp(z_i) / Σ_{j=0}^{9} exp(z_j)
        # 保证 p_i ≥ 0 且 Σp_i = 1，p_i 就是"图片是数字 i"的概率。
        #
        # 【为什么这里就做 softmax，而 PyTorch 版不做？】
        #   PyTorch 的 CrossEntropyLoss 内部自带 softmax，所以模型输出 logits；
        #   TensorFlow 的 SparseCategoricalCrossentropy 默认参数
        #   from_logits=False，即"输入已经 softmax 过"的概率。
        #   两套框架的约定不同，但数学上的交叉熵是一回事。
        self.d2 = Dense(10, activation='softmax')

    # 【和 PyTorch 的区别】PyTorch 里叫 forward()，TensorFlow 里叫 call()。
    # 都是"前向传播"：数据流过各层，算出输出。
    # **kwargs 是 Python 的"任意关键字参数"收集器（接受多余参数，照抄即可）。
    def call(self, x, **kwargs):
        x = self.conv1(x)      # input[batch, 28, 28, 1] output[batch, 26, 26, 32]
        x = self.flatten(x)    # output [batch, 21632]（26*26*32 = 21632）
        x = self.d1(x)         # output [batch, 128]
        return self.d2(x)      # output [batch, 10]（10 个类别的概率，和为 1）
