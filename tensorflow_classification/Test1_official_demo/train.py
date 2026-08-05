# =============================================================================
# train.py —— 用 TensorFlow 训练 LeNet（MNIST 手写数字识别）
#
# 【这个文件干什么】
#   用 MNIST 数据集（7 万张 28x28 的手写数字灰度图，10 类：数字 0~9）
#   训练 model.py 里定义的网络，打印每轮的损失和准确率。
#
# 【和 PyTorch 版 train.py 的整体流程对比】
#   PyTorch:  DataLoader 分批数据 → 前向算损失 → loss.backward() 自动求梯度
#            → optimizer.step() 更新参数
#   TensorFlow: tf.data 分批数据 → 前向算损失 → tape.gradient() 求梯度
#            → apply_gradients() 更新参数
#   数学本质完全一样（都是梯度下降 + 链式法则），只是 API 不同。
#
# 【训练的本质（回顾）】
#   网络有几十万个参数，初始随机。训练就是反复：
#     1. 喂一批图 → 得到每个类别的概率
#     2. 算损失（预测和真实标签差多少，越小越好）
#     3. 用链式法则求每个参数的梯度 ∂Loss/∂w
#     4. 参数更新：w ← w - lr * ∂Loss/∂w
# =============================================================================

# from __future__ import ... 是 Python 2/3 兼容的老写法（让旧版 Python 也支持
# 新版语法），现在写代码可以不用管它，照抄即可。
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf  # TensorFlow 主库（约定俗成简写成 tf）
from model import MyModel  # 从 model.py 导入我们自己定义的网络


def main():
    # ----------------------------------------------------------------------
    # 【加载 MNIST 数据集】
    # tf.keras.datasets.mnist 是 Keras 内置的数据集（第一次运行会自动下载）。
    # .load_data() 返回两个元组：
    #   (x_train, y_train)：训练集 6 万张
    #   (x_test, y_test) ：测试集 1 万张
    # x 是图片数据（形状 [60000, 28, 28]，每张 28x28 的灰度图），
    # y 是标签（0~9 的整数，即"这张图是数字几"）。
    # 【Python 知识：元组解包】一行代码把返回值拆给 4 个变量。
    mnist = tf.keras.datasets.mnist

    # download and load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # ----------------------------------------------------------------------
    # 【归一化：把 0~255 的像素值变成 0~1】
    # MNIST 的每个像素是 0~255 的整数（0 = 纯黑，255 = 纯白）。
    # 除以 255.0 后变成 0~1 的小数。注意是 255.0（浮点数），不是 255，
    # 否则整数除法会把 255 直接变 1、200 变 0，数据全毁。
    # 为什么要归一化？让数值范围统一在 [0,1]，梯度下降收敛更快更稳
    # （和 PyTorch 版 Normalize 是同一个目的，只是 MNIST 简单，只缩不放）。
    # ----------------------------------------------------------------------
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # ----------------------------------------------------------------------
    # 【加一个通道维：[60000, 28, 28] → [60000, 28, 28, 1]】
    # 卷积层要求输入 4 维：[batch, 高, 宽, 通道]（TensorFlow 的约定）。
    # 现在数据只有 [数量, 高, 宽] 3 维，还缺"通道"维。
    # MNIST 是灰度图，1 个通道就够了，所以在最后面"塞"一个长度为 1 的维度。
    #
    # 【Python 语法：x[..., tf.newaxis] 是什么意思？】
    #   ... 是"省略号"，表示"前面所有维度原样保留"；
    #   tf.newaxis 在某个位置插入一个大小为 1 的新维度。
    #   合起来就是：在所有维度后面追加一个长度为 1 的维度。
    #   等价写法：x_train = x_train.reshape(-1, 28, 28, 1)
    # ----------------------------------------------------------------------
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # ----------------------------------------------------------------------
    # 【tf.data.Dataset：TensorFlow 的数据流水线】
    # 一行一行拆开看：
    #   ① from_tensor_slices((x_train, y_train))
    #      把 numpy 数组按"第 0 维"切成一个个样本，组成一个"数据集"对象。
    #      可以想象成把 6 万张图和 6 万个标签"一一配对"装进一个容器。
    #   ② .shuffle(10000)
    #      打乱顺序（缓冲区大小 10000）。作用：让每个 batch 近似
    #      "独立同分布"(i.i.d.)，避免网络记住数据的固定顺序 ——
    #      相当于统计里"从总体随机抽样"。
    #   ③ .batch(32)
    #      把样本每 32 个合成一批。训练时一批一批喂，并行计算效率高。
    #      和 PyTorch 的 DataLoader(batch_size=32, shuffle=True) 完全对应。
    # ----------------------------------------------------------------------
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    # create model
    model = MyModel()  # 创建网络（参数此刻随机初始化）

    # ----------------------------------------------------------------------
    # 【核心概率概念：交叉熵损失 SparseCategoricalCrossentropy】
    # 逐个词拆开理解：
    #   Crossentropy（交叉熵）: 衡量"预测的概率分布"和"真实的分布"差多少。
    #     对一张真实数字为 y 的图，模型输出概率 p_0,...,p_9，
    #     损失 = -log(p_y)（真实类别的概率取负对数）。
    #     p_y → 1（自信且正确）时损失 → 0；p_y → 0（自信但错误）时损失 → +∞。
    #     数学本质：最小化它 = 最大化似然 = 统计里的最大似然估计 (MLE)。
    #   Sparse（稀疏）: 我们的标签是整数 0~9，不是 one-hot 向量
    #     （one-hot 是 [0,0,1,0,...] 这种 10 维向量）。
    #     整数标签叫"稀疏表示"，所以用 Sparse 开头的版本，少占内存。
    #   from_logits=False（默认）: 认为模型输出的已经是概率（softmax 过了）。
    #     —— 这就是为什么 model.py 里 Dense(10, activation='softmax')！
    #     如果模型输出 logits（原始得分），必须写 from_logits=True，
    #     两者数学等价，但"概率直接进损失"数值上要注意极端的 log(0)，
    #     所以一般推荐 from_logits=True + 模型不加 softmax（更稳定）。
    # ----------------------------------------------------------------------
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    # 优化器：梯度下降的升级版 Adam（自适应学习率 + 动量），
    # 更新公式本质：w ← w - lr * ∂Loss/∂w
    optimizer = tf.keras.optimizers.Adam()

    # ----------------------------------------------------------------------
    # 【指标 (metrics)：用来"记账"的工具】
    #   Mean：计算平均值。每步把当步的损失喂给它，它自动累加并计数，
    #         最后 .result() 返回"累计平均损失"。
    #         —— 相当于 PyTorch 版里 running_loss / 500 的手动记账。
    #   SparseCategoricalAccuracy：逐批计算"预测对的比例"。
    #         predictions 是概率，它内部取概率最大的类（argmax）和标签比较，
    #         统计正确率。sparse 同样是说标签是整数 0~9。
    # ----------------------------------------------------------------------
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # 测试集的指标（同样一份，但只在测试时用）
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # ----------------------------------------------------------------------
    # 【@tf.function 装饰器】
    # @ 是 Python 的"装饰器"语法：把下面的函数"包装"一下。
    # tf.function 会把整个 Python 函数编译成一张 TensorFlow 计算图，
    # 之后调用时直接跑图，速度更快（图 = 预先排好的运算清单）。
    # 学习阶段知道"加了它跑得快"即可，即使不理解也不影响正确性。
    # ----------------------------------------------------------------------
    @tf.function
    def train_step(images, labels):
        # ------------------------------------------------------------------
        # 【GradientTape：TensorFlow 的自动微分工具】
        # with tf.GradientTape() as tape: 这个"磁带"会记录里面做的
        # 所有运算，之后可以"倒带"求导数。
        # 数学上：损失 Loss 是最后一层输出的函数，最后一层又是前一层输出的
        # 函数……一层套一层（复合函数）。tape.gradient 用链式法则从输出端
        # 往回一层层求偏导，得到 Loss 对每个参数的梯度 ∂Loss/∂w。
        # 这就是 PyTorch 版 loss.backward() 的对应物。
        # ------------------------------------------------------------------
        with tf.GradientTape() as tape:
            predictions = model(images)  # ① 前向：一批图 → 每类的概率 [batch, 10]
            loss = loss_object(labels, predictions)  # ② 算交叉熵损失（标量）
        # ③ 求梯度：Loss 对"所有可训练参数"（卷积核、权重矩阵）的偏导数
        gradients = tape.gradient(loss, model.trainable_variables)
        # ④ 更新参数：w ← w - lr * 梯度（zip 把梯度和参数一一配对）
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # 把本步的损失和准确率"记到账本"上（自动累加，供 epoch 结束时汇总）
        train_loss(loss)
        train_accuracy(labels, predictions)

    # 测试：只需要前向传播算损失和准确率，不需要梯度
    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    EPOCHS = 5  # 把整个训练集完整过 5 遍（1 遍 = 1 个 epoch）

    for epoch in range(EPOCHS):
        # ------------------------------------------------------------------
        # reset_states()：把指标"清零"。
        # 指标是累加的（记录了所有历史步），每个 epoch 开始前必须清零，
        # 否则第 2 轮打印的还是"第 1 轮+第 2 轮"的累计平均，数据就乱了。
        # 对应 PyTorch 版里每 500 步 running_loss = 0.0 的做法。
        # ------------------------------------------------------------------
        train_loss.reset_states()        # clear history info
        train_accuracy.reset_states()    # clear history info
        test_loss.reset_states()         # clear history info
        test_accuracy.reset_states()     # clear history info

        # 【Python 知识：for ... in 遍历】
        # for images, labels in train_ds：train_ds 会一批一批吐出数据，
        # 每批自动拆成 (图片, 标签) 两个变量。把整个训练集遍历完 = 1 个 epoch。
        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        # 【字符串格式化：template.format(...)】
        # template 里的 {} 是"占位符"，format 按顺序把值填进去，
        # 生成完整句子。等价写法：f'Epoch {epoch+1}, ...'（新版 Python 写法）。
        # 准确率乘以 100 是为了显示成百分比（如 0.97 → 97%）。
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))


# 【Python 知识：if __name__ == '__main__'】
# 只有"直接运行本文件"（python train.py）时才执行 main()；
# 被别的文件 import 时不会自动执行（详见 PyTorch 版 model.py 的注释）。
if __name__ == '__main__':
    main()
