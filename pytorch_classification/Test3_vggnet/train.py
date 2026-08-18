import os  # 操作系统接口
import sys  # 系统参数
import json  # JSON数据处理

import torch  # PyTorch
import torch.nn as nn  # 神经网络模块
from torchvision import transforms, datasets  # 图像预处理与数据集
import torch.optim as optim  # 优化器
from tqdm import tqdm  # 进度条

from model import vgg  # 导入自定义VGG模型


def main():
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备(GPU/CPU)
    print("using {} device.".format(device))  # 打印设备信息

    data_transform = {  # 不同阶段的预处理
        "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪缩放
                                     transforms.RandomHorizontalFlip(),  # 随机水平翻转(数据增强)
                                     transforms.ToTensor(),  # 转为张量
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),  # 归一化
        "val": transforms.Compose([transforms.Resize((224, 224)),  # 缩放为224x224
                                   transforms.ToTensor(),  # 转为张量
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}  # 归一化

    # get data root path 获取数据根目录
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    # flower data set path 花朵数据集路径
    image_path = os.path.join(data_root, "data_set", "flower_data")
    assert os.path.exists(image_path), "{} path does not exist.".format(
        image_path)  # 校验路径存在
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),  # 加载训练集
                                         transform=data_transform["train"])  # 使用训练预处理
    train_num = len(train_dataset)  # 训练集图片数量

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx  # 类别名->索引映射
    cla_dict = dict((val, key)
                    for key, val in flower_list.items())  # 反转字典(索引->类别名)
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)  # 转成格式化JSON字符串
    with open('class_indices.json', 'w') as json_file:  # 写入json文件
        json_file.write(json_str)  # 保存类别索引

    batch_size = 32  # 批大小
    batch_size = 1
    # number of workers 数据加载线程数
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    nw = 0
    print('Using {} dataloader workers every process'.format(nw))  # 打印线程数

    train_loader = torch.utils.data.DataLoader(train_dataset,  # 构建训练数据加载器
                                               batch_size=batch_size, shuffle=True,  # 打乱数据
                                               num_workers=nw)  # 多进程加载

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),  # 加载验证集
                                            transform=data_transform["val"])  # 使用验证预处理
    val_num = len(validate_dataset)  # 验证集图片数量
    validate_loader = torch.utils.data.DataLoader(validate_dataset,  # 构建验证数据加载器
                                                  batch_size=batch_size, shuffle=False,  # 不打乱
                                                  num_workers=nw)  # 多进程加载
    print("using {} images for training, {} images for validation.".format(train_num,  # 打印数据量
                                                                           val_num))

    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()

    model_name = "vgg16"  # 模型名称
    model_name = "vgg11"
    net = vgg(model_name=model_name, num_classes=5,
              init_weights=True)  # 创建VGG16(5类)
    net.to(device)  # 模型转到设备
    loss_function = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.0001)  # Adam优化器

    epochs = 30  # 训练轮数
    best_acc = 0.0  # 最优准确率
    save_path = './{}Net.pth'.format(model_name)  # 权重保存路径
    train_steps = len(train_loader)  # 每个epoch的迭代步数
    for epoch in range(epochs):  # 遍历每个epoch
        # train
        net.train()  # 切换到训练模式(启用dropout等)
        running_loss = 0.0  # 累计损失
        train_bar = tqdm(train_loader, file=sys.stdout)  # 训练进度条
        for step, data in enumerate(train_bar):  # 遍历每个batch
            images, labels = data  # 解包图片和标签
            optimizer.zero_grad()  # 梯度清零
            outputs = net(images.to(device))  # 前向传播
            loss = loss_function(outputs, labels.to(device))  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            # print statistics
            running_loss += loss.item()  # 累加损失

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,  # 更新进度条显示
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()  # 切换到评估模式(关闭dropout)
        acc = 0.0  # accumulate accurate number / epoch 累计正确数
        with torch.no_grad():  # 不计算梯度
            val_bar = tqdm(validate_loader, file=sys.stdout)  # 验证进度条
            for val_data in val_bar:  # 遍历验证集
                val_images, val_labels = val_data  # 解包数据
                outputs = net(val_images.to(device))  # 前向传播
                predict_y = torch.max(outputs, dim=1)[1]  # 取每行最大值的索引作为预测类别
                acc += torch.eq(predict_y, val_labels.to(device)
                                ).sum().item()  # 统计预测正确的个数

        val_accurate = acc / val_num  # 计算验证准确率
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %  # 打印本epoch结果
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:  # 若准确率创新高
            best_acc = val_accurate  # 更新最优准确率
            torch.save(net.state_dict(), save_path)  # 保存最优权重

    print('Finished Training')  # 训练结束


if __name__ == '__main__':  # 主程序入口
    main()  # 调用main函数
