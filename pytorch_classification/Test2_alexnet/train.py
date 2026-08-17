import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet


def main():
    # 优先用 GPU 训练，没有 GPU 就用 CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 数据预处理：训练集做数据增强（随机裁剪/翻转），验证集只做缩放
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # 注意：必须是 (224, 224)，不能只写 224
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # 项目根目录
    image_path = os.path.join(data_root, "data_set", "flower_data")  # 花卉数据集路径
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # ImageFolder：按子文件夹自动给每张图打标签（一个花一个文件夹）
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx  # 类别名 -> 索引
    cla_dict = dict((val, key) for key, val in flower_list.items())  # 反转：索引 -> 类别名
    # 类别映射写入 json，供 predict.py 使用
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # 数据加载进程数
    print('Using {} dataloader workers every process'.format(nw))

    # DataLoader：把数据集按 batch 分批、打乱后喂给模型
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = AlexNet(num_classes=5, init_weights=True)  # 创建网络：识别 5 种花

    net.to(device)  # 把模型搬到 GPU/CPU 上
    loss_function = nn.CrossEntropyLoss()  # 交叉熵损失：分类任务的标准选择
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)  # Adam 优化器：负责更新权重

    epochs = 10
    save_path = './AlexNet.pth'
    best_acc = 0.0  # 记录历史最佳准确率
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()  # 训练模式（开启 Dropout）
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            # 训练五步曲：清空梯度 → 前向传播 → 算损失 → 反向传播 → 更新参数
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # 累加损失，用于打印每轮平均损失

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()  # 验证模式（关闭 Dropout）
        acc = 0.0  # 每轮累计预测正确的个数
        with torch.no_grad():  # 验证阶段不更新参数，省内存
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]  # 取得分最高的类别作为预测结果
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()  # 统计预测正确个数

        val_accurate = acc / val_num  # 准确率 = 预测正确数 / 总验证数
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:  # 准确率创新高就保存模型
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
