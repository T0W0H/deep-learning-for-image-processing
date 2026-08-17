import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 预处理：和训练时保持一致（缩放 → 转张量 → 归一化）
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 加载图片
    img_path = "../tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    img = data_transform(img)  # 图片 → 张量 [C, H, W]
    img = torch.unsqueeze(img, dim=0)  # 加上 batch 维度 → [1, C, H, W]，网络一次处理一批图

    # 读取类别映射 {索引: 类别名}
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 创建模型（5 类），并搬到 GPU/CPU
    model = AlexNet(num_classes=5).to(device)

    # 加载训练好的权重
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()  # 推理模式（关闭 Dropout）
    with torch.no_grad():  # 推理不计算梯度
        # 预测：模型输出 → 概率 → 取最大的类别
        output = torch.squeeze(model(img.to(device))).cpu()  # 去掉 batch 维度
        predict = torch.softmax(output, dim=0)  # 得分转成概率
        predict_cla = torch.argmax(predict).numpy()  # 概率最大的类别

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
