import os  # 操作系统接口
import json  # JSON数据处理

import torch  # PyTorch
from PIL import Image  # 图像处理
from torchvision import transforms  # 图像预处理
import matplotlib.pyplot as plt  # 绘图显示

from model import vgg  # 导入自定义VGG模型


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备(GPU/CPU)

    data_transform = transforms.Compose(  # 数据预处理
        [transforms.Resize((224, 224)),  # 缩放为224x224
         transforms.ToTensor(),  # 转为张量
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 归一化到[-1,1]

    # load image
    img_path = "../tulip.jpg"  # 待预测图片路径
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)  # 校验文件存在
    img = Image.open(img_path)  # 打开图片
    plt.imshow(img)  # 显示原始图片
    # [N, C, H, W]
    img = data_transform(img)  # 预处理
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)  # 增加batch维度

    # read class_indict
    json_path = './class_indices.json'  # 类别索引文件路径
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)  # 校验文件存在

    with open(json_path, "r") as f:  # 打开json文件
        class_indict = json.load(f)  # 加载类别索引字典
    
    # create model
    model = vgg(model_name="vgg16", num_classes=5).to(device)  # 创建模型并转到设备
    # load model weights
    weights_path = "./vgg16Net.pth"  # 权重文件路径
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)  # 校验存在
    model.load_state_dict(torch.load(weights_path, map_location=device))  # 加载权重

    model.eval()  # 切换到评估模式
    with torch.no_grad():  # 不计算梯度
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()  # 前向传播并去掉batch维度
        predict = torch.softmax(output, dim=0)  # softmax转为概率
        predict_cla = torch.argmax(predict).numpy()  # 取概率最大的类别索引

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],  # 格式化预测结果
                                                 predict[predict_cla].numpy())
    plt.title(print_res)  # 标题显示预测结果
    for i in range(len(predict)):  # 遍历所有类别
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],  # 打印每个类别概率
                                                  predict[i].numpy()))
    plt.show()  # 显示图像


if __name__ == '__main__':  # 主程序入口
    main()  # 调用main函数