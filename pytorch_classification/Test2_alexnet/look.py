# 数据预览脚本：加载数据并随机展示几张验证集图片（对应 train.py 里被注释掉的那段）
import os  # 导入os模块，用于文件路径操作
import sys  # 导入sys模块
import json  # 导入json模块，用于读写json文件

import torch  # 导入PyTorch
import torch.nn as nn  # 导入神经网络模块
from torchvision import transforms, datasets, utils  # 导入数据预处理、数据集、工具
import matplotlib.pyplot as plt  # 导入绘图库
import numpy as np  # 导入numpy
import torch.optim as optim  # 导入优化器
from tqdm import tqdm  # 导入进度条库

from model import AlexNet  # 导入自定义的AlexNet模型

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

data_root = os.path.abspath(os.path.join(
    os.getcwd(), "../.."))  # get data root path
image_path = os.path.join(data_root, "data_set",
                          "flower_data")  # flower data set path
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                     transform=data_transform["train"])
train_num = len(train_dataset)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32
# number of workers
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
print('Using {} dataloader workers every process'.format(nw))

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=nw)

validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=4, shuffle=True,
                                              num_workers=nw)

print("using {} images for training, {} images for validation.".format(train_num,
                                                                       val_num))
test_data_iter = iter(validate_loader)
test_image, test_label = next(test_data_iter)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
imshow(utils.make_grid(test_image))
