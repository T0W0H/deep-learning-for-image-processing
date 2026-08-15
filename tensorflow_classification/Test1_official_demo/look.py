from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf  # TensorFlow 主库（约定俗成简写成 tf）
from model import MyModel  # 从 model.py 导入我们自己定义的网络
import numpy as np
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


imgs = x_test[:3]
labs = y_test[:3]
print(labs)
plot_imgs = np.hstack(imgs)
plt.imshow(plot_imgs, cmap='gray')
plt.show()
