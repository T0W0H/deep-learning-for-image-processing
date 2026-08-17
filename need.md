# 巨无霸深度学习环境一键搭建脚本 (dl_env)

> 适用: 《深度学习之图像处理》教程环境
> GPU: NVIDIA GeForce GTX 1650 (4GB) | 驱动 530.30.02 (支持 CUDA 12.1)
> 说明: 本脚本修复了原版 `need.md` 的 numpy 冲突 bug, 并补齐全部可选组件。
> 用法: 逐段复制到终端执行, 每段可断点续跑 (幂等)。

---

## Step 0: 前置检查 (可选, 快速自检)

```bash
nvidia-smi | head -15
df -h /root | tail -1      # 需要 > 20GB 可用
free -h | head -2
```

---

## Step 1: 创建并激活环境 (仅首次)

```bash
mamba create -n dl_env python=3.10 -y
conda activate dl_env
```

> 注意: 原脚本第 3 行 `mamba activate dl_full` 是笔误, 应为 `dl_env`。

---

## Step 2: 锁 numpy 1.23.5 (关键修复!)

> ⚠️ 原环境装了 numpy 2.2.6, 导致 TensorFlow 2.11 import 崩溃:
> `TypeError: Unable to convert function return value to a Python type`
> TF 2.11 只兼容 numpy<2, 必须锁回 1.23.5。

```bash
conda activate dl_env
pip install "numpy==1.23.5"
```

---

## Step 3: 同步降级兼容 numpy 1.x 的科学计算库

> 当前 scipy 1.15 / pandas 2.3 / sklearn 1.7 / matplotlib 3.10 均为 numpy 2 时代版本,
> 降 numpy 后必须同步降级, 否则 import 报错。

```bash
conda activate dl_env
pip install "scipy==1.10.1" "pandas==2.0.3" "scikit-learn==1.3.2" "matplotlib==3.7.5"
```

---

## Step 4: PyTorch 2.1.1 + torchvision (自带 CUDA 12.1 运行时)

```bash
conda activate dl_env
pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## Step 5: TensorFlow 2.11 + CUDA 11.2 运行时 (mamba 装)

```bash
conda activate dl_env
mamba install "cudatoolkit=11.2.2" "cudnn=8.4.1.50" "h5py=3.11.0" "numpy=1.23.5" -y
pip install "tensorflow==2.11.0"
```

> 说明: conda 会把 cudatoolkit/cudnn 装进 `$CONDA_PREFIX/lib`,
> TF 通过 LD_LIBRARY_PATH 找到它们 (见 Step 8)。
>
> ⚠️ **h5py 必须锁 3.11.0 (关键修复!)**: 新版 h5py (3.12+, 如 3.16) 已无
> Python 3.10 的预编译 wheel, pip 会现场源码编译并失败
> (`Failed to build wheel for h5py`, 需要 HDF5 头文件)。
> 且 mamba 单独装 h5py 时会把 numpy 解析到 2.2.6 (会破坏 TF),
> 必须在同一条命令里锁 `numpy=1.23.5`。
> h5py 3.11.0 满足 TF 2.11 的 `h5py>=2.9.0` 约束, pip 装 TF 时不会覆盖它。

---

## Step 6: TensorRT (可选, 消除 TF-TRT 警告)

> 事实: TF 2.11 二进制 dlopen 的是 `libnvinfer.so.7` (TRT 7 soname),
> 而 pip 的 `nvidia-tensorrt==8.4.1.5` 提供 `libnvinfer.so.8`。
> 标准安装无法直接满足, 需用软链接方案 (社区公认做法, 仅消除警告 + 提供 TRT8 API)。

```bash
conda activate dl_env
pip install "nvidia-tensorrt==8.4.1.5"
```

软链接 (让 TF 的 dlopen 找到 .so.7):

```bash
TRT_DIR=$(/root/miniforge3/envs/dl_env/bin/python -c "import tensorrt, os; print(os.path.dirname(tensorrt.__file__))")
ln -sf "$TRT_DIR/libnvinfer.so.8"     "$TRT_DIR/libnvinfer.so.7"
ln -sf "$TRT_DIR/libnvinfer_plugin.so.8" "$TRT_DIR/libnvinfer_plugin.so.7"
ln -sf "$TRT_DIR/libnvparsers.so.8"   "$TRT_DIR/libnvparsers.so.7"
ln -sf "$TRT_DIR/libnvonnxparser.so.8" "$TRT_DIR/libnvonnxparser.so.7"
export LD_LIBRARY_PATH="$TRT_DIR:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

> ⚠️ 若后续真正跑 TF-TRT 转换报错, 说明 TRT8 与 TRT7 API 不完全兼容,
> 删除软链接并忽略警告即可 (不影响训练)。

---

## Step 7: 通用依赖 + 教程章节依赖

> 🔒 所有 pip 命令都带 `"numpy<2"` 约束, 防止任何安装把 numpy 拉回 2.x (会破坏 TF)。

```bash
conda activate dl_env
pip install "numpy<2" matplotlib opencv-python pillow tqdm pandas scikit-learn scipy
pip install cython pycocotools     # 检测/分割章节 COCO 工具
pip install thop                    # model_complexity 章节 FLOPs
```

---

## Step 8: 环境变量 (写入 ~/.bashrc 使其永久生效)

```bash
grep -q 'CONDA_PREFIX/lib' ~/.bashrc || echo 'export LD_LIBRARY_PATH="/root/miniforge3/envs/dl_env/lib/python3.10/site-packages/tensorrt:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
```

> 说明: 固定把 TRT 目录 + conda 库目录写进 `LD_LIBRARY_PATH`,
> 这样 Step 6 的 tensorrt 与 Step 5 的 TF/PT 开箱即用。
> 若换了 env 路径, 改这一行的两个路径即可。

---

## Step 9: 巨无霸选件 (全部可选, 按需逐行执行)

> 🔒 每行都带 `"numpy<2"` 约束, 防止破坏 TF。

### A. 数据科学 / 表格
```bash
conda activate dl_env
pip install "numpy<2" xgboost lightgbm catboost
```

### B. 图像 / 计算机视觉
> ⚠️ `opencv-contrib-python` 与 `opencv-python` 互斥, 装 contrib 前先卸基础版:
```bash
pip uninstall -y opencv-python
pip install "numpy<2" opencv-contrib-python imgaug albumentations
```

### C. NLP / 文本
```bash
pip install "numpy<2" spacy nltk
python -m spacy download en_core_web_sm
```

### D. 深度学习工具链
> ⚠️ 必须用旧版! 新版 onnx (1.22) 需要 protobuf>=4.25, 会破坏 TF 2.11 (需 protobuf 3.19)。
> 已实测: onnx 1.12.0 + onnxruntime 与 protobuf 3.19.6 兼容。
> ⚠️⚠️ tensorboard-plugin-profile 必须装 2.11.2! 新版 2.23.1 依赖 xprof → 会拉 protobuf 到 7.x 炸掉 TF,
> 且拉入整个 google-cloud 孤儿链 (需手动清理)。2.11.2 依赖干净。
```bash
pip install "numpy<2" "onnx==1.12.0" onnxruntime
pip install "numpy<2" "tensorboard-plugin-profile==2.11.2"
```

### E. 可视化
```bash
pip install "numpy<2" seaborn plotly kaleido
```

### F. 交互 / Notebook
```bash
pip install "numpy<2" jupyter ipykernel jupyterlab
python -m ipykernel install --user --name dl_env --display-name "Python (dl_env)"
```

### G. 机器学习生态
> ⚠️ mlflow 必须用 2.x! mlflow 3.x 会把 protobuf 拉到 6.x, 直接炸掉 TF 2.11。
> 已实测: mlflow 2.8.1 + protobuf 3.19.6 兼容 (且需 setuptools<81, 否则缺 pkg_resources)。
```bash
pip install "numpy<2" shap optuna hyperopt
pip install "numpy<2" "mlflow==2.8.1" "setuptools<81"
```

### H. 常用工具
```bash
pip install "numpy<2" pyyaml json5 humanize psutil
```

---

## Step 10: 完整验证

```bash
conda activate dl_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

echo "=== 1. TensorFlow ==="
TF_CPP_MIN_LOG_LEVEL=1 python -c "import tensorflow as tf; print('TF', tf.__version__, tf.config.list_physical_devices('GPU'))"

echo "=== 2. PyTorch ==="
python -c "import torch, torchvision; print(torch.__version__, torch.cuda.is_available(), torchvision.__version__)"

echo "=== 3. Keras ImageDataGenerator ==="
python -c "from tensorflow.keras.preprocessing.image import ImageDataGenerator; print('ImageDataGenerator OK')"

echo "=== 4. numpy 版本 (必须 1.x) ==="
python -c "import numpy; print('numpy', numpy.__version__)"

echo "=== 5. protobuf 版本 (必须 3.19.6!) ==="
python -c "import google.protobuf; print('protobuf', google.protobuf.__version__)"

echo "=== 6. 可选组件 ==="
python -c "import cv2, xgboost, lightgbm, catboost, onnx, onnxruntime, spacy, nltk, seaborn, plotly, shap, optuna, mlflow; print('all optional OK')"
```

---

## 已知限制 (诚实说明)

1. **TF 2.11 与 PT 2.1.1 的 CUDA 版本不同**: TF 用 cu11.2 (conda), PT 用 cu12.1 (自带)。
   两者共存通过 `LD_LIBRARY_PATH` 实现, PT 自带库优先, 实测可同时工作。
2. **TensorRT 警告**: 不装 TRT 时, TF 启动打印两条 `libnvinfer.so.7` 警告, 属正常可选项缺失。
   装 TRT 8 + 软链接可消除, 但 TF-TRT 实际转换可能不完全兼容 (TRT8 vs TRT7 API)。
3. **4GB 显存限制**: 同时加载 TF + PT 大模型会 OOM, 建议单进程只用一个框架。
4. **numpy 必须保持 1.x (1.23.5 或 1.26.x)**: 任何 `pip install` 若带 numpy 2.x 依赖会重新破坏 TF,
   重装 numpy 后需重跑 Step 2。
5. **protobuf 必须保持 3.19.6 (最关键!)**: 
   - mlflow 3.x / 新版 onnx (>=1.14) / opentelemetry / tensorboard-plugin-profile 2.23.1 (xprof)
     会把 protobuf 拉到 5.x/6.x/7.x, 导致 TF 崩溃 (`Descriptors cannot be created directly`)。
   - 修复: `pip install "protobuf==3.19.6"`。
   - 兼容组合 (已实测): onnx==1.12.0 + mlflow==2.8.1 + tensorboard-plugin-profile==2.11.2
     + protobuf==3.19.6 + setuptools<81 + wheel<=0.41 + packaging==23.2。
   - 装任何新包后若 TF 报错, 先查 protobuf: `pip show protobuf | grep Version`, 非 3.19.6 就降回。
6. **setuptools 必须 <81**: 新版 setuptools 移除 pkg_resources, mlflow 2.x 依赖它报
   `No module named 'pkg_resources'`。修复: `pip install "setuptools<81"`。
7. **wheel/packaging 版本绑定**: mlflow 2.8.1 要 packaging<24, 新版 wheel 要 packaging>=24,
   必须 wheel<=0.41.2 + packaging==23.2 同时满足。
8. **装包后必查**: 任何 `pip install` 装完跑一遍 `pip check`, 若报 protobuf 相关冲突,
   立即 `pip install "protobuf==3.19.6"` 修复, 不要忽略 pip 的 conflict 警告
   (pip 会明知冲突仍强行安装)。
9. **h5py 必须保持 3.11.0**: 新版 h5py (3.12+/3.16) 没有 Python 3.10 的预编译 wheel,
   pip 只能源码编译 (`Failed to build wheel for h5py`)。且通过 mamba/pip 装新 h5py 会把
   numpy 拉到 2.x (破坏 TF)。装任何新包后若 TF 报错或 h5py 被升级, 修复:
   `mamba install "h5py=3.11.0" "numpy=1.23.5" -y`。
