# qlib_benchmark Docker 镜像
# 用于运行各种基准测试模型（XGBoost, LightGBM, GRU, MLP, KRNN, Linear 等）

FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.1.1-runtime-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 设置工作目录
WORKDIR /workspace

# 使用阿里云 apt 镜像源（加速软件包下载）
RUN sed -i 's@archive.ubuntu.com@mirrors.aliyun.com@g' /etc/apt/sources.list && \
    sed -i 's@security.ubuntu.com@mirrors.aliyun.com@g' /etc/apt/sources.list

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置软链接
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# 配置 pip 源
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 config set install.trusted-host pypi.tuna.tsinghua.edu.cn    

# 升级 pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# 注意：qlib_benchmark 的 requirements.txt 中包含旧版本的 torch==1.7.0
# 如果需要使用 GPU，可能需要安装兼容的 PyTorch 版本
# 这里先安装 CPU 版本的 PyTorch 1.7.0，如果需要 GPU 版本可以后续调整
# 或者使用更新的 PyTorch 版本（如果代码兼容）
# RUN pip3 install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torch torchvision torchaudio 
# 复制 requirements.txt 并安装依赖
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# 复制 qlib_benchmark 源码到容器
# 注意：实际运行时会通过 volume 挂载最新代码
COPY . /workspace

# 设置 Python 路径（确保能导入 qlib 和其他模块）
ENV PYTHONPATH=/workspace:${PYTHONPATH}

# 创建输出目录（如果需要）
RUN mkdir -p /workspace/outputs

# 默认命令（可由 docker run 覆盖）
CMD ["python3", "--version"]

