# 华为云镜像：CUDA 12.4 + Ubuntu 22.04 (devel)
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 清华源
RUN echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list

# 基础依赖和OpenGL库
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    wget \
    bzip2 \
    openssh-server \
    sudo \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    build-essential \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装Miniconda（清华镜像）
RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda3 && \
    rm /tmp/miniconda.sh

# Miniconda和CUDA环境变量
ENV PATH="/opt/miniconda3/bin:/usr/local/cuda-12.4/bin:${PATH}" \
    CUDA_HOME="/usr/local/cuda-12.4" \
    LD_LIBRARY_PATH="/opt/miniconda3/envs/ai/lib:/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}" \
    CONDA_ALWAYS_YES=true \
    CONDA_AUTO_ACTIVATE_BASE=false

RUN /opt/miniconda3/bin/conda init bash

# 创建ai虚拟环境
RUN /opt/miniconda3/bin/conda create --name ai python=3.10 -c conda-forge --override-channels -y

# 接受Anaconda服务条款
RUN /opt/miniconda3/bin/conda run --no-capture-output -n ai conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN /opt/miniconda3/bin/conda run --no-capture-output -n ai conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 安装PyTorch 2.5.1 (CUDA 12.4)
RUN /opt/miniconda3/bin/conda run --no-capture-output -n ai conda install -y pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

# 固定MKL版本，避免iJIT_NotifyEvent问题；此步连带的 numpy 无需理会——
# 下一步 pip 按 requirements.txt 重装 numpy，以 pip 为准。clean 掉 conda
# 包缓存（tarballs + pkgs 目录，torch 链有好几个 GB），镜像显著减重。
RUN /opt/miniconda3/bin/conda run --no-capture-output -n ai conda install -y "mkl<2025" "mkl-service<2025" && \
    /opt/miniconda3/bin/conda clean -afy

# 验证PyTorch
RUN /opt/miniconda3/bin/conda run --no-capture-output -n ai python -c "import torch, torchvision; print(f'torch {torch.__version__} | torchvision {torchvision.__version__} | CUDA {torch.version.cuda}')"

# 运行环境变量（放在 conda 大层之后：改这些不会作废上面几个 GB 的层缓存）
# pip 默认清华源（构建与容器内一致，容器里手敲 pip 不用再带 -i）
# PYTHONUNBUFFERED：训练日志实时落盘，tail -f 不等缓冲
# YOLO_AUTOINSTALL=0：禁止 ultralytics 训练/导出时 AutoUpdate 偷偷 pip install 改动环境
#   （2026-09-11 numpy 被静默顶到 2.2.6 的根源；依赖已全部预装进 requirements.txt，
#    此开关兜底——真缺包就构建期/启动期大声报错，而不是悄悄换版本）
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    YOLO_AUTOINSTALL=0

# 安装 YOLO 源码运行的 Python 依赖（ultralytics 本体不安装，始终以挂载源码方式运行）。
# requirements 单独成层：日常改依赖只重建本层，上面几个 GB 的 conda 层直接命中缓存。
# --no-cache-dir 不留 pip 缓存；pip check 让 pin 冲突在构建期爆出来而不是训练期。
COPY requirements.txt /tmp/requirements.txt
RUN /opt/miniconda3/bin/conda run --no-capture-output -n ai pip install --no-cache-dir -r /tmp/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    /opt/miniconda3/envs/ai/bin/pip check && \
    rm /tmp/requirements.txt

# WeDetect 依赖。仓库中的 requirements.txt 主要服务于 SRT/ONNX 流程，
# 不包含 WeDetect 的 OpenMMLab 检测框架和视觉语言模型依赖。
RUN /opt/miniconda3/bin/conda run --no-capture-output -n ai pip install --no-cache-dir \
    "transformers==4.57.1" \
    "trl==0.17.0" \
    "accelerate==1.10.0" \
    "mmengine==0.10.7" \
    "timm" \
    "supervision==0.19.0" \
    "pycocotools" \
    "terminaltables" \
    "jsonlines" \
    "tabulate" \
    "lvis" \
    "webdataset" \
    "ddd-dataset" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    /opt/miniconda3/envs/ai/bin/pip check

# OpenMMLab 组件。当前 CUDA 12.4 + PyTorch 2.5.1 组合通常没有 mmcv 2.1.0
# 的预编译 wheel，因此关闭 PEP 517 隔离环境，使用容器内已有的 setuptools/pkg_resources
# 从源码编译 mmcv；这样也避免新版构建隔离环境移除了 pkg_resources。
RUN /opt/miniconda3/bin/conda run --no-capture-output -n ai pip install --no-cache-dir \
    "setuptools==70.3.0" "wheel" -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    /opt/miniconda3/bin/conda run --no-capture-output -n ai pip install --no-cache-dir --no-build-isolation \
    "mmcv==2.1.0" \
    -f https://download.openmmlab.com/mmcv/dist/cu124/torch2.5.0/index.html && \
    /opt/miniconda3/bin/conda run --no-capture-output -n ai pip install --no-cache-dir "mmdet==3.3.0" -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    /opt/miniconda3/envs/ai/bin/pip check

# 将源码放入镜像；模型权重建议运行时挂载到 checkpoints/，避免写入镜像层。
WORKDIR /workspace/wedetect
COPY . /workspace/wedetect/
ENV PYTHONPATH="/workspace/wedetect:${PYTHONPATH}"

# onnxruntime-gpu 运行库兜底：conda 安装的 torch 把 cuDNN/cuBLAS 放 env/lib（已在
# LD_LIBRARY_PATH）；若镜像迭代改为 pip 轮子 torch（库在 site-packages/torch/lib），
# 软链到 env/lib 保证 ORT 的 dlopen 找得到。已存在则软链无副作用。
RUN TL=$(/opt/miniconda3/envs/ai/bin/python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))") && \
    for pat in 'libcudnn*' 'libcublas*' 'libcufft*' 'libcurand*' 'libnvrtc*' 'libnvJitLink*'; do \
        ln -sf $TL/$pat /opt/miniconda3/envs/ai/lib/ 2>/dev/null || true; \
    done; \
    /opt/miniconda3/envs/ai/bin/python -c "import onnxruntime as ort; ps=ort.get_available_providers(); assert 'CUDAExecutionProvider' in ps, f'缺 CUDA EP: {ps}'; print(f'onnxruntime-gpu OK: {ort.__version__} | {ps}')"

# 终检：同时覆盖 SRT/ONNX 链和 WeDetect 推理链。
RUN /opt/miniconda3/bin/conda run --no-capture-output -n ai python -c "import torch, torchvision, numpy, cv2, onnx, onnxslim, ml_dtypes, albumentations, transformers, timm, supervision, mmcv, mmengine, mmdet, wedetect; print(f'torch {torch.__version__} | numpy {numpy.__version__} | cv2 {cv2.__version__} | onnx {onnx.__version__} | mmcv {mmcv.__version__} | mmengine {mmengine.__version__} | mmdet {mmdet.__version__}')"

# SSH配置
RUN echo 'Port 22' > /etc/ssh/sshd_config && \
    echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config && \
    echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config && \
    echo 'UsePAM no' >> /etc/ssh/sshd_config && \
    echo 'X11Forwarding yes' >> /etc/ssh/sshd_config

# 配置SSH服务
RUN mkdir /var/run/sshd && \
    echo 'root:123' | chpasswd && \
    chmod 600 /etc/ssh/sshd_config && \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH ssh-keygen -A

# 配置环境变量（交互shell自动激活ai环境）
RUN echo 'export CUDA_HOME=/usr/local/cuda-12.4' >> /root/.bashrc && \
    echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> /root/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/opt/miniconda3/envs/ai/lib:/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> /root/.bashrc && \
    echo 'export PATH=/opt/miniconda3/bin:$PATH' >> /root/.bashrc && \
    echo 'conda activate ai' >> /root/.bashrc

EXPOSE 22

# 启动脚本
RUN mkdir -p /app && \
    echo '#!/bin/bash' > /app/start.sh && \
    echo 'service ssh start' >> /app/start.sh && \
    echo 'ps aux | grep sshd' >> /app/start.sh && \
    echo 'tail -f /dev/null' >> /app/start.sh && \
    chmod +x /app/start.sh

CMD ["bash", "/app/start.sh"]

# 用法（在本 Srt/Docker 目录下执行；/work 整体映射进容器，路径与宿主机一致）：
# docker build -t srtcvimage:latest .
# ./run.sh 3428 zhn    # 创建自己的容器（端口、容器名必填；多卡训练参数见 run.sh）
# ssh root@localhost -p 3428 (密码123) -> cd /work/disk1/02.Code/srtcv/srt_sop_train
