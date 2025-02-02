# CUDA 11.3とPython 3.9ベースイメージを使用
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="3.7;5.0;6.0;7.0;7.5;8.0;8.6+PTX"
ENV CUB_HOME="/usr/local/cub"

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    ninja-build \
    gcc \
    g++ \
    cmake \
    wget \
    && rm -rf /var/lib/apt/lists/*

# CUBのインストール
RUN wget https://github.com/NVIDIA/cub/archive/refs/tags/1.8.0.tar.gz \
    && tar xzf 1.8.0.tar.gz \
    && mv cub-1.8.0 /usr/local/cub \
    && rm 1.8.0.tar.gz

# 作業ディレクトリの作成
WORKDIR /app

# pipのダウングレード（古いバージョンのパッケージをインストールするため）
RUN pip3 install --upgrade "pip<24.0"

# Pythonパッケージのインストール
RUN pip3 install \
    tqdm==4.64.0 \
    pandas==1.4.2 \
    torch==1.12.0 \
    torchvision \
    torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu113

# PyTorch3Dのインストール（GPU対応版）
RUN pip3 install \
    fvcore \
    iopath \
    && pip3 install --no-binary pytorch3d "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.0"

# 追加の深層学習ライブラリ
RUN pip3 install monai==0.7.0
RUN pip3 install torchmetrics==0.11.4
RUN pip3 install pytorch-lightning==1.7.7
RUN pip3 install \
    PyYAML>=5.1 \
    tqdm>=4.57.0 \
    fsspec[http]>=2021.05.0 \
    packaging>=17.1 \
    typing-extensions>=4.0.0 \
    lightning-utilities>=0.3.0

# VTKのインストール
RUN pip3 install vtk

# データ用のボリュームを作成
VOLUME ["/data/input", "/data/output", "/data/model"]

# アプリケーションコードのコピー
COPY CrownSegmentation /app/CrownSegmentation
COPY CrownSegmentationcli /app/CrownSegmentationcli

# 実行権限の付与
RUN chmod +x /app/CrownSegmentationcli/CrownSegmentationcli.py


# 実行コマンド例
# docker build -t dental-seg dockerfile/Dockerfile .
#
# docker run --rm --gpus all \
#   -v /path/to/input/data:/data/input \
#   -v /path/to/output:/data/output \
#   -v /path/to/model:/data/model \
#   dental-seg python3 CrownSegmentationcli/CrownSegmentationcli.py \
#   --input /data/input/your_model.vtk \
#   --output /data/output/result.vtk \
#   --model /data/model/segmentation_model.pth \

CMD ["python3", "CrownSegmentationcli.py"]