FROM --platform=linux/x86_64 nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG PYTHON_VERSION=3.11
ARG PACKAGES="git curl ca-certificates vim wget unzip build-essential cmake jq"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=on

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv f23c5a6cf475977595c89f51ba6932366a755776 \
 && echo "deb http://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main" > /etc/apt/sources.list.d/python.list \
 && echo "deb-src http://ppa.launchpad.net/deadsnakes/ppa/ubuntu jammy main" >> /etc/apt/sources.list.d/python.list

RUN apt-get update \
 && apt-get install -y --no-install-recommends ${PACKAGES} python${PYTHON_VERSION} \
 && ln -nfs /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
 && ln -nfs /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
 && rm -rf /var/lib/apt/lists/* \
 && curl -sSL https://bootstrap.pypa.io/get-pip.py | python -
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

WORKDIR /app
COPY ./requirements.txt ./
RUN pip install packaging
RUN pip install -r requirements.txt
# RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('guillaumekln/faster-whisper-large-v2')"
# ENV HF_HUB_OFFLINE=1

# CMD ["python", "app.py", "--input_audio_max_duration", "-1", "--server_name", "0.0.0.0", "--auto_parallel", "True", "--default_model_name", "large-v2"]
