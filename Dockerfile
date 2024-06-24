FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

WORKDIR /work
COPY . /work

RUN apt-get update && \
    apt-get install -y libgl1-mesa-dev python3 python3-pip

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt