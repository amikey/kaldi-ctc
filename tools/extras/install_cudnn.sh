#!/bin/bash

# https://developer.nvidia.com/rdp/cudnn-download

CUDA_VERSION=$(nvcc -V | tr '.,' '_ ' | awk '/release/{sub(/.*release/,""); print $1;}') # MAJOR_MINOR,
if [ -z "$CUDA_VERSION" ] ; then
  echo "Cannot figure out CUDA_VERSION from the nvcc output. Either your CUDA is too new or too old."
  exit 1
fi

function InstallCUDNN8 {
    if [ ! -f cudnn-8.0-linux-x64-v5.1-tgz ]; then
      wget -T 10 -t 3 https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v5.1/prod/8.0/cudnn-8.0-linux-x64-v5.1-tgz || exit 1;
    fi

    tar -xvzf cudnn-8.0-linux-x64-v5.1-tgz --transform 's/cuda/cudnn/'
}

function InstallCUDNN7 {
    if [ ! -f cudnn-7.5-linux-x64-v5.1-tgz ]; then
      wget -T 10 -t 3 https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v5.1/prod/7.5/cudnn-7.5-linux-x64-v5.1-tgz || exit 1;
    fi

    tar -xvzf cudnn-7.5-linux-x64-v5.1-tgz --transform 's/cuda/cudnn/'
}

case $CUDA_VERSION in
7_5) InstallCUDNN7;;
8_*) InstallCUDNN8 ;;
*) echo "CUDA is too old."; exit 1 ;;
esac


echo >&2 "Installation of cuDNN finished successfully"

