#!/bin/sh
#SBATCH -N 1  # nodes requested
#SBATCH -n 1  # tasks requested
#SBATCH --job-name=label_distillation
#SBATCH --gres=gpu:1
#SBATCH --partition=PGR-Standard
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=1-21:11:11

export CUDA_HOME=/opt/cuda-10.0.130/

export CUDNN_HOME=/opt/cuDNN-7.6.0.64_10.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

source /home/${STUDENT_ID}/miniconda3/bin/activate meta_learning_pytorch_env_2

python main.py --mode distill_basic --dataset Cifar10 --arch ResNet --distill_lr 0.001