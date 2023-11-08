#!/bin/bash

export PYTHONPATH=./:../../third_party/pytorch-attention/

python3 pytorch_fourier_train.py \
    --train_data_folder /media/alexey/SSDData/datasets/denoising_dataset/train/ \
    --validation_data_folder /media/alexey/SSDData/datasets/denoising_dataset/val/ \
    --synthetic_data_paths /media/alexey/SSDData/datasets/denoising_dataset/base_clear_images/ \
    --epochs 150 \
    --lr_milestones 0 \
    --image_size 512 \
    --batch_size 4 \
    --visdom 9001 \
    --njobs 4 \
    --exp /media/alexey/SSDData/experiments/denoising/fftcnn_exp/ \
    --preload_datasets \
    # --load /media/alexey/SSDData/experiments/denoising/fftcnn_exp/checkpoints/best.trh
