#!/bin/bash

OUT_DIR="output-resnet50-cub-contrastive"
if [[ ! -d "${OUT_DIR}" ]]; then
    echo "Creating output dir for training : ${OUT_DIR}"
    mkdir ${OUT_DIR}
fi
CUDA_VISIBLE_DEVICES=0 python3 ../tools/main.py --cfg ../configs/cfg_resnet50_contrastive_cub.yaml

CUDA_VISIBLE_DEVICES=0 python3 ../tools/main.py --cfg ../configs/cfg_resnet50_contrastive_cub_test.yaml --phase test