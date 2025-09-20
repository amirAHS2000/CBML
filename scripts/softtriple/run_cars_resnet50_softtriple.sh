#!/bin/bash

OUT_DIR="output-resnet50-cars196-softtriple"
if [[ ! -d "${OUT_DIR}" ]]; then
    echo "Creating output dir for training : ${OUT_DIR}"
    mkdir ${OUT_DIR}
fi
CUDA_VISIBLE_DEVICES=0 python3 ./tools/main.py --cfg ./configs/softtriple/cfg_resnet50_softtriple_cars.yaml

CUDA_VISIBLE_DEVICES=0 python3 ./tools/main.py --cfg ./configs/softtriple/cfg_resnet50_softtriple_cars_test.yaml --phase test