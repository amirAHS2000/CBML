#!/bin/bash

OUT_DIR="output-resnet50-cars196-proxynca"
if [[ ! -d "${OUT_DIR}" ]]; then
    echo "Creating output dir for training : ${OUT_DIR}"
    mkdir ${OUT_DIR}
fi
CUDA_VISIBLE_DEVICES=0 python3 ./tools/main.py --cfg ./configs/proxynca/cfg_resnet50_proxynca_cars.yaml

CUDA_VISIBLE_DEVICES=0 python3 ./tools/main.py --cfg ./configs/proxynca/cfg_resnet50_proxynca_cars_test.yaml --phase test