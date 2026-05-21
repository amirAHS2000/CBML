#!/bin/bash

OUT_DIR="output-googlenet-cub"
if [[ ! -d "${OUT_DIR}" ]]; then
    echo "Creating output dir for training : ${OUT_DIR}"
    mkdir ${OUT_DIR}
fi
CUDA_VISIBLE_DEVICES=0 python3 ./tools/main.py --cfg ./configs/mpcbml/cfg_googlenet_mpcbml_cub.yaml

CUDA_VISIBLE_DEVICES=0 python3 ./tools/main.py --cfg ./configs/mpcbml/cfg_googlenet_mpcbml_cub_test.yaml --phase test