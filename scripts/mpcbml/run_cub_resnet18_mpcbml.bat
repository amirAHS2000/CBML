@echo off
set OUT_DIR=output-resnet18-cub-mpcbml

if not exist "%OUT_DIR%" (
    echo Creating output dir for training : %OUT_DIR%
    mkdir "%OUT_DIR%"
)

set CUDA_VISIBLE_DEVICES=0
python ./tools/main.py --cfg ./configs/mpcbml/cfg_resnet18_mpcbml_cub.yaml

python ./tools/main.py --cfg ./configs/mpcbml/cfg_resnet18_mpcbml_cub_test.yaml --phase test