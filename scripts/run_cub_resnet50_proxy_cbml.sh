#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python tools/main.py --cfg configs/example_resnet50_proxy_cbml.yaml --phase train
python tools/main.py --cfg configs/example_resnet50_proxy_cbml_test.yaml --phase test
