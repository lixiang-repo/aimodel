#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}

exit 0
TF_CONFIG='{}'
python3 ${code_dir}/main.py --ckpt_dir "${model_dir}/ckpt" --export_dir ${model_dir}/export_dir --mode export


python3 ${code_dir}/common/warmup.py
python3 ${code_dir}/upload.py
