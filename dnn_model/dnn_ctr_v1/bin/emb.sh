#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}

TF_CONFIG='{}'

python3 ${code_dir}/main.py --ckpt_dir "${model_dir}/${1}" --mode emb --time_str "${time_str}"