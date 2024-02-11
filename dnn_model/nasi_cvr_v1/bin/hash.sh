#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}

time_str=$1

python3 ${code_dir}/main.py --model_dir "${model_dir}/ckpt/update" --mode hash \
    --data_path "${nas_path}" --time_str "${time_str}" --time_format "${time_format}"


