#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}

TF_CONFIG='{}'

time_str=$1
ckpt=$(date -d "${time_str:0:8} ${time_str:8:2} - ${infer_delta} hours" +"%Y%m%d%H%M")

python3 ${code_dir}/main.py --ckpt_dir "${model_dir}/${ckpt}" --mode eval --time_str "${time_str}" > ${model_dir}/logs/${time_str}.feature_eval 2>&1
rm -rf ${model_dir}/${ckpt}/join/eval


grep 'INFO:tensorflow:Saving dict for global step' ${model_dir}/logs/${time_str}.feature_eval|awk -F': ' '{print $2}' > ${model_dir}/logs/feature_eval






