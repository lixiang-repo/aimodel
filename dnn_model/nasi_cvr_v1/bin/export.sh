#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}

sh ${code_dir}/bin/stop.sh
TF_CONFIG='{}'
time_str=`awk '{print $1}' ${model_dir}/donefile|tail -1`
# time_str=202312022359

cp -r ${model_dir}/${time_str} ${model_dir}/${time_str}_tmp
python3 ${code_dir}/main.py --model_dir ${model_dir}/${time_str}_tmp/join --mode export --type join \
  --data_path "${nas_path}" --warm_path ${model_dir}/${time_str}_tmp/update \
  --time_str "${time_str}" --time_format "${time_format}"


rm -rf ${model_dir}/${time_str}_tmp


