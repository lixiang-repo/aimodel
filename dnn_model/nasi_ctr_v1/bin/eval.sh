#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}


time_str=$1
ckpt=$(python3 -c "from dateutil.parser import parse;import datetime;print((parse(str("${time_str}")) - datetime.timedelta(hours=${infer_delta})).strftime('%Y%m%d%H%M'))")

python3 ${code_dir}/main.py --model_dir "${model_dir}/${ckpt}/join" --mode eval --type "join" \
    --data_path "${nas_path}" --warm_path "${model_dir}/${ckpt}/update" \
    --time_str "${time_str}" --time_format "${time_format}" > ${model_dir}/logs/${time_str}.eval 2>&1
rm -rf ${model_dir}/${ckpt}/join/eval

msg=$(grep 'INFO:tensorflow:Saving dict for global step' ${model_dir}/logs/${time_str}.eval|awk -F': ' '{print $2}')
echo "task = $(basename ${code_dir}), time = ${time_str}, ${msg}" >> ${model_dir}/logs/eval


