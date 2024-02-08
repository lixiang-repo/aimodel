#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}


ckpt=$1
time_str=$(python3 -c "from dateutil.parser import parse;import datetime;print((parse(str("${ckpt}")) + datetime.timedelta(hours=${delta})).strftime('%Y%m%d%H%M'))")

sh ${code_dir}/bin/stop.sh
python3 ${code_dir}/main.py --model_dir "${model_dir}/${ckpt}/join" --mode feature_eval --type "join" \
    --data_path "${nas_path}" --warm_path "${model_dir}/${ckpt}/update" \
    --time_str "${time_str}" --time_format "${time_format}" > ${model_dir}/logs/${ckpt}.feature_eval 2>&1
rm -rf ${model_dir}/${ckpt}/join/eval


grep 'INFO:tensorflow:Saving dict for global step' ${model_dir}/logs/${ckpt}.feature_eval|awk -F': ' '{print $2}' > ${model_dir}/logs/feature_eval






