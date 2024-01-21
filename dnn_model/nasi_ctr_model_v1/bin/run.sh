#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}

while [ "${last_time}" != "${end_date}" ]; do
  last_time=$(awk '{print $1}' ${donefile} | tail -1)
  if [ "${last_time}" = "" ]; then
    last_time=${start_date}
    warm_path=""
  else
    warm_path="${model_dir}/ckpt/update"
  fi
  time_str=$(python3 -c "from dateutil.parser import parse;import datetime;print((parse(str("${last_time}")) + datetime.timedelta(hours=${delta})).strftime('%Y%m%d%H%M'))")
  
  ###########################join#################################
  sh ${code_dir}/bin/stop.sh
  python3 ${code_dir}/main.py --model_dir "${model_dir}/ckpt/join" --mode train --type "join" --warm_path "${warm_path}" \
    --data_path "${nas_path}" --time_str "${time_str}" --time_format "${time_format}"
  ##########################update##################################
  sh ${code_dir}/bin/stop.sh
  python3 ${code_dir}/main.py --model_dir "${model_dir}/ckpt/update" --mode train --type "update" \
    --data_path "${nas_path}" --time_str "${time_str}" --time_format "${time_format}"
  ############################################################
  #backup and export serving model
  if [ ${time_str} -ge 202301012359 ]; then
    #cd ${code_dir}/bin && sh -x export.sh && cd -
    rm -rf ckpt/*/events.out.tfevents* ckpt/*/eval
    cp -r ckpt ${time_str}

    cd ${code_dir}/bin && sh -x eval.sh ${last_time} && cd -
    
  fi
done
