#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}

## ps $1 index $2
job=$1
id=$2
export TF_CONFIG="$(python3 ${code_dir}/tfconfig.py $job $id)"
task_idx=$(python3 -c "print($2 + 1) if \"$1\" == \"worker\" else print($2)")
donefile=${model_dir}/logs/donefile.${task_idx}
touch ${donefile}
############################################################
time_str=$(date -d "${start_date:0:8} ${start_date:8:2} - ${delta} hours" +"%Y%m%d%H%M")
while true; do
  time_str=$(date -d "${time_str:0:8} ${time_str:8:2} + ${delta} hours" +"%Y%m%d%H%M")
  if [ $(grep ${time_str:0:8} ${donefile} |wc -l |awk '{print $1}') -ge 1 ];then continue; fi
  python3 ${code_dir}/main.py --job_name ${job}.${id} --ckpt_dir ${model_dir}/ckpt --time_str ${time_str}
  exit 0
  if [ "$(tail -1 ${donefile} |awk '{print $1}')" != "${time_str}" ];then
    time_str=$(date -d "${time_str:0:8} ${time_str:8:2} - ${delta} hours" +"%Y%m%d%H%M")
    sleep 1
    continue
  fi
  ############################################################
  #backup and export serving model
  if [ ${time_str:0:8} -ge ${backup} -a "${job}" == "chief" -a ${id} -eq 0 ]; then
    cd ${model_dir}
    rm -rf ckpt/events.out.tfevents* ckpt/eval && cp -r ckpt ${time_str}

    cd ${code_dir}/bin && sh -x export.sh && cd -
    # cd ${code_dir}/bin && sh -x eval.sh ${time_str} && cd -
    # cd ${code_dir}/bin && sh -x infer.sh ${time_str} && cd -
  fi
done
