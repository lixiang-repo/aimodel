#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}
sh stop.sh

# for i in `seq 0 0`
# #for i in `seq 32 63`
# do
#     nohup sh -x run.sh worker $i > ${model_dir}/logs/log.worker$i 2>&1 &
# done

nohup sh run.sh ps 0 > ${model_dir}/logs/log.ps0 2>&1 &

sleep  10
nohup sh -x run.sh chief 0 > ${model_dir}/logs/log.chief0 2>&1 &
