#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}

sh stop.sh

# 0-6
# 7-14
# 15-22
# # 23-30
# for i in `seq 11 14`
# #for i in `seq 32 63`
# do
#     nohup sh -x run.sh worker $i > ${model_dir}/logs/log.worker$i 2>&1 &
# done

i=3
nohup sh -x run.sh worker $i > ${model_dir}/logs/log.worker$i 2>&1 &