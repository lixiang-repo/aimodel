#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}

sh stop.sh

# 0-2
# 3-6
# 7-10
# 11-14
for i in `seq 0 1`
do
    nohup sh -x run.sh worker $i > ${model_dir}/logs/log.worker$i 2>&1 &
done
