#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && pwd)"

# clear serving model
base_path=/data/lixiang/model
tasks=$(ls ${base_path})
for task in ${tasks[@]}
do
    cd ${base_path}/${task}/export_dir
    num=`ls|wc -l`
    if [ $(expr ${num} - 3) -ge 0 ]; then
        ls|head -n `expr ${num} - 3` |xargs rm -rf
    fi
    cd -
done

# clear hdfs
start_date=`date -d "90 days ago" +%Y%m%d`
# start_date=20231201
end=`date -d "60 days ago" +%Y%m%d`
while [[ ${start_date} != ${end} ]]
do
    # hadoop fs -rmr 
    # rm -rf /data/lixiang/model
    start_date=$(date -d "${start_date/\// } +1 days" +%Y%m%d)
done

base_data_hdfs=/user/hive/warehouse/lixiang.db


