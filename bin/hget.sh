#!/usr/bin/env bash

start_date=`date -d "15 days ago" +%Y%m%d`
# start_date=20231201
end=`date -d "0 days ago" +%Y%m%d`
tasks=(nasi_ctr_v1 nasi_cvr_v1)

while [[ ${start_date} != ${end} ]]
do
    for task in ${tasks[@]}
    do
        if ! test -e /data/lixiang/data/${task}/${start_date}/_SUCCESS
        then
            # echo "lx>>>${start_date}>>>${task}"
            ret=`hadoop fs -test -e /user/hive/warehouse/lixiang.db/${task}/${start_date}/_SUCCESS;echo $?`
            if [ ${ret} -eq 0 ]
            then
                hadoop fs -get /user/hive/warehouse/lixiang.db/${task}/${start_date} /data/lixiang/data/${task}
            fi
        fi
    done
    start_date=$(date -d "${start_date/\// } +1 days" +%Y%m%d)
done


