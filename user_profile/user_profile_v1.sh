#!/usr/bin/env bash

start_date=`date -d "10 days ago" +%Y%m%d`
# start_date=20231201
end=`date -d "0 days ago" +%Y%m%d`
task=user_profile_v1

while [[ ${start_date} != ${end} ]]
do
    hadoop fs -rmr /user/hive/warehouse/lixiang.db/${task}/imp/*/${start_date} > /dev/null 2>&1
    hadoop fs -rmr /user/hive/warehouse/lixiang.db/${task}/click/*/${start_date} > /dev/null 2>&1
    
    hadoop fs -rmr /user/hive/warehouse/lixiang.db/${task}/imp/${start_date} > /dev/null 2>&1
    hadoop fs -rmr /user/hive/warehouse/lixiang.db/${task}/click/${start_date} > /dev/null 2>&1

    start_date=$(date -d "${start_date/\// } +1 days" +%Y%m%d)
    ret=`hadoop fs -test -e /user/hive/warehouse/lixiang.db/${task}/${start_date}/_SUCCESS;echo $?`
    if [ ! ${ret} -eq 0 ]
    then
        spark-submit --master yarn --deploy-mode cluster --queue root.jupyter \
             user_profile_v1.py --task ${task} --date ${start_date}
        #    --archives ./venv.zip#env \
        #    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=env/python-env/bin/python \
    fi
done


