#!/usr/bin/env bash

start_date=`date -d "10 days ago" +%Y%m%d`
# start_date=20231201
end=`date -d "1 days ago" +%Y%m%d`
task=nasi_ctr_data_v1

export HADOOP_CONF_DIR='/etc/spark/conf/yarn-conf:/etc/hive/conf'
while [[ ${start_date} != ${end} ]]
do
    ret=`hadoop fs -test -e /user/hive/warehouse/lixiang.db/${task}/${start_date}/_SUCCESS;echo $?`
    if [ ! ${ret} -eq 0 ]
    then
        spark-submit --master yarn --deploy-mode cluster --queue root.jupyter \
             nasi_ctr_data_v1.py --task ${task} --date ${start_date}
        #    --archives ./venv.zip#env \
        #    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=env/python-env/bin/python \
    fi
    start_date=$(date -d "${start_date/\// } +1 days" +%Y%m%d)

    exit 0

done


