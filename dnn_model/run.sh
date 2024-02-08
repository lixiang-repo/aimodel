#!/usr/bin/env bash

task=$1

docker run -itd -v /data/lixiang:/data/lixiang --privileged=true --name ${task} --rm registry.cn-hangzhou.aliyuncs.com/lixiang666/tfra:python3.8.10 \
    bash -c "cd /data/lixiang/aimodel/dnn_model/${task}/bin/ && nohup sh -x run.sh > log.train 2>&1"
