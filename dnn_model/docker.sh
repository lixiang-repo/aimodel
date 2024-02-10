#!/usr/bin/env bash

task=$1

docker run -itd -v /data/lixiang:/data/lixiang -w /data/lixiang/aimodel/dnn_model/${task}/bin \
    --privileged=true --name ${task} --rm registry.cn-hangzhou.aliyuncs.com/lixiang666/tfra:python3.8.10 \
    bash -c "nohup sh -x run.sh > log.train 2>&1"
