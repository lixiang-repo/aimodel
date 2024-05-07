#!/usr/bin/env python
# coding=utf-8
import os, json
import tensorflow as tf
from tensorflow_serving.apis import prediction_log_pb2
import tensorflow_serving.apis.predict_pb2  
import tensorflow_serving.apis.prediction_service_pb2_grpc  

dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
basename = os.path.basename(dirname)

def parse_int(x):
    try:
        return int(x)
    except:
        return -1

if __name__ == "__main__":
    os.chdir(f"/data/share/model/{basename}/export_dir")
    files = sorted(map(parse_int, os.listdir(".")))
    print(f"files: {files}")

    with open(f"{dirname}/common/body.json") as f:
        data = json.load(f)["inputs"]

    request = tensorflow_serving.apis.predict_pb2.PredictRequest()  
    request.model_spec.name = 'model'  # 替换为你的模型名称  
    request.model_spec.signature_name = 'serving_default'  # 使用默认签名函数  
    for k in data:
        request.inputs[k].CopyFrom(tf.make_tensor_proto(data[k], tf.string))  # 将图片数据添加到输入张量中 
    
    example = prediction_log_pb2.PredictionLog(predict_log=prediction_log_pb2.PredictLog(request=request))

    warmup_path = f"/data/share/model/{basename}/export_dir/{files[-1]}/assets.extra"
    os.makedirs(warmup_path)
    with tf.io.TFRecordWriter(f"{warmup_path}/tf_serving_warmup_requests") as writer:
        for _ in range(300):
            writer.write(example.SerializeToString())
