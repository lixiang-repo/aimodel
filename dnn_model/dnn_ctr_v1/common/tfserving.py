#!/usr/bin/env python
# coding=utf-8
import grpc  
import json
import tensorflow as tf
import tensorflow_serving.apis.predict_pb2  
import tensorflow_serving.apis.prediction_service_pb2_grpc  
import time


# 连接到 TensorFlow Serving 服务  
channel = grpc.insecure_channel('localhost:9500')  
stub = tensorflow_serving.apis.prediction_service_pb2_grpc.PredictionServiceStub(channel)  

with open("/root/lixiang/test.json") as f:
    data = json.load(f)["inputs"]

def request(data):
    # 构建预测请求
    t0 = time.time()
    request = tensorflow_serving.apis.predict_pb2.PredictRequest()  
    request.model_spec.name = 'model2'  # 替换为你的模型名称  
    request.model_spec.signature_name = 'serving_default'  # 使用默认签名函数  
    for k in data:
        request.inputs[k].CopyFrom(tf.make_tensor_proto(data[k], tf.string))  # 将图片数据添加到输入张量中  
    t1 = time.time()

    t2 = time.time()
    response = stub.Predict(request)
    t3 = time.time()
    print(f"build request waste: {(t1 - t0) * 1000:.4f} ms, request waste: {(t3 - t2) * 1000:.4f} ms")
    return response

# 发送预测请求并接收结果  
for _ in range(100000):
    response = request(data)
    #print(response)

    #break
