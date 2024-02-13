#!/usr/bin/env python
# coding=utf-8
import sys

import grpc
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
import tensorflow as tf

tfrecord_path = "/data/lixiang/rm_show_click_v2/2023/12/01/part-r-00000"


channel = grpc.insecure_channel("localhost:8500")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
examples = []
options = tf.io.TFRecordOptions()
for serialized_example in tf.compat.v1.io.tf_record_iterator(tfrecord_path, options=options):
    examples.append(serialized_example)
    if len(examples) >= 1000:
        break


def do_request_v2(proto, stub, run_mode='pred'):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "model"
    request.model_spec.signature_name = run_mode
    request.inputs['input'].CopyFrom(tf.compat.v1.make_tensor_proto(proto))
    request.inputs['record_type'].CopyFrom(tf.compat.v1.make_tensor_proto("Example"))
    rsp = stub.Predict(request)
    return rsp


res = do_request_v2(examples[:100], stub)
print(res)










