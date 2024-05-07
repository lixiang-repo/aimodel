#!/usr/bin/env python
# coding=utf-8
import tensorflow.compat.v1 as tf
import re, json
import time
from collections import OrderedDict
import os, oss2
import gzip
import pandas as pd
from odps import ODPS
import gzip
import numpy as np
try:
    from common.utils import *
except:
    from utils import *

logger = tf.compat.v1.logging
dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(f"{dirname}/schema.conf") as f:
    schema = [l.strip("\n") for l in f]
with open(f"{dirname}/slot.conf") as f:
    slots = [l.strip("\n") for l in f]

def mapper(x):
    lst = tf.split(x, [1] * 5)
    user_id, requestid, combination_un_id, is_click, features = [tf.squeeze(x) for x in lst]
    #return tf.strings.to_number(is_click, tf.float32), tf.strings.to_number(tf.strings.split(features, " "), tf.int64)
    features = tf.split(tf.strings.split(features, "\002"), [1] * len(schema))
    logger.info(f"features: {features}")
    _slots = tf.concat([v for k, v in zip(schema, features) if k in slots], axis=0)
    return tf.strings.to_number(is_click, tf.float32), _slots

def input_fn(filenames, model_dir, task_number=1, task_idx=0, shuffle=False, epochs=1, batch_size=1024):
    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if task_number > 1:
        dataset = dataset.shard(task_number, task_idx)

    # dataset = tf.data.TFRecordDataset(dataset)#, compression_type="GZIP")
    dataset = tf.data.TextLineDataset(dataset, compression_type='GZIP')
    #user_id, requestid, combination_un_id, is_click, features
    dataset = dataset.map(lambda x: tf.strings.split(x, "\t")).filter(lambda x: tf.math.equal(tf.shape(x)[0], 5))
    dataset = dataset.map(mapper).filter(lambda *x: tf.math.equal(tf.shape(x[1])[0], len(slots)))
    
    dataset = dataset.repeat(epochs).prefetch(batch_size * 100)
    # Randomizes input using a window of 256 elements (read into memory)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 20)
    # epochs from blending together.
    # elements = dataset.batch(batch_size).make_one_shot_iterator().get_next()
    elements = tf.compat.v1.data.make_one_shot_iterator(dataset.batch(batch_size)).get_next()
    # dataset = dataset.apply(tf.data.experimental.ignore_errors())
    features = dict(zip(["label", "features"], elements))
    return features


if __name__ == "__main__":
    # tf.disable_v2_behavior()
    t0 = time.time()
    import glob
    filenames = glob.glob("/data/share/data/deep_rank_v7/20240301/part*.gz")
    next_element = input_fn(filenames, "")
    with tf.compat.v1.Session() as sess:
        # sess.run(fetches=tf.global_variables_initializer())
        try:
            while True:
                # 通过session每次从数据集中取值
                serialized_examples = sess.run(fetches=next_element)
                print(serialized_examples)
                #logger.info(serialized_examples)
        except tf.errors.OutOfRangeError:
            logger.info("end!")

    logger.info(f"total cost: {(time.time() - t0) / 60:.2f} mins")

    # options = tf.io.TFRecordOptions(compression_type="GZIP")
    # tfrecord_path = "/tf/part-r-00000.gz"
    # for serialized_example in tf.compat.v1.io.tf_record_iterator(tfrecord_path, options=options):
    #     logger.info(serialized_example)
    #     break

