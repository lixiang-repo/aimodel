#!/usr/bin/env python
# coding: utf-8
import argparse

import warnings
from dateutil.parser import parse
from pyspark.sql.types import *
import pyspark.sql.functions as F
import datetime

from dateutil.parser import parse
import numpy as np
import json, os

import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('--task', type=str, default='')
parser.add_argument('--date', type=str, default='')
args = parser.parse_args()
contexts = ['display', 'cid', 'bundle', 'pub', 'subcat', 'adid', 'advertiser_id', 'tagid', 'size', 'crid',
            'make', 'state', 'adtype', 'month', 'camera_pixels', 'osv', 'reward', 'appcat', 'os', 'chipset',
            'devicetype', 'city', 'model', 'hour']
boundary_map = {
    "c": [0, 1, 2, 3, 5, 10, 15, 20, 25, 40, 80, 150]
}
os.environ['SPARK_HOME'] = '/opt/cloudera/parcels/CDH/lib/spark3'
# os.environ['SPARK_CONF_DIR'] = '/etc/spark/conf'
# os.environ['HADOOP_CONF_DIR'] = '/etc/spark/conf/yarn-conf:/etc/hive/conf'
# # 
# sys.path.append(f"{os.environ['SPARK_HOME']}/python")
# sys.path.append(f"{os.environ['SPARK_HOME']}/python/lib/py4j-0.10.9.5-src.zip")
print(os.environ['SPARK_HOME'])

# PYSPARK_PYTHON = "/data02/py37/bin/python3.7"
# os.environ['PYSPARK_PYTHON'] = PYSPARK_PYTHON
os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars hdfs://nameservice1/user/root/libs/spark-tensorflow-connector_2.12-1.11.0.jar pyspark-shell'

from pyspark.sql import SparkSession
spark = (SparkSession
         .builder
         .appName("spark_%s" % args.task)
         .config("spark.kryoserializer.buffer.max", "1024m")
         .config("spark.kryoserializer.buffer", "64m")
         .config("spark.executor.memory", "6g")
         .config("spark.executor.cores", "2")
         .config("spark.jars","/tmp/udfas.jar")
         .config("spark.jars", "hdfs://nameservice1/user/root/libs/udfas.jar")
         .config("spark.jars","/opt/cloudera/parcels/CDH/lib/hive/lib/json-serde-1.3.8-jar-with-dependencies.jar")
         .config("spark.jars","hdfs://nameservice1/user/root/libs/spark-tensorflow-connector_2.12-1.11.0.jar")
         .config("spark.driver.extraClassPath", "hdfs://nameservice1/user/root/libs/udfas.jar")
         .config("spark.local.dir", "/var/tmp")
         .config("spark.driver.memory", "4g")
         .config("spark.debug.maxToStringFields", "100")
         .config("spark.driver.maxResultSize", "10000M")
         .config("spark.sql.execution.pyspark.arrow.enabled", "true")
         .config("spark.dynamicAllocation.enabled", "true")
         .config("spark.shuffle.service.enabled", "true")
         .config("spark.dynamicAllocation.minExecutors", "5")
         .config("spark.dynamicAllocation.maxExecutors", "100")
         .config("spark.dynamicAllocation.executorIdleTimeout", "39000s")
         .config("spark.executor.heartbeatInterval", "39000s")
         .config("spark.network.timeout", "40000s")
         .config("spark.default.parallelism", "600")
         .config("spark.sql.shuffle.partitions", "600")
         .config("spark.executor.initialExecutors", "10")
         .config("spark.sql.broadcastTimeout", "10000")
         .config("spark.sql.autoBroadcastJoinThreshold", "900")
         .config("spark.sql.adaptive.enabled", "true")
#          .config("'spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB")
         .config("spark.sql.adaptive.shuffle.targetPostShuffleRowCount", "20000000")
         .config("spark.sql.adaptive.join.enabled", "true")
         .config("spark.sql.adaptive.skewedJoin.enabled", "true")
         .config("spark.sql.adaptive.skewedPartitionMaxSplits", "5")
         .config("spark.sql.adaptive.skewedPartitionRowCountThreshold", "1000W")
         .config("spark.sql.adaptive.skewedPartitionSizeThreshold", "64M")
         .config("spark.sql.adaptive.skewedPartitionFactor", "10")
         .config("spark.shuffle.useOldFetchProtocol", "true")
         .config("spark.master","yarn")
         .enableHiveSupport()
         .config("spark.yarn.queue", "root.jupyter")
         .getOrCreate())

sc = spark.sparkContext

def parse_float(v):
    try:
        v = float(v)
    except:
        v = -1.0
    return float(v)

def fill_na(x):
    if x is None or x == "":
        x = "_"
    return x

def discretizing(v, boundary):
    idx = 0
    for i in range(len(boundary)):
        if v >= boundary[i]:
            idx = i + 1
    return "%d" % idx


def parse_profile(uid, value):
    value = json.loads(value)

    imp = value.get("imp", {}).get("cnt", 0)
    click = value.get("clk", {}).get("cnt", 0)

    features = {
        "uid": uid,
        "u_imp_cnt_d90": discretizing(imp, boundary_map["c"]),
        "u_clk_cnt_d90": discretizing(click, boundary_map["c"]),
        "u_ctr_d90": discretizing(click / (imp + 1e-20), np.linspace(0, 1, 15))
    }
    #advertiser_id, cid, crid, adid, adtype, size
    for evt in ["imp", "clk"]:
        for c in contexts:
            boundary = boundary_map["c"]
            fea = sorted([(k, v) for k, v in value.get(evt, {}).get(c, {}).items()], key=lambda x: -x[1])
            fea = ["%s_%s" % (k, discretizing(v, boundary)) for (k, v) in fea]

            features["u_%s_%s_cnt_d90" % (c, evt)] = fea[:50] + ["_"] * max(50 - len(fea), 0)
    #序列特征
    for evt in ["imp", "clk"]:
        for c in ["advertiser_id", "cid", "crid", "adid"]:
            seq = sorted([(k, v) for k, v in value.get(evt, {}).get(c, {}).items()], key=lambda x: -x[1])
            features["u_v_%s_%s_d90" % (c, evt)] = list(map(lambda x: x[0], seq[:50])) + ["_"] * max(50 - len(seq), 0)
        
    return features



def main(args):
    day = parse(args.date)

    table = spark.table('dsp.rm_show_click_v2').withColumnRenamed("isclick", "label")

    y, m, d = day.strftime("%Y"), day.strftime("%m"), day.strftime("%d")

    df = table.where("year='%s' and month='%s' and day='%s'" % (y, m, d)) \
        .where("label in (0, 1) and if(btype is null, '1', btype) != '2'").persist()


    cnt = df.count()
    print("cnt>>>", day, cnt)
    if cnt > 10000:
        df = df.withColumn("bid_floor", F.udf(parse_float, returnType=FloatType())("bid_floor")) \
            .withColumn("modelprice", F.udf(parse_float, returnType=FloatType())("modelprice")) \
            .withColumn("label", F.udf(parse_float, returnType=FloatType())("label"))
        dtypes = dict(df.dtypes)
        for k, dtype in dtypes.items():
            if dtype == "string":
                df = df.withColumn(k, F.udf(fill_na)(k))

        profile_dt = day - datetime.timedelta(days=2)
        user_profile_path = "/user/hive/warehouse/lixiang.db/user_profile_v2/%s" % profile_dt.strftime("%Y%m%d")
        user_profile = sc.textFile(user_profile_path).map(lambda x: x.strip("\n").split("\t\t"))#.toDF(["uid", "value"])#.mapValues(json.loads)

        user_df = spark.createDataFrame(user_profile.map(lambda x: parse_profile(*x)))
        data = df.join(user_df, df.deviceid == user_df.uid, "left")

        for k, dtype in user_df.dtypes:
            if k.startswith("u_"):
                if dtype == 'array<string>':
                    data = data.withColumn(k, F.udf(lambda x: ["_"] * 50 if x is None else x, returnType=ArrayType(StringType()))(k))
                elif dtype == 'string':
                    data = data.withColumn(k, F.udf(lambda x: "1" if x is None else x, returnType=StringType())(k))

        savedPath = "/user/hive/warehouse/lixiang.db/%s/%s" % ("nasi_ctr_data_v1", day.strftime("%Y%m%d"))
        data.repartition(50).write.mode("overwrite").format("tfrecords").option("recordType", "Example").save(savedPath)


if __name__ == "__main__":
    main(args)