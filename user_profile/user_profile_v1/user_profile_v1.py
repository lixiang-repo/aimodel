#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import pandas as pd
import json
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
import operator
from dateutil.parser import parse

from tqdm import tqdm
import os, sys
import datetime
import argparse

import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('--task', type=str, default='')
parser.add_argument('--date', type=str, default='')
args = parser.parse_args()

os.environ['SPARK_HOME'] = '/opt/cloudera/parcels/CDH/lib/spark3'
# os.environ['SPARK_CONF_DIR'] = '/etc/spark/conf'
# os.environ['HADOOP_CONF_DIR'] = '/etc/spark/conf/yarn-conf:/etc/hive/conf'
# # 
# sys.path.append(f"{os.environ['SPARK_HOME']}/python")
# sys.path.append(f"{os.environ['SPARK_HOME']}/python/lib/py4j-0.10.9.5-src.zip")
print(os.environ['SPARK_HOME'])

# PYSPARK_PYTHON = "/data02/py37/bin/python3.7"
# os.environ['PYSPARK_PYTHON'] = PYSPARK_PYTHON
os.environ[
    'PYSPARK_SUBMIT_ARGS'] = '--jars hdfs://nameservice1/user/root/libs/spark-tensorflow-connector_2.12-1.11.0.jar pyspark-shell'

from pyspark.sql import SparkSession

spark = (SparkSession
         .builder
         .appName("lixiang_%s" % args.task)
         .config("spark.kryoserializer.buffer.max", "1024m")
         .config("spark.kryoserializer.buffer", "64m")
         .config("spark.driver.memory", "15g")
         .config("spark.executor.memory", "15g")
         .config("spark.executor.cores", "3")
         .config("spark.executor.memoryOverhead", "10g")
         .config("spark.jars", "/tmp/udfas.jar")
         .config("spark.jars", "hdfs://nameservice1/user/root/libs/udfas.jar")
         .config("spark.jars", "/opt/cloudera/parcels/CDH/lib/hive/lib/json-serde-1.3.8-jar-with-dependencies.jar")
         .config("spark.jars", "hdfs://nameservice1/user/root/libs/spark-tensorflow-connector_2.12-1.11.0.jar")
         .config("spark.driver.extraClassPath", "hdfs://nameservice1/user/root/libs/udfas.jar")
         .config("spark.local.dir", "/var/tmp")
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
         # .config("'spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB")
         .config("spark.sql.adaptive.shuffle.targetPostShuffleRowCount", "20000000")
         .config("spark.sql.adaptive.join.enabled", "true")
         .config("spark.sql.adaptive.skewedJoin.enabled", "true")
         .config("spark.sql.adaptive.skewedPartitionMaxSplits", "5")
         .config("spark.sql.adaptive.skewedPartitionRowCountThreshold", "1000W")
         .config("spark.sql.adaptive.skewedPartitionSizeThreshold", "64M")
         .config("spark.sql.adaptive.skewedPartitionFactor", "10")
         .config("spark.shuffle.useOldFetchProtocol", "true")
         .config("spark.master", "yarn")
         .enableHiveSupport()
         .config("spark.yarn.queue", "root.jupyter")
         .getOrCreate())
sc = spark.sparkContext
dtype_map = dict(spark.table('dsp.rm_show_click_v2').dtypes)


def fill_double(v):
    try:
        v = float(v)
    except:
        v = -1.0
    return float(v)


def fill_na(row):
    row = row.asDict()
    ans = {}
    for k, v in row.items():
        dtype = dtype_map.get(k)
        if v is None or str(v).lower() in ("", "null"):
            if dtype == 'double':
                v = -1.0
            elif dtype == 'tinyint':
                v = -1
            elif dtype == 'string':
                v = '_'
        ans[k] = v
    return ans


def map_union(lst):
    ans = {}
    for x in lst:
        ans.update(x)
    return ans


contexts = ['display', 'cid', 'bundle', 'pub', 'subcat', 'adid', 'advertiser_id', 'tagid', 'app_id', 'size', 'crid',
            'make', 'state', 'adtype', 'month', 'camera_pixels', 'osv', 'reward', 'appcat', 'os', 'chipset',
            'devicetype', 'city', 'model', 'hour']
yd = parse(args.date)
d89 = yd - datetime.timedelta(days=89)
dt_condition = " or ".join([
    "(year='%s' and month='%s' and day='%s')" % (dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d"))
    for dt in pd.date_range(d89, yd)
])

table = spark.table('dsp.rm_show_click_v2') \
    .where(dt_condition) \
    .where("isclick in (0, 1) and if(btype is null, '1', btype) != '2'")
table = spark.createDataFrame(table.rdd.map(fill_na)).persist()

base_path = "hdfs://nameservice1/user/hive/warehouse/lixiang.db/%s" % args.task
codec = "org.apache.hadoop.io.compress.GzipCodec"
for evt  in ["imp", "click"]:
    if evt == "imp":
        df = table
    elif evt == "click":
        df = table.where("isclick = 1")

    data = df.groupBy("deviceid").count().rdd.map(lambda x: (x["deviceid"], {"cnt": x["count"]}))
    paths = []
    for k in contexts:
        rdd = df.groupBy("deviceid", k).count().rdd \
            .map(lambda x: (x["deviceid"], [{x[k]: x["count"]}])) \
            .reduceByKey(operator.add).mapValues(map_union) \
            .mapValues(lambda x: {k: x})

        savedPath = "%s/%s/%s/%s" % (base_path, evt, k, yd.strftime("%Y%m%d"))
        paths.append(savedPath)
        rdd.mapValues(json.dumps).repartition(50).map(lambda x: "%s\t\t%s" % (x[0], x[1])) \
            .saveAsTextFile(savedPath, codec)

    rdd = sc.textFile(",".join(paths)).map(lambda x: x.strip("\n").split("\t\t")).mapValues(json.loads)
    data = data.union(rdd)

    data = data.mapValues(lambda x: [x]).reduceByKey(operator.add) \
        .mapValues(lambda x: json.dumps(map_union(x))).repartition(50) \
        .map(lambda x: "%s\t\t%s" % (x[0], x[1]))

    savedPath = "%s/%s/%s" % (base_path, evt, yd.strftime("%Y%m%d"))
    data.saveAsTextFile(savedPath, codec)


#merge imp/clk
path1 = "%s/imp/%s" % (base_path, yd.strftime("%Y%m%d"))
imp = sc.textFile(path1).map(lambda x: x.strip("\n").split("\t\t")).mapValues(lambda x: [{"imp": json.loads(x)}])

path2 = "%s/click/%s" % (base_path, yd.strftime("%Y%m%d"))
click = sc.textFile(path2).map(lambda x: x.strip("\n").split("\t\t")).mapValues(lambda x: [{"clk": json.loads(x)}])

data = imp.union(click).reduceByKey(operator.add) \
    .mapValues(lambda x: json.dumps(map_union(x))) \
    .repartition(50) \
    .map(lambda x: "%s\t\t%s" % (x[0], x[1]))

savedPath = "%s/%s" % (base_path, yd.strftime("%Y%m%d"))
data.saveAsTextFile(savedPath, codec)



    

# table = spark.table('dsp.rm_show_click_v2').where("isclick in (0, 1) and if(btype is null, '1', btype) != '2'").cache()
# for day in tqdm(pd.date_range("20231201", "20240101")):
# #     print(day.strftime("%Y"), day.strftime("%m"), day.strftime("%d"))
#     y, m, d = day.strftime("%Y"), day.strftime("%m"), day.strftime("%d")

#     df = table.where("year='%s' and month='%s' and day='%s'" % (y, m, d))
#     df = df.withColumn("bid_floor", F.udf(fill_double, returnType=DoubleType())("bid_floor"))

#     cnt = df.count()
#     print("cnt>>>", day, cnt)
#     if cnt > 10000:
#         dtype_map = dict(df.dtypes)
#         df = spark.createDataFrame(df.rdd.map(fill_na))
#         savedPath = "hdfs://nameservice1/user/hive/warehouse/lixiang.db/rm_show_click_v2/%s/%s/%s" % (y, m, d)
#         df.repartition(5).write.mode("overwrite").format("tfrecords").option("recordType", "Example").option("compression","gzip").save(savedPath)
