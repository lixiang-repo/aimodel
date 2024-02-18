#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import datetime
from collections import Counter
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

def exec_cmd(cmd):
    import subprocess

    # 定义要运行的Shell命令
    try:
        # 通过subprocess.run()函数调用Shell命令并获取输出结果
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            # 打印命令输出内容
            return result.stdout.strip("\n")
        else:
            print("执行Shell命令失败", "cmd>>>%s" % cmd)
    except Exception as e:
        print("发生错误：", str(e), "cmd>>>%s" % cmd)

def map_union(lst):
    ans = {}
    for x in lst:
        ans.update(x)
    return ans

def weight_decay(x):
    for evt in list(x.keys()):
        if evt == "time":
            continue
        x[evt]["cnt"] = x[evt].get("cnt", 0) * 0.98
        for k in x[evt].keys():
            if k != "cnt":
                x[evt][k] = (pd.Series(x[evt][k]) * 0.98).to_dict()
    return x

def map_reduce(lst):
    """
    {
        "time": 'unixtime'
        "imp": {
            "cnt": 10,
            "hour": {1: 2, 23: 5},
            "adid": {"ad1": 5, "ad2": 10}
        },
        "clk": {
            "cnt": 10,
            "hour": {1: 2, 23: 5},
            "adid": {"ad1": 5, "ad2": 10}
        }
    }
    """
    def reduce(x, y):
        ans = {"time": max(x.get("time", -1), y.get("time", -1))}
        for evt in list(set(list(x.keys()) + list(y.keys()))):
            if evt == "time":
                continue
            if evt in x and evt in y:
                ans[evt] = {"cnt": x[evt].get("cnt", 0) + y[evt].get("cnt", 0)}
                keys = list(set(list(x[evt].keys()) + list(y[evt].keys())))
                for k in keys:
                    if k != "cnt":
                        if k in x[evt] and k in y[evt]:
                            ans[evt][k] = dict(Counter(x[evt][k]) + Counter(y[evt][k]))
                        elif k in x[evt]:
                            ans[evt][k] = x[evt][k]
                        elif k in y[evt]:
                            ans[evt][k] = y[evt][k]
            elif evt in x:
                ans[evt] = x[evt]
            elif evt in y:
                ans[evt] = y[evt]
        return ans
    ans = {}
    for x in lst:
        ans = reduce(ans, x)
    return ans


def set_spark(task, flags):

    from pyspark.sql import SparkSession
    spark = (SparkSession
             .builder
             .appName("spark_jupyter_%s" % task)
             .config("spark.kryoserializer.buffer.max", "1024m")
             .config("spark.kryoserializer.buffer", "64m")
             .config("spark.executor.memory", "12g")
             .config("spark.executor.cores", "3")
             .config("spark.jars","/tmp/udfas.jar")
             .config("spark.jars", "hdfs://nameservice1/user/root/libs/udfas.jar")
             .config("spark.jars","/opt/cloudera/parcels/CDH/lib/hive/lib/json-serde-1.3.8-jar-with-dependencies.jar")
             .config("spark.jars","hdfs://nameservice1/user/root/libs/spark-tensorflow-connector_2.12-1.11.0.jar")
             .config("spark.driver.extraClassPath", "hdfs://nameservice1/user/root/libs/udfas.jar")
             .config("spark.local.dir", "/var/tmp")
             .config("spark.driver.memory", "4g")
    #          .config("spark.debug.maxToStringFields", "100")
             .config("spark.driver.maxResultSize", "10000M")
             .config("spark.shuffle.service.enabled", "true")
             .config("spark.sql.execution.pyspark.arrow.enabled", "true")
    #          .config("spark.dynamicAllocation.enabled", "true")
    #          .config("spark.dynamicAllocation.minExecutors", "5")
    #          .config("spark.dynamicAllocation.maxExecutors", "100")
    #          .config("spark.dynamicAllocation.executorIdleTimeout", "39000s")
             .config("spark.dynamicAllocation.enabled", "false")  # 禁用动态资源分配
             .config("spark.executor.instances", "30")  # 设置执行器的数量
             .config("spark.executor.heartbeatInterval", "39000s")
             .config("spark.network.timeout", "40000s")
             .config("spark.default.parallelism", "600")
             .config("spark.sql.shuffle.partitions", "600")
             .config("spark.executor.initialExecutors", "10")
             .config("spark.sql.broadcastTimeout", "10000")
             .config("spark.sql.autoBroadcastJoinThreshold", "900")
             .config("spark.sql.adaptive.enabled", "true")
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
    if not flags[0]:
        sc.addPyFile("utils.py")
        flags[0] = True
    return spark
