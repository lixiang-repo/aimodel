#!/usr/bin/env python
# coding=utf-8
import json, os, re

from dataset import input_fn, get_example_fmt
import tensorflow as tf
tf = tf.compat.v1

dirname = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    tf.disable_v2_behavior()
    # from flags import FLAGS
    file_pattern = ["/data/lixiang/data/nasi_ctr_v3/20240215/part-r-00000"]
    next_element = input_fn(file_pattern, batch_size=1, shuffle=False)

    tmp = []
    with tf.Session() as sess:
        # sess.run(fetches=tf.global_variables_initializer())
        try:
            while True:
                # 通过session每次从数据集中取值
                serialized_examples = sess.run(fetches=next_element)
                tmp.append(serialized_examples)
                # print(serialized_examples)
                if len(tmp) >= 1000:
                    break
        except tf.errors.OutOfRangeError:
            print("end!")

    serialized_examples = tmp[0]
    with open("./body.json") as f:
        body = json.load(f)

    body["userInfo"]["userMap"] = {}
    body["userInfo"]["contextMap"] = {}
    body["map"]["recall"][0]["map"] = {}
    with open("%s/../schema.conf" % dirname) as f:
        for line in f:
            if line.startswith("#") or line.startswith("label"):
                continue

            k = re.split(" +", line)[0]
            dtype = re.split(" +", line)[1]
            v = serialized_examples[k]
            print("lx>>>", k, dtype,  ">>>", v)
            if "ARRAY<STRING>" in line:
                v = ",".join(map(lambda x: x.decode(), v.tolist()[0]))
            elif "ARRAY<FLOAT>" in line:
                v = ",".join(map(lambda x: str(float(x)), v.tolist()[0]))
            elif "ARRAY<LONG>" in line:
                v = ",".join(map(lambda x: str(int(x)), v.tolist()[0]))
            elif "STRING" in line:
                v = v[0].decode()
            elif "FLOAT" in line:
                v = str(float(v[0]))
            elif "LONG" in line:
                v = str(int(v[0]))

            if "@user" in line:
                body["userInfo"]["userMap"][k] = v
            if "@ctx" in line:
                body["userInfo"]["contextMap"][k] = v
            if "@item" in line:
                body["map"]["recall"][0]["map"][k] = v

    with open("./body.json", "w") as wf:
        json.dump(body, wf, indent=4)

    print(body)
