#!/usr/bin/env python
# coding=utf-8
import json

from dataset import input_fn, get_example_fmt
import tensorflow as tf
tf = tf.compat.v1


if __name__ == "__main__":
    tf.disable_v2_behavior()
    # from flags import FLAGS
    file_pattern = ["/Users/lixiang/Downloads/part-r-00000"]
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

    serialized_examples = tmp[1]
    for k, v in serialized_examples.items():
        if isinstance(v[0], bytes):
            serialized_examples[k] = v[0].decode("utf-8")
        else:
            serialized_examples[k] = v[0]

    with open("./body.json") as f:
        body = json.load(f)

    for k in body["userInfo"]["userMap"]:
        if k not in serialized_examples: continue
        body["userInfo"]["userMap"][k] = serialized_examples[k]

    for k in body["userInfo"]["contextMap"]:
        if k not in serialized_examples: continue
        body["userInfo"]["contextMap"][k] = serialized_examples[k]

    for k in body["map"]["recall"][0]["map"]:
        if k not in serialized_examples: continue
        body["map"]["recall"][0]["map"][k] = serialized_examples[k]

    with open("./body.json", "w") as wf:
        json.dump(body, wf, indent=4)

    print(body)
