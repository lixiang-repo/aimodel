#!/usr/bin/env python
# coding=utf-8
import json

from dataset import input_fn, get_example_fmt
import tensorflow as tf
tf = tf.compat.v1


if __name__ == "__main__":
    tf.disable_v2_behavior()
    # from flags import FLAGS
    file_pattern = ["/data/lixiang/rm_show_click_v2/2023/12/01/part-r-00000"]
    next_element = input_fn(file_pattern, batch_size=1, shuffle=False)
    example_fmt = get_example_fmt()

    with open("./body.json") as f:
        body = json.load(f)

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
                # sess.run(fetches=train, feed_dict={x: image, y_: label})
                # if i % 100 == 0:
                #     train_accuracy = sess.run(fetches=accuracy, feed_dict={x: image, y_: label})
                #     print(i, "accuracy=", train_accuracy)
                # i = i + 1
        except tf.errors.OutOfRangeError:
            print("end!")
    serialized_examples = tmp[0]
    for k, v in serialized_examples.items():
        if isinstance(v[0], bytes):
            serialized_examples[k] = v[0].decode("utf-8")
        else:
            serialized_examples[k] = v[0]
    for k in body["userInfo"]["contextMap"].keys():
        body["userInfo"]["contextMap"][k] = serialized_examples[k]
    for k in body["map"]["recall"][0]["map"]:
        body["map"]["recall"][0]["map"][k] = serialized_examples[k]

    with open("./reqbody.json", "w") as wf:
        json.dump(body, wf, indent=4)

    print(body)
