#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
from common.ple_networks import PleNetworks

def get_table_name(k):
    return k

boundary_map = {}
boundary_map['bid_floor'] = [-99, -0.001, 0.03, 0.1, 0.12, 0.15, 0.17, 0.2, 0.26, 0.3, 0.35, 37.37]

class DnnModel:
    def __init__(self, emb_map, features, slot_list, is_training=True):
        super(DnnModel, self).__init__()
        self.is_training = is_training
        ######################label################################
        label1 = features["label"]
        label1 = tf.cast(tf.reshape(label1, (-1,)), dtype=tf.float32)

        self.labels = [label1]
        # emb_map = {k: tf.keras.layers.Dropout(0.2)(emb_map[k]) for k in emb_map}
        ######################pop features不用特征################################

        #ui_etype = emb_map.pop("ui_etype")
        #slot_list.remove("ui_etype")
        input_emb = tf.concat([tf.multiply(tf.Variable(1.0, name="GateW_%s" % k), emb_map[k]) for k in slot_list], axis=1)
        ######################forward################################
        outs = [self.build_layers(input_emb, [512, 512, 512, 512, 256, 256], "hid")]
        self.logits = []
        for i, out in enumerate(outs, start=1):
            logit = self.build_layers(out, [256, 128, 1], "task%s" % i)
            logit = tf.clip_by_value(tf.reshape(logit, (-1,)), -15, 15)
            self.logits.append(logit)

        ######################################################
        self.probs = [tf.sigmoid(logit) for logit in self.logits]
        self.outputs = self.probs[0]#tf.concat([self.probs[0], self.probs[1]], axis=-1)
        self.loss = 0
        for i, (label, logit) in enumerate(zip(self.labels, self.logits), start=1):
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
            self.loss += loss
            tf.summary.scalar("loss%s" % i, loss)
        self.predictions = {
            "id": features["deviceid"],
            "out": tf.concat(tf.concat([tf.reshape(x, [-1, 1]) for x in self.labels + self.probs], axis=1), axis=1)
        }

    def build_layers(self, inp, units, prefix, activation=None):
        act = "relu"
        layers = []
        for i, unit in enumerate(units):
            name = 'dnn_hidden_%s_%d' % (prefix, i)
            if i == len(units) - 1:
                act = activation
            #     dropout = None
            # else:
            #     dropout = tf.keras.layers.Dropout(0.5)
            layer = tf.keras.layers.Dense(
                units=unit, activation=act,
                kernel_initializer=tf.compat.v1.glorot_uniform_initializer(),
                bias_initializer=tf.compat.v1.glorot_uniform_initializer(),
                name=name
            )
            layers.append(layer)
            # if dropout is not None:
            #     layers.append(dropout)
        return tf.keras.Sequential(layers)(inp)
