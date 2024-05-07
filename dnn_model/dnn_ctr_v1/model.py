#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf


class DnnModel:
    def __init__(self, features, emb_lookuped, is_training=True):
        super(DnnModel, self).__init__()
        self.is_training = is_training
        ######################_label################################
        batch_size = tf.shape(features["features"])[0]
        self.labels = []
        for name in ["label"]:
            _label = tf.cast(features[name], tf.float32) if name in features else tf.zeros((batch_size, ))
            self.labels.append(_label)
        # emb_map = {k: tf.keras.layers.Dropout(0.2)(emb_map[k]) for k in emb_map}
        ######################pop features不用特征################################
        # (None, 50, 9), (None, 1, 9)
        ######################forward################################
        outs = [self.build_layers(emb_lookuped, [256, 128, 64, 32, 1], "task1")]
        self.logits = []
        for i, logit in enumerate(outs, start=1):
            logit = tf.clip_by_value(tf.reshape(logit, (-1,)), -15, 15)
            self.logits.append(logit)
        ######################################################
        self.probs = [tf.sigmoid(logit) for logit in self.logits] #1/(w/p-w+1)
        self.outputs = {
            "output%s" % i: prob for i, prob in enumerate(self.probs, start=1)
        }
        self.loss = 0
        for i, (label, logit) in enumerate(zip(self.labels, self.logits), start=1):
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
            self.loss += loss
            tf.summary.scalar("loss%s" % i, loss)
        
        self.predictions = {
            "id": features["user_id"] if "user_id" in features else tf.as_string(tf.zeros((batch_size, ), tf.int16)),
            "out": tf.concat(tf.concat([tf.reshape(x, [-1, 1]) for x in self.labels + self.probs], axis=1), axis=1)
        }

    def build_layers(self, inp, units, prefix, activation=None):
        act = "relu"
        layers = []
        for i, unit in enumerate(units):
            name = 'dnn_hidden_%s_%d' % (prefix, i)
            if i == len(units) - 1:
                act = activation
            layer = tf.keras.layers.Dense(
                units=unit, activation=act,
                kernel_initializer=tf.compat.v1.glorot_uniform_initializer(),
                bias_initializer=tf.compat.v1.glorot_uniform_initializer(),
                name=name
            )
            layers.append(layer)
            #if i < max(len(units) - 2, 0):
            #    dropout = tf.keras.layers.Dropout(0.2)
            #    layers.append(dropout)
        return tf.keras.Sequential(layers)(inp)
