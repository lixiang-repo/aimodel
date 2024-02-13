#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

import tensorflow_recommenders_addons as tfra
import tensorflow_recommenders_addons.dynamic_embedding as de
from model import DnnModel, get_table_name, boundary_map
from common.metrics import evaluate
from common.utils import RestoreTfraVariableHook
from collections import OrderedDict
import os, re
dirname = os.path.dirname(os.path.abspath(__file__))
logger = tf.compat.v1.logging


def build_bucket_custom(features, feature_name, boundary):
    """分桶"""
    raw_data = tf.cast(tf.zeros_like(features[feature_name]), tf.float32)
    for index in range(len(boundary)):
        temp_data = tf.cast(tf.ones_like(features[feature_name]), tf.float32) * (index + 1)
        value_data = tf.cast(tf.ones_like(features[feature_name]), tf.float32) * boundary[index]
        raw_data = tf.where(tf.greater_equal(tf.cast(features[feature_name], tf.float32), value_data), temp_data, raw_data)
    return tf.cast(raw_data, dtype=tf.int64)


def mask_tensor(tensor, padding="_"):
    # mask = tf.where(tensor == padding, tf.zeros_like(tensor, dtype=tf.float32), tf.ones_like(tensor, dtype=tf.float32))
    padding = tf.constant(padding, dtype=tf.string)
    mask = tf.where(tf.equal(tensor, padding), tf.zeros_like(tensor, dtype=tf.float32), tf.ones_like(tensor, dtype=tf.float32))
    return mask


def model_fn(features, labels, mode, params):
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    embedding_size = 9
    ######################print features################################
    logger.warning("start>>>%s>>>%s" % (params["type"], mode))
    logger.info("strategy>>>%s" % tf.compat.v1.distribute.get_strategy())
    ######################devide################################
    if is_training:
        devices_info = ["/job:ps/replica:0/task:{}/CPU:0".format(i) for i in range(params["ps_num"])]
        initializer = tf.compat.v1.random_normal_initializer(0, 1)
    else:
        devices_info = ["/job:localhost/replica:0/task:{}/CPU:0".format(0) for i in range(params["ps_num"])]
        initializer = tf.compat.v1.zeros_initializer()
    if len(devices_info) == 0: devices_info = None
    logger.info("------ dynamic_embedding devices_info is {}-------".format(devices_info))
    ######################slot reshape################################
    # (None,), (None, 50)
    mask_dict = {}
    text_map = {}
    for k in features:
        logger.debug("features>>>%s>>>%s>>>%s" % (k, features[k].shape, features[k].dtype))
        if k.startswith("label"):
            continue
        text_map[k] = features[k]
        ############bucketing#########
        if k in boundary_map:
            text_map[k] = build_bucket_custom(text_map, k, boundary_map[k])
        if text_map[k].dtype != tf.string:
            text_map[k] = tf.as_string(text_map[k])
        if len(text_map[k].shape) == 1:
            text_map[k] = tf.reshape(text_map[k], [-1, 1])
        else:
            logger.debug("mask>>>%s" % k)
            mask_dict[k] = mask_tensor(text_map[k])
    ######################slot################################
    hash_collision_inps = []
    slot_list = []
    hash_map = {}
    with open("%s/../slot.conf" % dirname) as f:
        for l in f:
            if l.startswith("#") or l.startswith("label"):
                continue
            l = re.split(" +", l.strip("\n"))[0]
            if l in slot_list: continue
            slot_list.append(l)
            inputs = [get_table_name(k) for k in l.split("-")] + [text_map[k] for k in l.split("-")]
            inps = tf.strings.join(inputs, "\001\001")
            # hash_map[l] = tf.strings.to_hash_bucket_fast(tf.strings.join(inputs, "\001\001"), 2 ** 63 - 1)
            hash_map[l] = tf.strings.to_hash_bucket_strong(inps, 2 ** 63 - 1, [1, 2])

            #hash_collision
            hash_collision_inps.append(tf.reshape(inps, [-1, 1]))

    logger.info("slot_list>>>%s" % slot_list)
    ######################dnn################################
    assert len(slot_list) == len(set(slot_list))
    feature_list = [hash_map[k] for k in slot_list]
    ids = tf.concat(feature_list, axis=1)  # [None, 1] concat
    logger.info("ids_shape>>>%s" % ids.shape)
    # with tf.name_scope("embedding"):
    groups = []
    embeddings = tfra.dynamic_embedding.get_variable(
        name="embeddings",
        dim=embedding_size,
        devices=devices_info,
        trainable=params["type"] == "update" and is_training,
        initializer=initializer)
    policy = tfra.dynamic_embedding.TimestampRestrictPolicy(embeddings)
    update_tstp_op = policy.apply_update(ids)
    restrict_op = policy.apply_restriction(int(1e8))
    groups.append(update_tstp_op)
    if params["restrict"]: groups.append(restrict_op)
    ######################lookup################################
    emb_shape = tf.concat([tf.shape(ids), [embedding_size]], axis=0)
    id_val, id_idx = tf.unique(tf.reshape(ids, (-1,)))
    unique_embs, trainable_wrapper = de.embedding_lookup(embeddings, id_val, return_trainable=True, name="lookup")
    # tf.compat.v1.add_to_collections(de.GraphKeys.DYNAMIC_EMBEDDING_VARIABLES, embeddings)
    flat_emb = tf.gather(unique_embs, id_idx)
    emb_lookuped = tf.reshape(flat_emb, emb_shape)
    ######################lookup end################################
    num_or_size_splits = [hash_map[k].shape[1] for k in slot_list]
    emb_splits = tf.split(emb_lookuped, num_or_size_splits, axis=1)
    assert len(emb_splits) == len(slot_list)
    emb_map = dict(zip(slot_list, emb_splits))
    slot = params["slot"]
    for k in slot_list:
        logger.debug("emb_shape>>>%s>>>%s" % (k, emb_map[k].shape))
        if k == slot:
            logger.debug("miss_slot:%s" % k)
            emb_map[slot] = tf.zeros_like(emb_map[slot], dtype=tf.float32)
        if k in mask_dict:
            logger.debug("mask_slot:%s" % k)
            emb_map[k] = emb_map[k] * tf.expand_dims(mask_dict[k], 2)
    ######################predictions################################
    with tf.name_scope("dnn"):
        model = DnnModel(emb_map, features, slot_list, is_training)
    if params["mode"] in ["hash", "emb"]:
        model.predictions = {
            "str": tf.concat(hash_collision_inps, axis=0),
            "hash": tf.concat(feature_list, axis=0),
        }
        if params["mode"] == "emb":
            model.predictions["emb"] = tf.concat([tf.reduce_sum(emb_map[k], axis=1) for k in slot_list], axis=0)
    # predictions = {"id": tf.reshape(ids, (-1,)), "out": flat_emb}
    ######################metrics################################
    global_step = tf.compat.v1.train.get_or_create_global_step()
    loggings = OrderedDict({
        "step": global_step,
        "%s_emb_size" % params["type"]: embeddings.size(),
        "loss": model.loss
    })
    with tf.name_scope('metrics'):
        eval_metric_ops = OrderedDict()
        for i, (label, prob) in enumerate(zip(model.labels, model.probs), start=1):
            evaluate(label, prob, "task%s" % i, eval_metric_ops)
        for k in eval_metric_ops:
            loggings[k] = eval_metric_ops[k][0]
            groups.append(eval_metric_ops[k][1])
        if params["slot"]:
            eval_metric_ops['slot_%s' % params["slot"]] = eval_metric_ops[k]

    ######################train################################
    _hooks = None
    if params["warm_path"]:
        logger.info("%s warm start>>>%s" % (mode, params["warm_path"]))
        # restore_hook = de.WarmStartHook(params["warm_path"], [".*emb.*"])
        restore_hook = RestoreTfraVariableHook(params["warm_path"], [embeddings])
        _hooks = [restore_hook]
    if mode == tf.estimator.ModeKeys.TRAIN:
        ######################optimizer################################
        #sparse_opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1)
        sparse_opt = tf.compat.v1.train.AdamOptimizer()
        sparse_opt = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(sparse_opt)
        #dense_opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.02 * params["lr"])
        dense_opt = tf.compat.v1.train.AdagradOptimizer(learning_rate=0.001 * params["lr"])

        trainable_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        dense_vars = [var for var in trainable_variables if not var.name.startswith("lookup")]
        sparse_vars = [var for var in trainable_variables if var.name.startswith("lookup")]
        global_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)

        logger.debug("trainable_variables>>>%s" % trainable_variables)
        logger.debug("dense_vars>>>%s" % dense_vars)
        logger.debug("sparse_vars>>>%s" % sparse_vars)
        logger.debug("global_vars>>>%s" % global_variables)
        logger.debug("#" * 100)

        dense_op = dense_opt.minimize(model.loss, global_step=global_step, var_list=dense_vars)
        if params["type"] == "join":
            train_op = tf.group(dense_op, *groups)
        elif params["type"] == "update":
            sparse_op = sparse_opt.minimize(model.loss, global_step=global_step, var_list=sparse_vars)
            train_op = tf.group(dense_op, sparse_op, *groups)
        else:
            raise RuntimeError("Unsupport type", params["type"])

        log_hook = tf.compat.v1.estimator.LoggingTensorHook(loggings, every_n_iter=10)
        ######################WarmStartHook################################
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=model.predictions, loss=model.loss,
            train_op=train_op,
            training_hooks=[log_hook],
            training_chief_hooks=_hooks
        )
    ######################infer################################
    elif mode in [tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL]:
        export_outputs = {
            "pred": tf.compat.v1.estimator.export.PredictOutput(model.outputs)
        }
        return tf.estimator.EstimatorSpec(
                mode, model.predictions, model.loss, export_outputs=export_outputs,
                prediction_hooks=_hooks, evaluation_hooks=_hooks,
                eval_metric_ops=eval_metric_ops
        )

