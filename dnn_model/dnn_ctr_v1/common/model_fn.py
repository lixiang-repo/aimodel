#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

import tensorflow_recommenders_addons as tfra
import tensorflow_recommenders_addons.dynamic_embedding as de
from model import DnnModel
from common.metrics import evaluate
from common.utils import get_exists_schema
from collections import OrderedDict
import os, re
dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = tf.compat.v1.logging


def mask_tensor(tensor, padding="_"):
    # mask = tf.where(tensor == padding, tf.zeros_like(tensor, dtype=tf.float32), tf.ones_like(tensor, dtype=tf.float32))
    padding = tf.constant(padding, dtype=tensor.dtype)
    mask = tf.where(tf.equal(tensor, padding), tf.zeros_like(tensor, dtype=tf.float32), tf.ones_like(tensor, dtype=tf.float32))
    return mask


def model_fn(features, labels, mode, params):
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    embedding_size = 9
    ######################print features################################
    logger.info(f"------ mode: {mode} strategy is {tf.compat.v1.distribute.get_strategy()} -------")
    logger.info(f"------ features: {features} -------")
    ######################devide################################
    if is_training:
        devices_info = ["/job:ps/replica:0/task:{}/CPU:0".format(i) for i in range(params["ps_num"])]
        initializer = tf.compat.v1.random_normal_initializer(-1, 1)
    else:
        devices_info = ["/job:localhost/replica:0/task:{}/CPU:0".format(0) for i in range(params["ps_num"])]
        initializer = tf.compat.v1.zeros_initializer()
    if len(devices_info) == 0: devices_info = None
    logger.info("------ dynamic_embedding devices_info is {} -------".format(devices_info))
    ##################################################################
    feat_val, id_idx = tf.unique(tf.reshape(features["features"], (-1, )))
    id_val = tf.strings.to_hash_bucket_strong(feat_val, 2 ** 63 - 1, [1, 2])
    ######################dnn################################
    # with tf.name_scope("embedding"):
    groups = []
    embeddings = tfra.dynamic_embedding.get_variable(
        name="embeddings",
        dim=embedding_size,
        devices=devices_info,
        trainable=is_training,
        initializer=initializer)
    policy = tfra.dynamic_embedding.TimestampRestrictPolicy(embeddings)
    update_tstp_op = policy.apply_update(id_val)
    restrict_op = policy.apply_restriction(int(1e8))
    groups.append(update_tstp_op)
    if params["restrict"]: groups.append(restrict_op)
    ######################lookup################################
    # id_val, id_idx = tf.unique(tf.reshape(ids, (-1,)))
    sparse_weights, trainable_wrapper = de.embedding_lookup(embeddings, id_val, return_trainable=True, name="lookup")
    # tf.compat.v1.add_to_collections(de.GraphKeys.DYNAMIC_EMBEDDING_VARIABLES, embeddings)
    weights = tf.gather(sparse_weights, id_idx)  #(None * 150, 9)
    ######################lookup end################################
    with open(f"{dirname}/slot.conf") as f:
        slots = [l.strip("\n") for l in f]
    
    emb_lookuped = tf.reshape(weights, [-1, len(slots) * embedding_size])
    logger.info(f"------embeddings: {embeddings} emb_lookuped: {emb_lookuped} -------")
    with tf.name_scope("dnn"):
        model = DnnModel(features, emb_lookuped, is_training)
    # model.predictions = {"value": feat_val, "sign": id_val, "emb": sparse_weights}
    ######################metrics################################
    global_step = tf.compat.v1.train.get_or_create_global_step()
    loggings = OrderedDict({
        "step": global_step,
        "emb_size": embeddings.size(),
        "loss": model.loss
    })
    with tf.name_scope('metrics'):
        eval_metric_ops = OrderedDict()
        for i, (label, prob) in enumerate(zip(model.labels, model.probs), start=1):
            evaluate(label, prob, f"task{i}", eval_metric_ops)
        for k in eval_metric_ops:
            loggings[k] = eval_metric_ops[k][0]
            groups.append(eval_metric_ops[k][1])
        if params["slot"]:
            eval_metric_ops[f'slot_{params["slot"]}'] = eval_metric_ops[k]

    ######################train################################
    if mode == tf.estimator.ModeKeys.TRAIN:
        trainable_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        dense_vars = [var for var in trainable_variables if not var.name.startswith("lookup")]
        sparse_vars = [var for var in trainable_variables if var.name.startswith("lookup")]
        global_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)

        logger.debug(f"trainable_variables: {trainable_variables}")
        logger.debug(f"dense_vars: {dense_vars}")
        logger.debug(f"sparse_vars: {sparse_vars}")
        logger.debug(f"global_vars: {global_variables}")
        logger.debug(f"l2_loss_vars: {[v for v in trainable_variables if 'bias' not in v.name]}")
        logger.debug("#" * 100)
        ######################l2_regularization################################
        l2_regularization = 1e-06
        l2_loss = l2_regularization * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in trainable_variables if 'bias' not in v.name])
        model.loss += l2_loss
        ######################optimizer################################
        # sparse_opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1)
        # sparse_opt = tf.compat.v1.train.AdamOptimizer()
        # sparse_opt = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(sparse_opt)
        # dense_opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.02 * params["lr"])
        #optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.02)
        optimizer = tf.compat.v1.train.AdamOptimizer()
        #optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=0.001)
        optimizer = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(optimizer)
        # if 1 == 1:
        #     #同步模式
        #     dense_opt = tf.compat.v1.train.SyncReplicasOptimizer(dense_opt,
        #                                             replicas_to_aggregate=params["task_number"],
        #                                             total_num_replicas=params["task_number"],
        #                                             use_locking=True)
        #     dense_opt = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(dense_opt)
        #     sync_replicas_hook = dense_opt.make_session_run_hook(params["task_idx"] == 0)

        dense_op = optimizer.minimize(model.loss, global_step=global_step)#, var_list=dense_vars)
        train_op = tf.group(dense_op, *groups)

        log_hook = tf.compat.v1.estimator.LoggingTensorHook(loggings, every_n_iter=100)
        ######################WarmStartHook################################
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=model.predictions, loss=model.loss,
            train_op=train_op,
            training_hooks=[log_hook, ],
            training_chief_hooks=None#[sync_replicas_hook]
        )
    elif mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            "serving_default": tf.compat.v1.estimator.export.PredictOutput(model.outputs)
        }
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=model.predictions,
                export_outputs=export_outputs)
    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=model.predictions,
                loss=model.loss,
                eval_metric_ops=eval_metric_ops)
