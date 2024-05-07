#!/usr/bin/env python
# coding=utf-8
import sys

import tensorflow as tf
import tensorflow_recommenders_addons as tfra
import pandas as pd
from absl import app, flags
from common.model_fn import model_fn
import os
import json
import time
import glob, datetime
from dateutil.parser import parse
import numpy as np
import logging
import re
from tqdm import tqdm
from collections import defaultdict

from common.dataset import input_fn
from common.utils import write_donefile, get_metrics, serving_input_receiver_dense_fn

flags.DEFINE_string('job_name', "", 'export_dir')
flags.DEFINE_string('ckpt_dir', "./ckpt", 'export_dir')
flags.DEFINE_string('export_dir', "./export_dir", 'export_dir')
flags.DEFINE_string('mode', "train", 'train or export')
flags.DEFINE_string('data_path', '/data/share/data/deep_rank_v7', 'data path')
flags.DEFINE_string('time_str', '202403012359', 'training time str')
flags.DEFINE_string('file_list', '', 'file list')
flags.DEFINE_string('slot', "", 'miss slot')
FLAGS = flags.FLAGS

dirname = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger('tensorflow')
logger.propagate = False
logger = tf.compat.v1.logging

tf_config = json.loads(os.environ.get("TF_CONFIG") or '{}')
# tf_config = eval(os.environ.get("TF_CONFIG") or '{}')

task_type = tf_config.get('task', {}).get('type', "chief")
task_idx = task_index = tf_config.get('task', {}).get('index', 0)
ps_num = len(tf_config.get('cluster', {}).get('ps', []))
task_number = len(tf_config.get('cluster', {}).get('worker', [])) + 1
task_idx = task_idx + 1 if task_type == 'worker' else task_idx

logger.set_verbosity(logger.DEBUG)
# logger.set_verbosity(logger.INFO)


def train(filenames, params, model_config, steps=None):
    # ==========  执行任务  ========== #
    model_dir = os.path.dirname(FLAGS.ckpt_dir)
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.ckpt_dir, params=params, config=model_config)
    if FLAGS.mode == "train":
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(filenames, model_dir, task_number, task_idx))
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn([f"{os.path.dirname(filenames[0])}/_SUCCESS"], model_dir, task_number, task_idx),
            start_delay_secs=1e20)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        #estimator.train(lambda: input_fn(filenames, model_dir, task_number, task_idx), steps=steps)

    elif FLAGS.mode in ['eval', 'feature_eval']:
        estimator.evaluate(input_fn=lambda: input_fn(filenames, model_dir, task_number, task_idx))

    elif FLAGS.mode == "infer":
        with open(f"{model_dir}/logs/pred_{FLAGS.time_str}.csv", "w") as wf:
            for x in tqdm(estimator.predict(input_fn=lambda: input_fn(filenames, model_dir))):
                lst = [x["id"].decode("utf8")] + list(map(str, x["out"]))
                wf.write("\t".join(lst) + "\n")

    elif FLAGS.mode == "dump":
        logger.info(f"lx: {estimator.get_variable_names()}")
        keys = estimator.get_variable_value("embeddings/embeddings_mht_1of1-keys")
        values = estimator.get_variable_value("embeddings/embeddings_mht_1of1-values")
        emb = np.concatenate((np.reshape(keys, [-1, 1]), values), axis=1)
        np.savetxt(f"{FLAGS.ckpt_dir}/emb.txt", emb, fmt=['%d'] + ['%.16f'] * 9)
    elif FLAGS.mode == "preview":
        logger.info(f"lx: {estimator.get_variable_names()}")
        values = estimator.get_variable_value("embeddings/embeddings_mht_1of1-values")
        np.savetxt(f"{FLAGS.ckpt_dir}/var.txt", values, fmt='%.16f')
    elif FLAGS.mode == "export" and task_type == "chief" and int(task_idx) == 0:
        tfra.dynamic_embedding.enable_inference_mode()
        estimator.export_saved_model(FLAGS.export_dir, lambda: serving_input_receiver_dense_fn())

    elif FLAGS.mode == "emb":
        outs = estimator.predict(input_fn=lambda: input_fn(filenames, model_dir))
        with open(f"{model_dir}/emb.txt", "w") as wf:
            for out in tqdm(outs):
                sign = out["sign"]
                value = out["value"].decode()
                emb = "".join(map(lambda x: f"{x:.6f}", out["emb"]))
                wf.write(f"{sign}\t{value}\t{emb}\n")


def main(argv):
    del argv
    model_dir = os.path.dirname(FLAGS.ckpt_dir)
    t0 = time.time()
    params = {
        "mode": FLAGS.mode,
        "ps_num": ps_num,
        "task_number": task_number,
        "task_type": task_type,
        "task_idx": task_idx,
        "slot": FLAGS.slot,
        "restrict": False,
    }
    model_config = tf.estimator.RunConfig().replace(
        keep_checkpoint_max=1,
        save_checkpoints_steps=100000,
        log_step_count_steps=5000,
        save_summary_steps=10000,
        session_config=tf.compat.v1.ConfigProto(
            inter_op_parallelism_threads=os.cpu_count() // 2,
            intra_op_parallelism_threads=os.cpu_count() // 2))

    # filenames = parse(FLAGS.time_str).strftime("%Y%m%d")
    # filenames = ["/tf/data/matchmaking_girls_real_feas-tfrecord-train/part-r-00000.gz"]
    if len(FLAGS.file_list) == 0:
        success = f"{FLAGS.data_path}/{parse(FLAGS.time_str).strftime('%Y%m%d/_SUCCESS')}"
        for i in range(int(1e20)):
            if os.path.exists(success):
                break
            if i % 1000 == 0:
                logger.info(f"success file not exits: {success}")
            time.sleep(30)
        file_pattern = f"{FLAGS.data_path}/{parse(FLAGS.time_str).strftime('%Y%m%d/part-*.gz')}"
        # filenames = sorted(glob.glob(file_pattern), key=lambda x: int(x.split("/")[-1].split(".")[0].split("-")[-1]))
        filenames = sorted(glob.glob(file_pattern), key=hash)
    else:
        with open(FLAGS.file_list) as f:
            filenames = [l.strip("\n") for l in f]
    
    logger.info(f"filenames: {filenames[:10]}")
    if FLAGS.mode == "export":
        train(filenames, params, model_config)
    
    elif FLAGS.mode == "train":
        t1 = time.time()
        params["restrict"] = False
        train(filenames, params, model_config)

        # t2 = time.time()
        # params["restrict"] = True
        # train(filenames, params, model_config, 1)
        # logger.info(f"restrict waste time>>>{(time.time() - t2) / 60:.2f} mins")

        # if FLAGS.mode == "train" and task_type == "chief" and int(task_idx) == 0:
        write_donefile(FLAGS.time_str, f"{model_dir}/logs/donefile.{task_idx}")
    elif FLAGS.mode == "feature_eval":
        train(filenames, params, model_config)
        with open(f"{dirname}/slot.conf") as rf:
            slots = [re.split(" +", l)[0] for l in rf if not (l.startswith("#") or l.startswith("label"))]
        for slot in slots:
            FLAGS.slot = params["slot"] = slot
            train(filenames, params, model_config)
    else:
        train(filenames, params, model_config)

    msg = f"mode: {FLAGS.mode} task_type: {task_type} task_idx: {task_idx} time_str: {FLAGS.time_str} waste: {(time.time() - t0) / 60:.2f} mins"
    logger.info(msg)


if __name__ == "__main__":
    app.run(main)

