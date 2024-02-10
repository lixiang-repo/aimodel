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

from common.dataset import input_fn, get_slot_list
from common.utils import write_donefile, get_metrics, serving_input_receiver_dense_fn

flags.DEFINE_string('model_dir', "./ckpt", 'export_dir')
flags.DEFINE_string('export_dir', "./export_dir", 'export_dir')
flags.DEFINE_string('mode', "train", 'train or export')
flags.DEFINE_string('warm_path', '', 'warm start path')
flags.DEFINE_string('data_path', '', 'data path')
flags.DEFINE_string("type", "join", "join or update model")
# flags.DEFINE_string('time_format', '%Y%m%d/%H/%M', 'time format for training')
flags.DEFINE_string('time_format', '%Y%m%d', 'time format for training')
flags.DEFINE_string('time_str', '202305270059', 'training time str')
flags.DEFINE_float('lr', 1.0, 'lr dense train ')
flags.DEFINE_string('file_list', '', 'file list')
flags.DEFINE_string('slot', "", 'miss slot')
FLAGS = flags.FLAGS

dirname = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger('tensorflow')
logger.propagate = False

tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
task_type = tf_config.get('task', {}).get('type', "chief")
task_idx = tf_config.get('task', {}).get('index', 0)
ps_num = len(tf_config.get('cluster', {}).get('ps', []))
task_number = len(tf_config.get('cluster', {}).get('worker', [])) + 1
task_idx = task_idx + 1 if task_type == 'worker' else task_idx

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


def train(filenames, params, model_config, steps=None):
    # ==========  执行任务  ========== #
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=params, config=model_config)
    if FLAGS.mode == "train":
        # train_spec = tf.estimator.TrainSpec(
        #     input_fn=lambda: input_fn(filenames, task_number, task_idx, shuffle=True))
        # eval_spec = tf.estimator.EvalSpec(
        #     input_fn=lambda: input_fn([x.replace(".gz", "_SUCCESS") for x in filenames[:2]], task_number, task_idx),
        #     start_delay_secs=1e20)
        # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        estimator.train(lambda: input_fn(filenames, task_number, task_idx), steps=steps)

    elif FLAGS.mode in ['eval', 'feature_eval']:
        estimator.evaluate(input_fn=lambda: input_fn(filenames, task_number, task_idx))

    elif FLAGS.mode in ["infer", "embeddings"]:
        outs = list(estimator.predict(input_fn=lambda: input_fn(filenames, task_number, task_idx)))
        print("outs>>>", outs[:10])
        df = pd.DataFrame(map(lambda x: x["out"], outs))
        df.index = map(lambda x: x["id"].decode("utf8"), outs)
        df.index.name = "id"

        if FLAGS.mode == "embeddings":
            df = df.reset_index().drop_duplicates(["id"] + list(range(9))).sort_values("id")
            df.to_csv("./embeddings.csv", sep="\t", index=False)
        else:
            model_dir = os.path.dirname(os.path.dirname(FLAGS.model_dir))
            df.to_csv("%s/logs/pred_%s.csv" % (model_dir, FLAGS.time_str), sep="\t")
            
            # metric_map = get_metrics(df)
        # df = df.reset_index().drop_duplicates(["id"] + list(range(9))).sort_values("id")
        # df.to_csv("./embeddings.csv", sep="\t", index=False)

    elif FLAGS.mode == "dump":
        tf.compat.v1.logging.info("lx>>>%s" % estimator.get_variable_names())
        keys = estimator.get_variable_value("embeddings/embeddings_mht_1of1-keys")
        values = estimator.get_variable_value("embeddings/embeddings_mht_1of1-values")
        emb = np.concatenate((np.reshape(keys, [-1, 1]), values), axis=1)
        np.savetxt("%s/emb.txt" % FLAGS.model_dir, emb, fmt=['%d'] + ['%.6f'] * 9)
    elif FLAGS.mode == "preview":
        tf.compat.v1.logging.info("lx>>>%s" % estimator.get_variable_names())
        values = estimator.get_variable_value("embeddings/embeddings_mht_1of1-values")
        np.savetxt("%s/var.txt" % FLAGS.model_dir, values, fmt='%.6f')
    elif FLAGS.mode == "export" and task_type == "chief" and int(task_idx) == 0:
        tfra.dynamic_embedding.enable_inference_mode()
        estimator.export_saved_model(FLAGS.export_dir, lambda: serving_input_receiver_dense_fn())

def main(argv):
    del argv
    t0 = time.time()
    assert FLAGS.type in ["join", "update"]
    assert FLAGS.mode in ["train", "eval", "infer", "export", "embeddings", "feature_eval", "dump", "preview"]
    params = {
        "mode": FLAGS.mode,
        "type": FLAGS.type,
        "ps_num": ps_num,
        "task_number": task_number,
        "task_type": task_type,
        "task_idx": task_idx,
        "warm_path": FLAGS.warm_path,
        "lr": FLAGS.lr,
        "slot": FLAGS.slot,
        "restrict": False
    }
    model_config = tf.estimator.RunConfig().replace(
        keep_checkpoint_max=1,
        save_checkpoints_steps=100000,
        log_step_count_steps=5000,
        save_summary_steps=10000,
        session_config=tf.compat.v1.ConfigProto(
            inter_op_parallelism_threads=os.cpu_count() // 2,
            intra_op_parallelism_threads=os.cpu_count() // 2))


    # filenames = ["/tf/data/matchmaking_girls_real_feas-tfrecord-train/part-r-00000.gz"]
    if len(FLAGS.file_list) == 0:
        time_format = parse(FLAGS.time_str).strftime(FLAGS.time_format)
        filenames = glob.glob("%s/%s" % (FLAGS.data_path, time_format))
        i = 0
        while len(filenames) < 5:
            i += 1
            if i % 100 == 0:
                tf.compat.v1.logging.info("file not exits %s %s" % ("%s/%s" % (FLAGS.data_path, time_format), filenames))
            time.sleep(300)
            filenames = glob.glob("%s/%s" % (FLAGS.data_path, time_format))
    else:
        with open(FLAGS.file_list) as f:
            filenames = [l.strip("\n") for l in f]

    tf.compat.v1.logging.info("filenames>>>%s" % filenames[:10])
    if FLAGS.mode == "export":
        params["lr"] = 0.0
        FLAGS.mode = "train"
        train(filenames, params, model_config, 1)
        params["warm_path"] = ""
        FLAGS.mode = "export"
        train([], params, model_config)
    
    elif FLAGS.mode == "train":
        n = len(filenames) // 336 + 1
        for i in range(n):
            files = filenames[i * 336:(i + 1) * 336]

            t1 = time.time()
            params["restrict"] = False
            train(files, params, model_config)
            tf.compat.v1.logging.info("waste time>>>%s>>>%s mins" % (n, (time.time() - t1) / 60))

            if FLAGS.type == "update":
                t2 = time.time()
                params["restrict"] = True
                train(files, params, model_config, 1)
                tf.compat.v1.logging.info("restrict waste time>>>%s mins" % ((time.time() - t2) / 60))
    elif FLAGS.mode == "feature_eval":
        train(filenames, params, model_config)
        with open("%s/slot.conf" % dirname) as rf:
            slots = [re.split(" +", l)[0] for l in rf if not (l.startswith("#") or l.startswith("label"))]
        for slot in slots:
            FLAGS.slot = params["slot"] = slot
            train(filenames, params, model_config)
    else:
        train(filenames, params, model_config)

    if FLAGS.mode == "train" and task_type == "chief" and int(task_idx) == 0:
        write_donefile(FLAGS.time_str, FLAGS.type, "%s/../../donefile" % FLAGS.model_dir)

    msg = "total waste>>>%s>>>%s>>>%s>>>%s>>>%s>>>%s mins" % (task_type, task_idx, FLAGS.mode, FLAGS.type, FLAGS.time_str, (time.time() - t0) / 60)
    tf.compat.v1.logging.info(msg)


if __name__ == "__main__":
    app.run(main)
