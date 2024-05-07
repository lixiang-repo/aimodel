# coding=utf-8
import pandas as pd
from odps import ODPS
import sys, os
import time
import gzip
from odps.tunnel import TableTunnel
import threading
import queue
import multiprocessing

dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(f"{dirname}/schema.conf") as f:
    schema = [l.strip("\n") for l in f]
with open(f"{dirname}/slot.conf") as f:
    slots = [l.strip("\n") for l in f]

def get_odps():
    """ 链接odps """
    odps_obj = ODPS(your_accesskey_id, your_accesskey_secret, your_default_project, endpoint=your_end_point,
                    tunnel_endpoint=tunnel_endpoint)
    return odps_obj

def features_mapper(features):
    lst = features.strip("\n").split("\002")
    if len(lst) != len(schema):
        return ""
    return "\002".join([v for k, v in zip(schema, lst) if k in slots])

def write(file, batch):
    df = batch.to_pandas()[["user_id", "requestid", "combination_un_id", "is_click", "features"]]
    # df["features"] = df["features"].map(features_mapper)
    # df = df[df["features"] != ""]
    df.to_csv(file, mode="a", sep="\t", compression="gzip", index=False, header=None, )

def consumer(q, i):
    while True:
        data = q.get()
        if data is None:
            break
        file, batch = data
        write(file % str(i).zfill(2), batch)

def task(idx_date, idx, start, end):
    time.sleep(1)
    print(f'idx_date: {idx_date} {start} {end}')

    t0 = time.time()
    basepath = f"/data/share/data/deep_rank_v7/{idx_date}"
    success = f"{basepath}/_SUCCESS"
    if os.path.exists(success):
        print(f"success file exits: {success}")
        return
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    #################################################################################
    odps = get_odps()
    tunnel = TableTunnel(odps)
    table = odps.get_table("adx_dmp.dnn_online_deep_rank_rank_sample_fg_encoded_v7")
    for i in range(int(1e20)):
        if table.exist_partition(f"idx_date={idx_date}"):
            break
        if i % 100 == 0:
            print(f"idx_date not exits: {idx_date}")
        time.sleep(300)
    #################################################################################
    q = queue.Queue()
    # 创建多个线程并启动
    threads = []
    for i in range(8):
        t = threading.Thread(target=consumer, args=(q, i, ))
        t.start()
        threads.append(t)

    download_session = tunnel.create_download_session('dnn_online_deep_rank_rank_sample_fg_encoded_v7', partition_spec=f'idx_date={idx_date}')
    # file = f"{basepath}/part-r-{idx}-t-%s-s-{start}-{end}.gz"
    file = f"{basepath}/part-r-{idx}-t-%s-{start}-{end}.gz"
    with download_session.open_arrow_reader(start, end) as reader:
        for batch in reader:
            q.put((file, batch))

    # open(success, "w").close()
    # 等待所有线程结束
    for t in threads:
        t.join()
    t1 = time.time()
    time.sleep(1)
    print(f'idx_date: {idx_date} {start} {end} waste: {(t1 - t0) // 60}')

if __name__ == '__main__':
    # for day in ['20230930', '20231031', '20231130', '20231231', '20240131', '20240229', '20240331', '20240415']:
    #     print(f"start: {day}")
    #     main(day)
    # for day in pd.date_range("20230928", "20500101").map(lambda x: x.strftime("%Y%m%d")):
    #     print(f"start: {day}")
    #     main(day)
    # for day in ['20230930', '20231031', '20231130', '20231231', '20240131', '20240229', '20240331', '20240415']:
    # day = '20240416'

    for day in pd.date_range("20240301", "20500101").map(lambda x: x.strftime("%Y%m%d")):
        t0 = time.time()
        odps = get_odps()
        tunnel = TableTunnel(odps)
        pool = multiprocessing.Pool(8)
        download_session = tunnel.create_download_session('dnn_online_deep_rank_rank_sample_fg_encoded_v7', partition_spec=f'idx_date={day}')
        step = download_session.count // 19
        for idx in range(20):
            start = idx * step
            end = min((idx + 1) * step - 1, download_session.count)
            print("task: ", day, idx, start, end)
            pool.apply_async(task, args=(day, idx, start, end))

        pool.close()
        pool.join()

        success = f"/data/share/data/deep_rank_v7/{day}/_SUCCESS"
        t1 = time.time()
        f = open(success, "w")
        f.write(f"{(t1 - t0) / 60:2.f}")
        f.close()
