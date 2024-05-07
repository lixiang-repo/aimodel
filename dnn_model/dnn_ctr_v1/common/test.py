import multiprocessing
import os, time


def work(i):
    print(f'进程名{multiprocessing.current_process().name}，{i}')
    time.sleep(5)


if __name__ == '__main__':
    print('--------开始主进程------------')
    print(f'主进程名{multiprocessing.current_process().name},进程号{os.getpid()}')
    start_time = time.time()
    pool = multiprocessing.Pool(3)
    for i in range(1, 6):
        pool.apply_async(func=work, args=(i, ))
    time.sleep(1)
    pool.close()
    pool.join()
    end_time = time.time()
    print(f'----------主进程结束，耗时:{end_time - start_time}-------------------')

#####运行结果输出：
#--------开始主进程------------
#主进程名MainProcess,进程号19360
#进程名SpawnPoolWorker-1，进程号17640
#进程名SpawnPoolWorker-2，进程号18588
#进程名SpawnPoolWorker-4，进程号18404
#进程名SpawnPoolWorker-3，进程号18376
#进程名SpawnPoolWorker-1，进程号17640
#----------主进程结束，耗时:2.1518847942352295-------------------
