#!/usr/bin/env python
# coding=utf-8
import oss2, os

dirname = os.path.dirname(os.path.abspath(__file__))
basename = os.path.basename(dirname)

auth = oss2.Auth('', '')
bucket = oss2.Bucket(auth, "", "")

def parse_int(x):
    try:
        return int(x)
    except:
        return -1

def upload_folder(folder_name, target_folder=None):
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            # 构造文件的完整路径
            file_path = os.path.join(root, file)
            # 构造OSS上的路径
            if target_folder:
                target_path = os.path.join(target_folder, file_path.lstrip(os.path.sep))
            else:
                target_path = file_path.lstrip(os.path.sep)
            # 上传文件
            with open(file_path, 'rb') as fileobj:
                bucket.put_object(target_path, fileobj)
 
if __name__ == "__main__":
    os.chdir(f"/data/share/model/{basename}/export_dir")
    files = sorted(map(parse_int, os.listdir(".")))
    print(f"files: {files}")
    if os.path.exists(f"{files[-1]}") and not bucket.object_exists(f"model/{basename}/{files[-1]}/saved_model.pb"):
        print(f"upload_folder: {files[-1]}")
        #basename = "dnn_ctr_v1"
        upload_folder(f"{files[-1]}", f"model/{basename}")
    
