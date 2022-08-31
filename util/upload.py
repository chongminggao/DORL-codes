# -*- coding: utf-8 -*-
# @Time    : 2021/10/27 3:43 下午
# @Author  : Chongming GAO
# @FileName: upload.py
import re
import paramiko  # 用于调用scp命令
import os

# Todo: This is for only me, do not make public!
host = "chongming.myds.me"  # 服务器ip地址
port = 10022  # 端口号
username = "root"  # ssh 用户名
password = "laogaozhengao"  # 密码

ignore_list = [
    re.compile("\.DS_Store$"),
    re.compile("\.pickle$"),
    re.compile("\.pt$"),
    re.compile("\.csv$"),
    re.compile("events\."),
    re.compile("\.pth$"),
]

def get_sftp():
    sf = paramiko.Transport((host, port))
    sf.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(sf)
    return sftp

def is_ignored(name, ignore_list):
    for pattern in ignore_list:
        match = re.search(pattern, name)
        if match:
            return True
    return False

def sftp_exists(sftp, path):
    try:
        sftp.stat(path)
        return True
    except FileNotFoundError:
        return False


def sftp_upload(sftp, local, remote_path):
    n_path = 0
    n_file = 0
    n_existed = 0

    if is_ignored(local, ignore_list):
        return n_path, n_file, n_existed

    if os.path.isfile(local):
        # if not sftp_exists(sftp, os.path.join(remote_path, os.path.basename(local))): # Todo
        sftp.put(local, os.path.join(remote_path, os.path.basename(local)))
        n_file = 1
        return n_path, n_file, n_existed
        # else:
        #     n_existed = 1
        #     return n_path, n_file, n_existed

    dirname = os.path.basename(local)
    new_path = os.path.join(remote_path, dirname)

    if not sftp_exists(sftp, new_path):
        sftp.mkdir(new_path)
        n_path += 1

    for f in os.listdir(local):
        newlocal = os.path.join(local, f)
        a,b,c = sftp_upload(sftp, newlocal, new_path)
        n_path += a
        n_file += b
        n_existed += c

    return n_path, n_file, n_existed



def h_mkdir(sftp, remote_path):
    if sftp_exists(sftp, remote_path):
        return
    parent = os.path.dirname(remote_path)
    if not sftp_exists(sftp, parent):
        h_mkdir(sftp, parent)
    sftp.mkdir(remote_path)


def my_upload(local_path, remote_path, remote_root):
    sftp = get_sftp()

    h_mkdir(sftp, remote_path)

    n_path, n_file, n_existed = sftp_upload(sftp, local_path, remote_path)
    sftp.close()
    print("Files upload completed ======")
    print('    From: "{}"'.format(local_path))
    print('    To: "{}"'.format(remote_path))
    print('==== [{}] paths are created, [{}] files are uploaded. [{}] files existed'.format(n_path, n_file, n_existed))


if __name__ == '__main__':
    LOCAL_PATH = "../saved_models"
    REMOTE_PATH = "/root/Counterfactual_IRS/nihao"
    REMOTE_ROOT = "/root/Counterfactual_IRS"
    my_upload(LOCAL_PATH, REMOTE_PATH, REMOTE_ROOT)

