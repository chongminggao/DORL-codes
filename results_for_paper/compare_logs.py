# -*- coding: utf-8 -*-

import argparse
import os

import sys
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import seaborn as sns

import re
import json

# matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

CODEPATH = os.path.dirname(__file__)
ROOTPATH = os.path.dirname(CODEPATH)
sys.path.extend([ROOTPATH])

CODEPATH = os.path.dirname(__file__)
ROOTPATH = os.path.dirname(CODEPATH)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="./saved_models/VirtualTB-v0/CIRS/logs")
    parser.add_argument("--use_filename", type=str, default="Yes")
    parser.add_argument("--user_model_name", type=str, default="DeepFM")
    parser.add_argument("--read_message", type=str, default="point")

    args = parser.parse_known_args()[0]
    return args

def loaddata(dirpath, filenames, args, is_info=False):
    pattern_epoch = re.compile("Epoch: \[(\d+)]")
    pattern_info = re.compile("Info: \[(\{.+\})]")
    pattern_message = re.compile('"message": "(.+)"')
    pattern_array = re.compile("array\((.+?)(,\s*dtype=.+?)*\)")

    pattern_tau = re.compile('"tau": (.+),')
    pattern_read = re.compile('"read_message": "(.+)"')

    dfs = {}
    infos = {}
    df = pd.DataFrame()
    for filename in filenames:
        # if filename == ".DS_Store":
        #     continue
        if filename[0] == '.' or filename[0] == '_':
            continue
        df = pd.DataFrame()
        message = "None"
        filepath = os.path.join(dirpath, filename)
        with open(filepath, "r") as file:
            lines = file.readlines()
            start = False
            add = 0
            info_extra = {'tau':0, 'read':""}
            for i, line in enumerate(lines):
                res_tau = re.search(pattern_tau, line)
                if res_tau:
                    info_extra['tau'] = res_tau.group(1)
                res_read = re.search(pattern_read, line)
                if res_read:
                    info_extra['read'] = res_read.group(1)

                res = re.search(pattern_epoch, line)
                if res:
                    epoch = int(res.group(1))
                    if (start == False) and epoch == 0:
                        add = 1
                        start = True
                    epoch += add
                    info = re.search(pattern_info, line)
                    try:
                        info1 = info.group(1).replace("\'", "\"")
                    except Exception as e:
                        print("jump incomplete line: [{}]".format(line))
                        continue
                    info2 = re.sub(pattern_array, lambda x: x.group(1), info1)

                    data = json.loads(info2)
                    df_data = pd.DataFrame(data, index=[epoch],dtype=float)
                    df = df.append(df_data)
                res_message = re.search(pattern_message, line)
                if res_message:
                    message = res_message.group(1)

            if args.use_filename == "Yes":
                message = filename[:-4]

            # df.rename(
            #     columns={"RL_val_trajectory_reward": "R_tra",
            #              "RL_val_trajectory_len": 'len_tra',
            #              "RL_val_CTR": 'ctr'},
            #     inplace=True)
            # # print("JJ", filename)
            # df = df[["R_tra","len_tra","ctr"]]

            df = df[["val_MAE", "val_NDCG@5"]]

        dfs[message] = df
        infos[message] = info_extra

    dfs = OrderedDict(sorted(dfs.items(), key=lambda item: len(item[1]), reverse=True))

    indices = [list(dfs.keys()), df.columns.to_list()]

    df_all = pd.DataFrame(columns=pd.MultiIndex.from_product(indices, names=["Exp", "metrics"]))

    for message, df in dfs.items():
        # print(message, df)
        for col in df.columns:
            df_all[message, col] = df[col]

    # # Rename MultiIndex columns in Pandas
    # # https://stackoverflow.com/questions/41221079/rename-multiindex-columns-in-pandas
    # df_all.rename(
    #     columns={"RL_val_trajectory_reward": "R_tra", "RL_val_trajectory_len": 'len_tra', "RL_val_CTR": 'ctr'},
    #     level=1,inplace=True)

    # change order of levels
    # https://stackoverflow.com/questions/29859296/how-do-i-change-order-grouping-level-of-pandas-multiindex-columns
    df_all.columns = df_all.columns.swaplevel(0, 1)
    df_all.sort_index(axis=1, level=0, inplace=True)

    if is_info:
        return df_all, infos

    return df_all


def walk_paths(result_dir):
    g = os.walk(result_dir)

    files = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if file_name[0] == '.' or file_name[0] == '_' or\
                    os.path.basename(path)[0] == '.' or os.path.basename(path)[0] == '_':
                continue
            print(os.path.join(path, file_name))
            files.append(file_name)
    return files

def get_data(dataset):
    if dataset == "CoatEnv":
        env = "CoatEnv-v0"
        yname = "rating"

    if dataset == "YahooEnv":
        env = "YahooEnv-v0"
        yname = "rating"

    if dataset == "KuaiRandEnv":
        env = "KuaiRand-v0"
        yname = "is_click"

    if dataset == "KuaiEnv":
        env = "KuaiEnv-v0"
        yname = "watch_ratio_normed"

    return env, yname

def visual(df, savepath=CODEPATH, filename="aCompare"):


    metrics = list(set([metric for (metric, name) in df.columns]))
    exps = sorted(list(set([name for (metric, name) in df.columns])))


    fig = plt.figure(figsize=(12, 7))
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.3)
    axs = []

    ax1 = plt.subplot2grid((1, 2), (0, 0))
    df[metrics[0]].plot(kind="line", linewidth=1, ax=ax1, legend=None)

    ax2 = plt.subplot2grid((1, 2), (0, 1))
    df[metrics[1]].plot(kind="line", linewidth=1, ax=ax2, legend=None)

    legend_name = df[metrics[0]].columns.to_list()
    ax1.legend(labels=legend_name, ncol=5,
               loc='lower left',
               bbox_to_anchor=(0.3, 1.02), fontsize=11)

    plt.savefig(os.path.join(savepath, f"{filename}.pdf", ), bbox_inches='tight', pad_inches=0)

    b = df[metrics[0]]
    c = df[metrics[1]]

    plt.close(fig)

if __name__ == '__main__':
    args = get_args()



    datasets = [
        # "CoatEnv",
        # "YahooEnv",
        "KuaiRandEnv",
        # "KuaiEnv"
    ]
    for i in range(len(datasets)):
        envname, _ = get_data(datasets[i])

    UM_SAVE_PATH = os.path.join(ROOTPATH, "saved_models", envname, args.user_model_name)
    result_dir = os.path.join(UM_SAVE_PATH, "logs")

    result_dir = "/Users/gaochongming/Downloads/kuairand_res"
    result_dir = "/Users/gaochongming/Downloads/coat_res"
    result_dir = "/Users/gaochongming/Downloads/yahoo_res"

    filenames = walk_paths(result_dir)

    df = loaddata(result_dir, filenames, args)

    visual(df, CODEPATH, os.path.basename(result_dir))
