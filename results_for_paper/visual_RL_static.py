
# -*- coding: utf-8 -*-
# @Time    : 2021/9/13 10:25 上午
# @Author  : Chongming GAO
# @FileName: visual_RL_static.py

import argparse
import os
import re
import json
import seaborn as sns

import pandas as pd
from matplotlib import pyplot as plt

from util.utils import create_dir

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="/Users/gaochongming/fsdownload/explore")
    # parser.add_argument("--result_dir", type=str, default="../saved_models/PPO_realEnv/logs")



    args = parser.parse_known_args()[0]
    return args


def walk_paths(result_dir):
    g = os.walk(result_dir)

    files = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            print(os.path.join(path, file_name))
            files.append(file_name)
    return files


def loaddata(dirpath, filenames):
    pattern_epoch = re.compile("Epoch: \[(\d+)]")
    pattern_info = re.compile("Info: \[(\{.+})]")
    pattern_message = re.compile('"message": "(.+)"')

    dfs = {}
    df = pd.DataFrame()
    for filename in filenames:
        if filename[0] == '.' or filename[0] == '_':
            continue
        df = pd.DataFrame()
        message = "None"
        filepath = os.path.join(dirpath, filename)
        with open(filepath, "r") as file:
            try:
                lines = file.readlines()
            except Exception:
                raise "problem"
            for i, line in enumerate(lines):

                res = re.search(pattern_epoch, line)
                if res:
                    epoch = int(res.group(1))
                    info = re.search(pattern_info, line)
                    data = json.loads(info.group(1).replace("\'", "\""))
                    df_data = pd.DataFrame(data, index=[epoch])
                    df = df.append(df_data)
                res_message = re.search(pattern_message, line)
                if res_message:
                    message = res_message.group(1)
        dfs[message] = df

    indices = [list(dfs.keys()), df.columns.to_list()]

    df_all = pd.DataFrame(columns=pd.MultiIndex.from_product(indices, names=["Exp", "metrics"]))

    for message, df in dfs.items():
        for col in df.columns:
            df_all[message, col] = df[col]

    # change order of levels
    # https://stackoverflow.com/questions/29859296/how-do-i-change-order-grouping-level-of-pandas-multiindex-columns
    df_all.columns = df_all.columns.swaplevel(0, 1)
    df_all.sort_index(axis=1, level=0, inplace=True)
    return df_all


def visual(df_all, save_fig_dir):
    # visual_cols = ['RL_val_trajectory_reward', 'RL_val_trajectory_len', 'RL_val_CTR']
    # visual_cols = ['R_tra', 'len_tra', 'ctr']

    visual_cols = list(set([metric for (metric, name) in df_all.columns]))
    exps = sorted(list(set([name for (metric, name) in df_all.columns])))

    df_all = df_all[visual_cols]

    # LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    LINE_STYLES = ['solid', 'dashdot']
    LINE_STYLES = ["", (1,1)]

    for _, y_name in enumerate(visual_cols):
        fig = plt.figure()
        df = df_all[y_name].astype(float)
        # gca = sns.lineplot(data=df, palette='Set3')
        # gca = sns.lineplot(data=df, palette=sns.color_palette("Paired", df.columns.__len__()))

        mydash = LINE_STYLES * (len(df.columns)//2) + LINE_STYLES[:len(df.columns) % 2]
        # gca = sns.lineplot(data=df, palette=sns.color_palette("tab20", df.columns.__len__()),
        #                    dashes=mydash)
        # gca = sns.lineplot(data=df, palette="tab20", dashes=mydash)
        gca = sns.lineplot(data=df, palette="tab20")

        # for i, line in enumerate(gca.lines):
        #     line.set_linestyle(LINE_STYLES[i % len(LINE_STYLES)])

        plt.legend(bbox_to_anchor=(1.03, 1), loc=2, borderaxespad=0.)

        fig.savefig(os.path.join(save_fig_dir, y_name + '.eps'), format='eps', dpi=1000, bbox_inches='tight')
        fig.savefig(os.path.join(save_fig_dir, y_name + '.pdf'), format='pdf', bbox_inches='tight')

        # plt.show()
        # a = 1


def main(args):
    result_dir = args.result_dir

    realpath = os.path.dirname(__file__)
    save_fig_dir = os.path.join(realpath, "figures")
    save_fig_dir = result_dir

    create_dirs = [save_fig_dir]
    create_dir(create_dirs)

    filenames = walk_paths(result_dir)
    df = loaddata(result_dir, filenames)

    visual(df, save_fig_dir)


if __name__ == '__main__':
    args = get_args()
    main(args)
