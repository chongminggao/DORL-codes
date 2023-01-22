# -*- coding: utf-8 -*-
# @Time    : 2023/1/19 21:06
# @Author  : Chongming GAO
# @FileName: visual_var_ent.py
import os
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from results_for_paper.visual_utils import walk_paths, loaddata, organize_df, handle_table

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42



def draw_bar_line(df_visual, save_fig_dir, savename="result_conservative", line_left=r'$\text{R}_\text{each}$', line_right=r'$\text{R}_\text{tra}$', bar="MCD"):
    df_visual2 = df_visual.groupby("Exp").agg(np.mean)

    visual_range = [0,2,3,4,5,6,7,8,9,10,11]
    df_visual2_filtered = df_visual2.loc[visual_range]

    x_ticks = {0: 0, 1: 0.001, 2: 0.005, 3: 0.01, 4: 0.05, 5: 0.1, 6: 0.5, 7: 1, 8: 5, 9: 10, 10: 50, 11: 100}
    x_ticks_filtered = {k: x_ticks[k] for k in visual_range}

    colors = sns.color_palette()
    color_bar =  colors[3]
    red2 = [1.        , 0.75294118, 0.79607843]

    fig = plt.figure(figsize=(5, 2))
    # ax_bar = ax_left.twinx()
    ax_bar = sns.barplot(data=df_visual2_filtered, x=df_visual2_filtered.index, y=bar, edgecolor=red2, color=red2)
    plt.ylim([0.475, 0.63])
    plt.ylabel("Majority Domination", fontsize=12)
    plt.xlabel(r"$\lambda$", fontsize=12)
    ax_bar.xaxis.set_label_coords(0.5, -0.25)
    ax_bar.yaxis.label.set_color(color_bar)
    ax_bar.tick_params(axis='y', colors=color_bar)
    plt.xticks(range(len(visual_range)), x_ticks_filtered.values(), rotation=40)


    ax_left = ax_bar.twinx()
    ax_left = sns.lineplot(data=df_visual2_filtered, x=range(len(visual_range)), y=line_left, ax=ax_left,
                           marker='o',markeredgecolor=None)
    # ax_left.yaxis.tick_left()
    # ax_left.yaxis.label_left()
    ax_left.yaxis.label.set_color(colors[0])
    ax_left.tick_params(axis='y', colors=colors[0])
    plt.ylabel("Reward", fontsize=12)

    ax_bar.yaxis.set_label_position("right")
    ax_bar.yaxis.tick_right()
    ax_left.yaxis.set_label_position("left")
    ax_left.yaxis.tick_left()

    # plt.axis('off')

    # ax_right = ax_left.twinx()
    #
    # sns.lineplot(data=df_visual2, x=df_visual2.index, y=line_right, ax=ax_right)
    # plt.ylabel("Cumulative Reward")
    # ax_right.yaxis.label.set_color('red')
    # ax_right.tick_params(axis='x', colors='red')
    # # plt.xticks(range(12), x_ticks.values(), rotation=40)
    #
    # plt.xticks(range(12), x_ticks.values(), rotation=40)


    fig.savefig(os.path.join(save_fig_dir, savename + '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    plt.show()

def visual_var(df_avg, save_fig_dir):
    way = "No Overlapping"
    metrics = df_avg.columns.levels[1]

    var_pattern = re.compile("var(\d+)$")
    visual_methods_dict = {}

    for method in df_avg.index:
        res = var_pattern.match(method)
        if res:
            var_n = int(res.group(1))
            visual_methods_dict[method] = var_n

    var_pattern = re.compile("var(\d+)")
    # df_visual = pd.DataFrame([], columns=metrics)
    visual_methods_dict = {}
    for method in df_avg.index:
        res = var_pattern.match(method)
        if res:
            var_n = int(res.group(1))
            visual_methods_dict[method] = var_n



    df_visual = df_avg.loc[visual_methods_dict.keys()]
    df_visual = df_visual[way]
    df_visual.rename({k:v for k,v in visual_methods_dict.items()},inplace=True)
    df_visual.sort_index(inplace=True)

    draw_bar_line(df_visual, save_fig_dir)


if __name__ == '__main__':
    realpath = os.path.dirname(__file__)
    save_fig_dir = os.path.join(realpath, "figures")

    datasets = ["kuairec_12_0"]
    datasets = ["kuairec_2023 2022"]
    datasets = ["kuairec_10_2"]

    for dataset in datasets:

        dirpath = os.path.join("./results", dataset)
        filenames = walk_paths(dirpath)
        print(f"Loading data for {dataset}...")
        dfs = loaddata(dirpath, filenames)

        ways = {'FB', 'NX_0_', 'NX_10_'}
        metrics = {"R_tra", "ctr", 'len_tra', 'CV', 'CV_turn'}
        all_metrics = list(list(dfs.values())[0].columns)
        pattern = re.compile("^ifeat_(.*)")
        # metrics = set()
        for metric in all_metrics:
            res = re.search(pattern, metric)
            if res:
                metrics.add("ifeat_" + res.group(1))

        df_all = organize_df(dfs, ways, metrics)

        print("Producing the table...")
        savename = dataset
        df_latex, df_excel, df_avg = handle_table(df_all, methods=None)

        visual_var(df_avg, save_fig_dir)




        # visual_bar_mix(df_avg, save_fig_dir, savename, is_group=False)
        # visual_bar_mix(df_avg, save_fig_dir, savename, is_group=True)
        print(f"All results of {dataset} rendered done!")