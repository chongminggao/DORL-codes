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
from matplotlib.ticker import MaxNLocator

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def get_certain_methods(df_visual, var_groups, pattern, group_num):
    # pattern = re.compile("var(\d+)")
    methods_dict = {}

    for method in df_visual.index:
        res = pattern.match(method)
        if res:
            var_n = int(res.group(group_num))
            methods_dict[method] = var_n

    reserved_methods = [method for method in methods_dict if methods_dict[method] in var_groups]
    df_visual1 = df_visual.loc[reserved_methods]

    return df_visual1



def draw_bar_line(df_visual, save_fig_dir, visual_methods_dict, savename="result_entropy", line_left=r'Length', line_right=r'$\text{R}_\text{tra}$', bar="MCD"):
    var_groups = [5,8]
    df_visual1 = get_certain_methods(df_visual, var_groups, re.compile("var(\d+)"), 1)
    df_visual1.rename({k:v for k,v in visual_methods_dict.items()},inplace=True)

    # df_visual2 = df_visual1.rename({k:v for k,v in visual_methods_dict.items()})


    df_visual_mean = df_visual1.groupby("Exp").agg(np.mean)
    df_visual_std = df_visual1.groupby("Exp").agg(np.std)

    line_left = r'$\text{R}_\text{tra}$'
    line_left = r'Length'
    visual_range = [0,1,2,3,4,5,6,7,]
    df_visual_mean = df_visual_mean.loc[visual_range]
    df_visual_std = df_visual_std.loc[visual_range]

    x_ticks = {0: 0, 1: 0.001, 2: 0.005, 3: 0.01, 4: 0.05, 5: 0.1, 6: 0.5, 7: 1, 8: 5, 9: 10, 10: 50, 11: 100}
    x_ticks_filtered = {k: x_ticks[k] for k in visual_range}

    colors = sns.color_palette()
    color_bar =  colors[0]
    red2 = [1.        , 0.75294118, 0.79607843]

    fig = plt.figure(figsize=(2.5, 3))
    # ax_bar = ax_left.twinx()
    ax_bar = sns.barplot(data=df_visual_mean, x=df_visual_mean.index, y=bar, edgecolor=red2, color=red2)
    plt.ylim([0.3, 0.63])
    plt.xlabel(r"$\lambda_2$", fontsize=12)
    plt.ylabel("Majority Domination", fontsize=12)
    ax_bar.xaxis.set_label_coords(0.5, -0.25)
    ax_bar.yaxis.label.set_color(color_bar)
    ax_bar.tick_params(axis='y', colors=color_bar)
    plt.xticks(range(len(visual_range)), x_ticks_filtered.values(), rotation=40)


    ax_left = ax_bar.twinx()
    ax_left = sns.lineplot(data=df_visual_mean, x=range(len(visual_range)), y=line_left, ax=ax_left,
                           marker='o',markeredgecolor=None)
    # ax_left.fill_between(x=range(len(visual_range)),
    #                      y1=df_visual_mean[line_left] - df_visual_std[line_left],
    #                      y2=df_visual_mean[line_left] + df_visual_std[line_left],
    #                      color=colors[0], alpha=.2)
    # ax_left.yaxis.tick_left()
    # ax_left.yaxis.label_left()
    ax_left.yaxis.label.set_color(colors[0])
    ax_left.tick_params(axis='y', colors=colors[0])
    plt.ylabel("Length", fontsize=12)

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
    plt.show()
    plt.close()


def visual_ent(df_avg, save_fig_dir):
    way = "Free"
    metrics = df_avg.columns.levels[1]

    # var_pattern = re.compile("var(\d+)$")
    # visual_methods_dict = {}
    #
    # for method in df_avg.index:
    #     res = var_pattern.match(method)
    #     if res:
    #         var_n = int(res.group(1))
    #         visual_methods_dict[method] = var_n

    var_pattern = re.compile("var(\d+)_ent(\d+)")
    # df_visual = pd.DataFrame([], columns=metrics)
    visual_methods_dict = {}
    for method in df_avg.index:
        res = var_pattern.match(method)
        if res:
            var_n = int(res.group(2))
            visual_methods_dict[method] = var_n

    df_visual = df_avg.loc[visual_methods_dict.keys()]
    df_visual = df_visual[way]
    # df_visual.rename({k:v for k,v in visual_methods_dict.items()},inplace=True)
    df_visual.sort_index(inplace=True)

    draw_bar_line(df_visual, save_fig_dir, visual_methods_dict)


def get_entropy_group(df_avg):
    way = "Free"
    ent_pattern = re.compile("var(\d+)_ent(\d+)")
    # df_visual = pd.DataFrame([], columns=metrics)
    visual_methods_dict = {}
    for method in df_avg.index:
        res = ent_pattern.match(method)
        if res:
            var_n = int(res.group(2))
            visual_methods_dict[method] = var_n
    df_visual = df_avg.loc[visual_methods_dict.keys()]
    df_visual = df_visual[way]
    # df_visual.rename({k:v for k,v in visual_methods_dict.items()},inplace=True)
    df_visual.sort_index(inplace=True)
    return df_visual, visual_methods_dict



def draw(df_visual, line_left, bar, display_y_right=True, display_y_left=True,
         bar_ylim=None, title=None, line_ticks=None, bar_ticks=None):
    df_visual = df_visual.groupby("Exp").agg(np.mean)
    # df_visual_std = df_visual1.groupby("Exp").agg(np.std)

    visual_range = [0, 1, 2, 3, 4, 5, 6, 7, ]
    df_visual = df_visual.loc[visual_range]
    # df_visual_std = df_visual_std.loc[visual_range]

    x_ticks = {0: 0, 1: 0.001, 2: 0.005, 3: 0.01, 4: 0.05, 5: 0.1, 6: 0.5, 7: 1, 8: 5, 9: 10, 10: 50, 11: 100}
    x_ticks_filtered = {k: x_ticks[k] for k in visual_range}

    colors = sns.color_palette()
    color_bar = colors[3]
    color_line = colors[0]
    color_line = "green"
    red2 = [1., 0.75294118, 0.79607843]

    # colors = sns.color_palette("Paired")
    # color_bar = colors[5]
    # color_line = colors[9]
    # red2 = colors[4]

    # fig = plt.figure(figsize=(2.5, 3))
    # ax_bar = ax_left.twinx()
    ax_bar = sns.barplot(data=df_visual, x=df_visual.index, y=bar, edgecolor=red2, color=red2)
    if bar_ylim:
        plt.ylim(bar_ylim)
    if display_y_right:
        plt.ylabel("Majority Domination", fontsize=12, rotation=90)
    else:
        plt.ylabel(None)
    # plt.yticks(rotation=90)
    # plt.yticks(rotation=45)
    # ax_bar.set_yticklabels(ax_bar.get_yticks(), rotation=270)
    # ax_bar.set_yticklabels(ax_bar.get_yticks())
    # plt.yticks(range(len(visual_range)), x_ticks_filtered.values())


    plt.xlabel(r"$\lambda_2$", fontsize=12)
    ax_bar.xaxis.set_label_coords(0.5, -0.3)
    ax_bar.yaxis.label.set_color(color_bar)
    ax_bar.tick_params(axis='y', colors=color_bar)
    plt.xticks(range(len(visual_range)), x_ticks_filtered.values(), rotation=45)
    if bar_ticks:
        plt.yticks(bar_ticks, bar_ticks)

    ax_left = ax_bar.twinx()
    ax_left = sns.lineplot(data=df_visual, x=range(len(visual_range)), y=line_left, ax=ax_left,
                           marker='o', markeredgecolor=None, color=colors[2])
    ax_left.yaxis.label.set_color(color_line)
    ax_left.tick_params(axis='y', colors=color_line)
    if display_y_left:
        plt.ylabel("Length", fontsize=12)
    else:
        plt.ylabel(None)
    # ax_left.set_yticklabels(ax_left.get_yticks(), rotation=90)

    ax_bar.yaxis.set_label_position("right")
    ax_bar.yaxis.tick_right()
    ax_left.yaxis.set_label_position("left")
    ax_left.yaxis.tick_left()
    if line_ticks:
        plt.yticks(line_ticks, line_ticks)

    # ax_bar.yaxis.set_major_locator(MaxNLocator(integer=True))
    # ax_left.yaxis.set_major_locator(MaxNLocator(integer=True))

    if title:
        plt.title(title)


def draw_bar_line2(df_avg_kuairec, df_avg_kuairand, save_fig_dir, visual_methods_dict, savename="result_entropy", line_left=r'Length', line_right=r'$\text{R}_\text{tra}$', bar="MCD"):
    var_groups_kuairec = [7]
    df_visual_kuairec = get_certain_methods(df_avg_kuairec, var_groups_kuairec, re.compile("var(\d+)"), 1)
    df_visual_kuairec.rename({k: v for k, v in visual_methods_dict.items()}, inplace=True)

    var_groups_kuairand = [5,8]
    df_visual_kuairand = get_certain_methods(df_avg_kuairand, var_groups_kuairand, re.compile("var(\d+)"), 1)
    df_visual_kuairand.rename({k:v for k,v in visual_methods_dict.items()},inplace=True)



    # df_visual2 = df_visual1.rename({k:v for k,v in visual_methods_dict.items()})

    fig = plt.figure(figsize=(5, 2))
    plt.subplots_adjust(wspace=0.45)

    plt.subplot2grid((1, 2), (0, 0))
    draw(df_visual_kuairec, line_left, bar, display_y_right=False, bar_ylim=[0.4, 0.65], title="KuaiRec", line_ticks=[14, 16, 18, 20, 22])

    plt.subplot2grid((1, 2), (0, 1))
    draw(df_visual_kuairand, line_left, bar, display_y_left=False,bar_ylim=[0.35, 0.6], title="KuaiRand", line_ticks=[24,25,26])


    fig.savefig(os.path.join(save_fig_dir, savename + '.pdf'), format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    print("done")

def visual_ent_two(df_avg_list, save_fig_dir):

    # var_groups= list(range(12))
    df_visual_list = []
    for df_avg in df_avg_list:
        df_visual, visual_methods_dict = get_entropy_group(df_avg)
        df_visual_list.append(df_visual)
    df_avg_kuairec = df_visual_list[0]
    df_avg_kuairand = df_visual_list[1]

    draw_bar_line2(df_avg_kuairec, df_avg_kuairand, save_fig_dir, visual_methods_dict)


if __name__ == '__main__':
    realpath = os.path.dirname(__file__)
    save_fig_dir = os.path.join(realpath, "figures")

    datasets = ["kuairec_12_0"]
    datasets = ["kuairec_2023 2022"]
    datasets = ["kuairec_10_2"]
    datasets = ["kuairec_10_1","kuairand51"]

    df_avg_list = []
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
        # savename = dataset
        df_latex, df_excel, df_avg = handle_table(df_all, methods=None)


        # visual_ent(df_avg, save_fig_dir)
        df_avg_list.append(df_avg)

    visual_ent_two(df_avg_list, save_fig_dir)




    # visual_bar_mix(df_avg, save_fig_dir, savename, is_group=False)
    # visual_bar_mix(df_avg, save_fig_dir, savename, is_group=True)
    print(f"All results of {dataset} rendered done!")