# -*- coding: utf-8 -*-


import os
import re
from collections import OrderedDict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox

import seaborn as sns

from results_for_paper.rename import rename_dataset
from visual_utils import walk_paths, loaddata, organize_df, create_dir


def axis_shift(ax1, x_shift=0.01, y_shift=0):
    position = ax1.get_position().get_points()
    pos_new = position
    pos_new[:, 0] += x_shift
    pos_new[:, 1] += y_shift
    ax1.set_position(Bbox(pos_new))


def compute_improvement(df, col, last=0):
    our = df.iloc[-5:][col]["CIRS"].mean()
    prev = df.iloc[-last:][col]["CIRS w_o CI"].mean()
    print(f"Improvement on [{col}] of last [{last}] count is {(our - prev) / prev}")


def draw(df_metric, ax1, color, marker, name):
    df_metric.plot(kind="line", linewidth=1, ax=ax1, legend=None, color=color, markevery=int(len(df_metric) / 10),
                   fillstyle='none', alpha=.8, markersize=3)
    # for i, line in enumerate(ax1.get_lines()):
    #     line.set_marker(marker[i])

    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.grid(linestyle='dashdot', linewidth=0.8)
    ax1.set_ylabel(name, fontsize=10, fontweight=700)


def visual(df_all, save_fig_dir, savename="three"):
    df_all.rename(columns={r"$\text{CV}_\text{M}$": r"CV_M"}, level=1,
                  inplace=True)
    ways = df_all.columns.levels[0]
    metrics = df_all.columns.levels[1]
    methods = df_all.columns.levels[2]

    # fontsize = 11.5

    methods_list = list(set(methods))
    num_methods = len(methods_list)

    colors = sns.color_palette(n_colors=num_methods)
    # markers = ["o", "s", "p", "P", "X", "*", "h", "D", "v", "^", ">", "<", "x", "H"][:num_methods]

    color_kv = dict(zip(methods_list, colors))
    # marker_kv = dict(zip(methods_list, markers))

    fig = plt.figure(figsize=(10, 15))
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.3)
    axs = np.empty([len(metrics), len(ways)], dtype=object)

    for col, way in enumerate(ways):
        df = df_all[way]

        color = [color_kv[name] for name in methods]
        # marker = [marker_kv[name] for name in methods]
        marker = None

        for row, metric in enumerate(metrics):
            # print(metric, row, col)
            df_metric = df[metric]
            ax1 = plt.subplot2grid((len(metrics), len(ways)), (row, col))
            axs[row, col] = ax1
            draw(df_metric, ax1, color, marker, metric)

    ax_legend = axs[0][1]
    lines, labels = ax_legend.get_legend_handles_labels()
    dict_label = dict(zip(labels, lines))
    dict_label = OrderedDict(sorted(dict_label.items(), key=lambda x: x[0]))

    ax_legend.legend(handles=dict_label.values(), labels=dict_label.keys(), ncol=10,
                     loc='lower left', columnspacing=0.7,
                     bbox_to_anchor=(-0.20, 1.24), fontsize=10.5)

    fig.savefig(os.path.join(save_fig_dir, savename + '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)


def get_top2_methods(col, is_largest):
    if is_largest:
        top2_name = col.nlargest(2).index.tolist()
    else:
        top2_name = col.nsmallest(2).index.tolist()
    name1, name2 = top2_name[0], top2_name[1]
    return name1, name2


def handle_one_col(df_metric, final_rate, is_largest):
    length = len(df_metric)
    res_start = int((1 - final_rate) * length)
    mean = df_metric[res_start:].mean()
    std = df_metric[res_start:].std()

    # mean.nlargest(2).index[1]
    res_latex = pd.Series(map(lambda mean, std: f"${mean:.4f}\pm {std:.4f}$", mean, std),
                          index=mean.index)
    res_excel = pd.Series(map(lambda mean, std: f"{mean:.4f}+{std:.4f}", mean, std),
                          index=mean.index)
    res_avg = mean

    name1, name2 = get_top2_methods(mean, is_largest=is_largest)
    res_latex.loc[name1] = r"$\mathbf{" + r"{}".format(res_latex.loc[name1][1:-1]) + r"}$"
    res_latex.loc[name2] = r"\underline{" + res_latex.loc[name2] + r"}"

    return res_latex, res_excel, res_avg


def get_name_order(methods):
    pat = re.compile(r"-?\d+\.?\d*e??\d*?")
    method_number = {}
    for method in methods:
        res = pat.findall(method)
        method_number[method] = list(map(float, res))
    sorted_methods = sorted(method_number.items(), key=lambda x: (len(x[1]), x[1]))
    methods = [pair[0] for pair in sorted_methods]

    # methods.remove("CIRS")
    # methods.remove("CIRS w/o CI")
    # methods = methods + ["CIRS", "CIRS w/o CI"]
    methods_order = dict(zip(methods, list(range(len(methods)))))
    return methods_order


def handle_table(df_all, save_fig_dir, savename="all_results", final_rate=1):
    df_all.rename(columns={r"FB": r"Standard", "NX_0_": r"No Overlapping", "NX_10_": r"No Overlapping for 10 turns"},
                  level=0, inplace=True)
    df_all.rename(columns={"CV_turn": r"$\text{CV}_\text{M}$", "len_tra": "Length"}, level=1,
                  inplace=True)

    ways = df_all.columns.levels[0][::-1]
    metrics = df_all.columns.levels[1]
    methods = df_all.columns.levels[2].to_list()

    methods_order = get_name_order(methods)

    df_latex = pd.DataFrame(columns=pd.MultiIndex.from_product([ways, metrics], names=["ways", "metrics"]))
    df_excel = pd.DataFrame(columns=pd.MultiIndex.from_product([ways, metrics], names=["ways", "metrics"]))
    df_avg = pd.DataFrame(columns=pd.MultiIndex.from_product([ways, metrics], names=["ways", "metrics"]))

    for col, way in enumerate(ways):
        df = df_all[way]
        for row, metric in enumerate(metrics):
            df_metric = df[metric]
            is_largest = False if metric == "MCD" else True
            df_latex[way, metric], df_excel[way, metric], df_avg[way, metric] = handle_one_col(df_metric, final_rate,
                                                                                               is_largest=is_largest)

    df_latex.sort_index(key=lambda index: [methods_order[x] for x in index.to_list()], inplace=True)
    df_excel.sort_index(key=lambda index: [methods_order[x] for x in index.to_list()], inplace=True)
    df_avg.sort_index(key=lambda index: [methods_order[x] for x in index.to_list()], inplace=True)

    # print(df_latex.to_markdown())
    # excel_path = os.path.join(save_fig_dir, savename + '.xlsx')
    # df_excel.to_excel(excel_path)

    return df_latex, df_excel, df_avg


def group(df_avg):
    ways = df_avg.columns.levels[0]
    metrics = df_avg.columns.levels[1]

    df_mean = pd.DataFrame(columns=pd.MultiIndex.from_product([ways, metrics], names=["ways", "metrics"]))
    methods = df_avg.index
    methods_set = set(methods)

    pattern = re.compile("var(\d+)_ent(\d+)")

    while methods_set:
        name = methods_set.pop()
        methods_set.add(name)
        res = pattern.match(name)
        if res:
            ent = res.group(2)
            pattern_ent = re.compile(f"var(\d+)_ent{ent}")
            a_group = [method for method in methods_set if pattern_ent.match(method)]
            a_row = df_avg.loc[a_group].mean()
            df_mean.loc[f"var000_ent{ent}"] = a_row
            for method in a_group:
                methods_set.remove(method)

        else:  # if len(a_group) == 0:
            pattern_var = re.compile(f"{name}$")
            a_group = [method for method in methods_set if pattern_var.match(method)]
            a_row = df_avg.loc[a_group].mean()
            df_mean.loc[name] = a_row
            for method in a_group:
                methods_set.remove(method)

    methods = df_mean.index
    methods_order = get_name_order(methods)
    df_mean.sort_index(key=lambda index: [methods_order[x] for x in index.to_list()], inplace=True)
    return df_mean


def visual_bar_mix(df_avg, save_fig_dir, savename, is_group=True):
    df_avg.rename(columns={r"$\text{CV}_\text{M}$": "CV_M"}, level=1, inplace=True)

    fig = plt.figure(figsize=(40, 20))
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.4)

    ways = df_avg.columns.levels[0]
    metrics = df_avg.columns.levels[1]

    if is_group:
        df_draw = group(df_avg)
    else:
        df_draw = df_avg

    axs = np.empty([len(metrics), len(ways)], dtype=object)

    for col, way in enumerate(ways):
        for row, metric in enumerate(metrics):
            # df_metric = df[metric]
            ax = plt.subplot2grid((len(metrics), len(ways)), (row, col))
            axs[row, col] = ax
            # draw(df_metric, ax1, color, marker, metric)

            sns.barplot(data=df_draw, x=df_draw.index, y=df_draw[(way, metric)], ax=ax, palette=sns.color_palette())
            plt.xticks(rotation=90, fontsize=6)
    plt.savefig(os.path.join(save_fig_dir, f"bar_{savename}{'_mix' if is_group else ''}.pdf"), format='pdf',
                bbox_inches='tight')
    plt.close()
    a = 1


def visual_one_group():
    realpath = os.path.dirname(__file__)
    save_fig_dir = os.path.join(realpath, "figures")

    create_dirs = [save_fig_dir]
    create_dir(create_dirs)

    datasets = ["kuairec_2023 2022", "kuairec_2023", "kuairand_l8"]
    datasets = ["kuairec_l8"]
    datasets = ["kuairec_ez"]
    datasets = ["kuairec_tan"]
    datasets = ["kuairand_lab5"]
    datasets = ["kuairec_ez1", "kuairec_ez2"]
    datasets = ["kuairand_l8"]
    datasets = ["kuairand51"]
    patterns = ["\[l15_0_var"]; datasets = ["kuairec_15_0"]
    patterns = ["\[l10_0_var"]; datasets = ["kuairec_10_0"]
    patterns = ["\[l10_1_var", "\[l10_2_var"]; datasets = ["kuairec_10_2", "kuairec_10_1"]
    patterns = ["\[l10_1_var", "\[l10_2_var"]; datasets = ["kuairec_10_1"]
    patterns = ["\[l15_1_var"]; datasets = ["kuairand_15_1"]
    patterns = ["\[l12_0_var"]; datasets = ["kuairand_12_0"]

    rename_dataset(datasets, patterns)

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
        df_latex, df_excel, df_avg = handle_table(df_all, save_fig_dir, savename=savename)

        # please install openpyxl if you want to write to an excel file.
        excel_path = os.path.join(save_fig_dir, savename + '.xlsx')
        df_excel.to_excel(excel_path)
        print("Producing the figure...")
        # visual(df_all, save_fig_dir, savename=savename)

        visual_bar_mix(df_avg, save_fig_dir, savename, is_group=False)
        visual_bar_mix(df_avg, save_fig_dir, savename, is_group=True)
        print(f"All results of {dataset} rendered done!")


if __name__ == '__main__':
    visual_one_group()
