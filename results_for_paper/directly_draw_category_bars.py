# -*- coding: utf-8 -*-
# @Time    : 2023/1/28 21:29
# @Author  : Chongming GAO
# @FileName: directly_draw_category_bars.py
import json
import os
import re

import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

from results_for_paper.get_discrepency import get_percentage


def transfer(recommended_feats, df_deepFM, sorted_index):
    # recommended_feats
    cat_set = sorted_index
    res_predicted = get_percentage(recommended_feats, cat_set, is_test=False, sorted_idx=cat_set)
    newline = pd.DataFrame({str(k): [v] for k, v in zip(range(1, 47), res_predicted)}, index=[3])
    newline["domain"] = "Recommendations"
    df = pd.concat([df_deepFM, newline], axis=0)
    return df


def draw(df, message, epoch, fig_path, prefix):
    colors = sns.color_palette("muted", n_colors=num)
    colors = sns.color_palette("Set2", n_colors=num)
    colors = sns.color_palette("Paired", n_colors=num * 2 + 12)
    colors1 = sns.color_palette("pastel", n_colors=num)
    colors2 = sns.color_palette(n_colors=num)

    fig = plt.figure(figsize=(5, 1))
    hatchs = ["////", "\\\\\\\\"]

    df.loc[2, "domain"] = "User model rated"

    for i in range(num, 0, -1):
        # print(i)
        # sns.barplot(x=i, y="domain", data=df, color=colors[2 * i + 10], hatch=hatchs[i%len(hatchs)], edgecolor=colors[2 * i - 1], lw=0.5)
        # plt.rcParams['hatch.color'] = colors[2 * i - 1 + 10]

        sns.barplot(x=str(i), y="domain", data=df, color=colors1[i - 1], hatch=hatchs[i % len(hatchs)],
                    edgecolor=colors2[i - 1], lw=0.4)

    # plt.xlabel(featname)
    plt.xlabel(None)
    plt.ylabel(None)
    # plt.axis("off")
    plt.xticks([])
    # dirpath = os.path.join("results_for_paper", "figures", "recommended_category_kuairand")
    pdfpath = os.path.join(fig_path, f'feat_KuaiRand_{prefix}_{message}_e{epoch}.pdf')
    # csv_path = os.path.join(dirpath, f'feat_KuaiRand_{message}_e{epoch}.csv')

    plt.savefig(pdfpath, bbox_inches='tight', pad_inches=0)
    # df.to_csv(csv_path, index=False)
    # plt.savefig(os.path.join(CODEPATH, f'feat_{envname}_{featname}.png'), bbox_inches='tight',
    #             pad_inches=0)
    # plt.show()
    plt.close()


def load_log_draw_feat(dirpath, fig_path, filename, sorted_index, df_deepFM):
    pattern_epoch = re.compile("Epoch: \[(\d+)]")
    pattern_info = re.compile("Info: \[(\{.+\})]")
    # pattern_message = re.compile('"message": "(.+)"')
    pattern_array = re.compile("array\((.+?)\)")

    pattern_tau = re.compile('"tau": (.+),')
    pattern_read = re.compile('"read_message": "(.+)"')
    pattern_message = re.compile("\[(.+?)]")

    res_message = re.search(pattern_message, filename)
    message = res_message.group(1)

    # df = pd.DataFrame()
    # message = "None"
    filepath = os.path.join(dirpath, filename)
    with open(filepath, "r") as file:
        lines = file.readlines()
        start = False
        add = 0
        info_extra = {'tau': 0, 'read': ""}
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
                if epoch < 190:
                    continue

                info = re.search(pattern_info, line)
                try:
                    info1 = info.group(1).replace("\'", "\"")
                except Exception as e:
                    print("jump incomplete line: [{}]".format(line))
                    continue
                info2 = re.sub(pattern_array, lambda x: x.group(1), info1)

                data = json.loads(info2)
                for prefix in ["", "NX_0_", f"NX_10_"]:
                    recommended_feats = data[prefix + "all_feats"]
                    df = transfer(recommended_feats, df_deepFM, sorted_index)
                    draw(df, message, epoch, fig_path, prefix if len(prefix) else "FB")


fig_dirpath = os.path.join("figures", "recommended_category_kuairand")
res_dirpath = os.path.join("figures", "recommended_category_kuairand")

df_deepFM = pd.read_csv(os.path.join(fig_dirpath, "feat_KuaiRand_s1_user10000_top_10_e5.csv"))
num = 46

sorted_index = pd.read_csv("majority_indices.csv").to_numpy().squeeze()

filenames = [
    "[bar_ent_10]_2023_01_30-14_20_55.log",
    "[bar_ent_0.1]_2023_01_30-14_20_55.log",
    "[bar_ent_0]_2023_01_30-14_20_55.log",
    "[bar_ent_1]_2023_01_30-14_20_55.log",
]

filenames = [
    "[leave3_var2_ent_5]_2023_01_30-17_12_11.log",
    "[leave3_ent_0.5]_2023_01_30-17_07_12.log",
    "[leave2_ent_0]_2023_01_30-17_07_11.log",
    "[leave2_ent_0.05]_2023_01_30-17_07_11.log",
    "[leave2_ent_0.5]_2023_01_30-17_07_11.log",
    "[leave2_ent_0.1]_2023_01_30-17_07_11.log",
    "[leave3_ent_0.05]_2023_01_30-17_07_12.log",
    "[leave2_ent_1]_2023_01_30-17_07_11.log",
    "[leave3_ent_0]_2023_01_30-17_07_12.log",
    "[leave3_ent_1]_2023_01_30-17_07_12.log",
    "[leave3_ent_0.1]_2023_01_30-17_07_12.log",
    "[leave2_ent_5]_2023_01_30-17_07_11.log",
    "[leave2_ent_10]_2023_01_30-17_07_11.log",
    "[leave3_ent_5]_2023_01_30-17_07_12.log",
    "[leave3_ent_10]_2023_01_30-17_07_12.log",
    "[leave3_var2_ent_1]_2023_01_30-17_12_11.log",
    "[leave2_ent_50]_2023_01_30-17_07_12.log",
    "[leave3_var2_ent_50]_2023_01_30-17_12_11.log",
    "[leave3_var2_ent_0]_2023_01_30-17_12_11.log",
    "[leave3_var2_ent_0.5]_2023_01_30-17_12_11.log",
    "[leave3_ent_50]_2023_01_30-17_07_12.log",
    "[leave3_var2_ent_10]_2023_01_30-17_12_11.log",
    "[leave3_var2_ent_0.1]_2023_01_30-17_12_11.log",
    "[leave3_var2_ent_0.05]_2023_01_30-17_12_11.log",
]

for filename in filenames:
    load_log_draw_feat(res_dirpath, fig_dirpath, filename, sorted_index, df_deepFM)
