# -*- coding: utf-8 -*-
# @Author  : Chongming GAO
# @FileName: statistics_recommendation.py

import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import run_worldModel
from core.util import load_static_validate_data_kuaishou



import pandas as pd
from scipy.sparse import csr_matrix

from core.user_model_pairwise import UserModel_Pairwise
from environments.KuaiRec.env.KuaiEnv import KuaiEnv


from run_A2CPolicy import get_args

args = get_args()

args.tau = 10000
args.read_message = "UM tau10000"


def load_model(args):

    UM_SAVE_PATH = os.path.join("..", "saved_models", args.env, args.user_model_name)
    MODEL_MAT_PATH = os.path.join(UM_SAVE_PATH, "mats", f"[{args.read_message}]_mat.pickle")
    MODEL_PARAMS_PATH = os.path.join(UM_SAVE_PATH, "params", f"[{args.read_message}]_params.pickle")
    MODEL_PATH = os.path.join(UM_SAVE_PATH, "models", f"[{args.read_message}]_model.pt")
    MODEL_EMBEDDING_PATH = os.path.join(UM_SAVE_PATH, "embeddings", f"[{args.read_message}]_emb.pt")
    USER_EMBEDDING_PATH = os.path.join(UM_SAVE_PATH, "embeddings", f"[{args.read_message}]_emb_user.pt")
    ITEM_EMBEDDING_PATH = os.path.join(UM_SAVE_PATH, "embeddings", f"[{args.read_message}]_emb_item.pt")

    with open(MODEL_PARAMS_PATH, "rb") as file:
        model_params = pickle.load(file)

    model_params["device"] = "cpu"
    user_model = UserModel_Pairwise(**model_params)
    user_model.load_state_dict(torch.load(MODEL_PATH))

    return user_model

mat, lbe_user, lbe_video, list_feat, df_video_env, df_dist_small = KuaiEnv.load_mat()

def load_big():
    CODEPATH = os.path.dirname(__file__)
    DATAPATH = os.path.join(CODEPATH, "..", "environments", "KuaiRec", "data")
    filename = os.path.join(DATAPATH, "big_matrix.csv")
    df_big = pd.read_csv(filename, usecols=['user_id', 'video_id', 'timestamp', 'watch_ratio', 'video_duration'])
    df_big['video_duration'] /= 1000

    # load feature info
    list_feat, df_feat = KuaiEnv.load_category()

    df_big = df_big.join(df_feat, on=['video_id'], how="left")
    df_big.loc[df_big['watch_ratio'] > 5, 'watch_ratio'] = 5

    df_pop = df_big[["user_id","video_id"]].groupby("video_id").agg(len)
    df_pop.rename(columns={"user_id":"count"},inplace=True)

    ind = df_pop["count"].argsort()
    ind = ind[::-1]
    df_pop = df_pop["count"][ind]

    df_small_pop = pd.DataFrame(df_pop)
    df_small_pop = df_small_pop[df_small_pop.index.isin(lbe_video.classes_)]

    df_small_pop["item_id_sorted"] = range(len(df_small_pop))
    df_small_pop["item_id_raw"] = df_small_pop.index
    df_small_pop["item_id"] = lbe_video.transform(df_small_pop.index)

    df_small_pop.index.name = "item_id"
    df_small_pop.index = df_small_pop["item_id"]
    df_small_pop = df_small_pop[["count","item_id_sorted", "item_id_raw"]]

    return df_small_pop

def get_recommended_items(user_model):
    CODEPATH = os.path.dirname(__file__)
    DATAPATH = os.path.join(CODEPATH, "..", "environments", "KuaiRec", "data")
    user_features = ["user_id"]
    item_features = ["video_id"] + ["feat" + str(i) for i in range(4)] + ["video_duration"]
    reward_features = ["watch_ratio"]
    args_um = run_worldModel.get_args()
    dataset_val = load_static_validate_data_kuaishou(user_features, item_features, reward_features,
                                                         args_um.entity_dim, args_um.feature_dim, DATAPATH)

    K = 10
    is_softmax = True
    epsilon = 0
    is_ucb = False

    count = {i:0 for i in range(len(lbe_video.classes_))}
    for uesr_big_id in lbe_user.classes_:
        recommendation, reward_pred = user_model.recommend_k_item(uesr_big_id, dataset_val, k=K, is_softmax=is_softmax, epsilon=epsilon, is_ucb=is_ucb)
        for i in lbe_video.transform(recommendation):
            count[i] += 1
    return count

def visual(mat, df_small_pop, count):
    mean_ratio = mat.mean(axis=0)
    ind = mean_ratio.argsort()
    ind = ind[::-1]
    sorted_ratio = mean_ratio[ind]

    df = pd.DataFrame(sorted_ratio, columns=["watch_ratio"])
    df["item_id"] = ind
    df["item_id_sorted"] = range(len(df))
    df = df.set_index("item_id")

    df_visual = df_small_pop.join(df[['watch_ratio']], on=['item_id'], how="left")

    df_hit = pd.DataFrame(count.items(), columns=["item_id", "hit"])
    df_hit = df_hit.set_index("item_id")

    df_visual = df_visual.join(df_hit, on=['item_id'], how="left")

    sep = [0, 10, 100, 500, 1000, 2000, 5000, 10000, max(df_visual["count"]) + 1]

    group_info = {"group":[], "count":[], "watch_ratio":[], "hit":[]}
    for left, right in zip(sep[:-1], sep[1:]):

        df_group = df_visual[df_visual['count'].map(lambda x: x>=left and x < right)]
        res = df_group[["count", "hit"]].sum()
        res2 = (df_group["watch_ratio"] * df_group["count"]).sum()/res["count"]

        group_info["group"].append(f"[{left},{right})")
        group_info["count"].append(res["count"])
        group_info["hit"].append(res["hit"])
        group_info["watch_ratio"].append(res2)

    df_groups = pd.DataFrame(group_info)

    fig = plt.figure(figsize = (7, 3.5))
    sns.displot(df_visual.iloc[1:], x="count",bins=100)
    plt.show()

    sns.lineplot(data=df_visual.iloc[1:], x="item_id_sorted", y="count")
    ax1 = plt.gca()

    ax2 = ax1.twinx()
    sns.lineplot(data=df_visual.iloc[1:], x="item_id_sorted", y="watch_ratio", ax=ax2)
    ax2.set_ylim(-4, 6)

    fig.savefig("small_mat.pdf", format='pdf', bbox_inches='tight')
    plt.show()



    # fig = plt.figure(figsize = (7, 3.5))
    # sns.lineplot(data=df_visual.iloc[1:], x="item_id_sorted", y="count")
    # ax1 = plt.gca()
    #
    # ax2 = ax1.twinx()
    # sns.lineplot(data=df_visual.iloc[1:], x="item_id_sorted", y="watch_ratio", ax=ax2)
    # ax2.set_ylim(-4, 6)
    #
    # fig.savefig("small_mat.pdf", format='pdf', bbox_inches='tight')
    # plt.show()


    # sorted_mat = mat[:, ind]
    # mydict = {"user_id":[], "item_id":[], "item_id_sorted":[], "watch_ratio":[]}
    # for i in range(len(sorted_mat)):
    #     for j in range(len(sorted_mat[0])):
    #         mydict["user_id"].append(i)
    #         mydict["item_id"].append(ind[j])
    #         mydict["item_id_sorted"].append(j)
    #         mydict["watch_ratio"].append(sorted_mat[i,j])
    #
    # df_mat = pd.DataFrame(mydict)
    #
    # sns.lineplot(data=df_mat, x="item_id_sorted", y="watch_ratio")

    # sns.lineplot(data=df_mat)

    return ind

user_model = load_model(args)
count = get_recommended_items(user_model)
df_small_pop = load_big()
visual(mat, df_small_pop, count)
