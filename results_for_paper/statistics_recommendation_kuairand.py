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
from tqdm import tqdm



plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import sys

CODEPATH = os.path.dirname(__file__)
sys.path.extend(
    [f"{CODEPATH}/..", f"{CODEPATH}/../src", f"{CODEPATH}/../src/DeepCTR-Torch", f"{CODEPATH}/../src/tianshou"])

from run_worldModel_kuairand import get_df_train, get_args, load_static_validate_data_kuairand
import pandas as pd
from scipy.sparse import csr_matrix

from core.user_model_pairwise import UserModel_Pairwise
from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv, DATAPATH


# from run_A2CPolicy import get_args


def load_model(args):
    UM_SAVE_PATH = os.path.join(f"{CODEPATH}", "..", "saved_models", args.env, args.user_model_name)
    MODEL_MAT_PATH = os.path.join(UM_SAVE_PATH, "mats", f"[{args.read_message}]_mat.pickle")
    MODEL_PARAMS_PATH = os.path.join(UM_SAVE_PATH, "params", f"[{args.read_message}]_params.pickle")
    MODEL_PATH = os.path.join(UM_SAVE_PATH, "models", f"[{args.read_message}]_model.pt")
    MODEL_EMBEDDING_PATH = os.path.join(UM_SAVE_PATH, "embeddings", f"[{args.read_message}]_emb.pt")
    USER_EMBEDDING_PATH = os.path.join(UM_SAVE_PATH, "embeddings", f"[{args.read_message}]_emb_user.pt")
    ITEM_EMBEDDING_PATH = os.path.join(UM_SAVE_PATH, "embeddings", f"[{args.read_message}]_emb_item.pt")

    print(MODEL_PARAMS_PATH)
    with open(MODEL_PARAMS_PATH, "rb") as file:
        model_params = pickle.load(file)

    model_params["device"] = "cpu"
    user_model = UserModel_Pairwise(**model_params)
    user_model.load_state_dict(torch.load(MODEL_PATH))

    return user_model


def load_big():
    df_train, _, _, _ = get_df_train()

    # df_train = df_train[df_train["is_click"] > 0]

    df_pop = df_train[["user_id", "video_id"]].groupby("video_id").agg(len)
    df_pop.rename(columns={"user_id": "count"}, inplace=True)

    miss_id = list(set(range(7583)) - set(df_pop.index))
    df_miss = pd.DataFrame({"id": miss_id, "count": 0})
    df_miss.set_index("id", inplace=True)

    df_pop = df_pop.append(df_miss)

    ind = df_pop["count"].argsort()
    ind = ind[::-1]
    df_pop = df_pop.iloc[ind]

    # df_small_pop = pd.DataFrame(df_pop)
    # df_small_pop = df_small_pop[df_small_pop.index.isin(lbe_video.classes_)]

    df_pop["item_id_sorted"] = range(len(df_pop))
    df_pop["item_id"] = df_pop.index
    # df_pop["item_id"] = lbe_video.transform(df_pop.index)

    df_pop.index.name = "item_id"
    df_pop.index = df_pop["item_id"]
    df_pop = df_pop[["count", "item_id_sorted"]]

    df_train_y = df_train[["video_id", "is_like", "is_click", "long_view", "watch_ratio"]].groupby("video_id").agg(np.mean)
    df_pop = df_pop.join(df_train_y, on=['item_id'], how="left")

    return df_pop


def get_recommended_items(args_um, user_model):
    user_features = ["user_id", 'user_active_degree', 'is_live_streamer', 'is_video_author', 'follow_user_num_range',
                     'fans_user_num_range', 'friend_user_num_range', 'register_days_range'] \
                    + [f'onehot_feat{x}' for x in range(18)]
    item_features = ["video_id"] + ["feat" + str(i) for i in range(3)] + ["duration_ms"]

    reward_features = ["is_click"]
    dataset_val = load_static_validate_data_kuairand(user_features, item_features, reward_features,
                                                     args_um.entity_dim, args_um.feature_dim, DATAPATH)

    df_y = pd.DataFrame({"item_id":dataset_val.x_numpy[:, 26], "y":dataset_val.y_numpy.squeeze()})
    df_y_mean = df_y.groupby("item_id").agg(np.mean)

    K = 10
    is_softmax = False
    epsilon = 0
    is_ucb = False

    # load user info
    df_user = KuaiRandEnv.load_user_info()
    # df_user = None # Todo!!!!!!!!!!!

    count = {i: 0 for i in range(len(dataset_val.df_item_env))}
    # for uesr_big_id in tqdm(range(dataset_val.x_columns[0].vocabulary_size)):
    for user in tqdm(range(100)):
        recommendation, reward_pred = user_model.recommend_k_item(user, dataset_val, k=K, is_softmax=is_softmax,
                                                                  epsilon=epsilon, is_ucb=is_ucb, df_user=df_user)
        for i in recommendation:
            count[i] += 1
    return count, df_y_mean


def visual(df_small_pop, count, df_y_mean, tau):
    df_visual = df_small_pop

    df_hit = pd.DataFrame(count.items(), columns=["item_id", "hit"])
    df_hit = df_hit.set_index("item_id")

    df_visual = df_visual.join(df_hit, on=['item_id'], how="left")
    df_visual = df_visual.join(df_y_mean, on=['item_id'], how="left")

    # sep = [500, 1000, 1500, 2000, 5000, 10000, max(df_visual["count"]) + 1]
    sep = list(range(0, 1000, 100)) + [df_visual['count'].max() + 1]

    group_info = {"group": [], "count": [], "y": [], "hit": [], "train_like":[], "train_view":[], "train_click":[], "watch_ratio":[]}
    for left, right in zip(sep[:-1], sep[1:]):
        df_group = df_visual[df_visual['count'].map(lambda x: x >= left and x < right)]
        res = df_group["hit"].sum()
        # res2 = (df_group["y"] * df_group["count"]).sum() / df_group["count"].sum()
        res2 = df_group["y"].mean()


        group_info["group"].append(f"[{left},{right})")
        group_info["count"].append(len(df_group))
        group_info["hit"].append(res)
        group_info["y"].append(res2)
        group_info["train_like"].append(df_group["is_like"].mean())
        group_info["train_view"].append(df_group["long_view"].mean())
        group_info["train_click"].append(df_group["is_click"].mean())
        group_info["watch_ratio"].append(df_group["watch_ratio"].mean())

    df_groups = pd.DataFrame(group_info)

    fig = plt.figure(figsize=(7, 3.5))
    sns.barplot(data=df_groups, x="group", y="count")
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    sns.lineplot(data=df_groups, x="group", y="watch_ratio", ax=ax2)
    fig.savefig(f"label_watch.pdf", format='pdf', bbox_inches='tight')

    fig = plt.figure(figsize=(7, 3.5))
    sns.barplot(data=df_groups, x="group", y="count")
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    sns.lineplot(data=df_groups, x="group", y="train_like", ax=ax2)
    fig.savefig(f"label_like.pdf", format='pdf', bbox_inches='tight')

    fig = plt.figure(figsize=(7, 3.5))
    sns.barplot(data=df_groups, x="group", y="count")
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    sns.lineplot(data=df_groups, x="group", y="train_click", ax=ax2)
    fig.savefig(f"label_click.pdf", format='pdf', bbox_inches='tight')

    fig = plt.figure(figsize=(7, 3.5))
    sns.barplot(data=df_groups, x="group", y="count")
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    sns.lineplot(data=df_groups, x="group", y="train_view", ax=ax2)
    fig.savefig(f"label_longview.pdf", format='pdf', bbox_inches='tight')

    fig = plt.figure(figsize=(7, 3.5))
    sns.barplot(data=df_groups, x="group", y="count")
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    sns.lineplot(data=df_groups, x="group", y="hit", ax=ax2)
    fig.savefig(f"visual_{tau}.pdf", format='pdf', bbox_inches='tight')
    plt.show()
    a = 1

    # sns.displot(df_visual.iloc[1:], x="count", bins=100)
    # plt.show()

    # sns.lineplot(data=df_visual.iloc[1:], x="item_id_sorted", y="count")
    # ax1 = plt.gca()
    # ax2 = ax1.twinx()
    # sns.lineplot(data=df_visual.iloc[1:], x="item_id_sorted", y="hit", ax=ax2)
    # # ax2.set_ylim(0,1)
    # fig.savefig("small_mat.pdf", format='pdf', bbox_inches='tight')
    # plt.show()

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

    # return ind


args = get_args()

for tau in [0]:
    args.tau = tau
    args.read_message = f"sgd 0.1 long_view"
    print(args.read_message)

    user_model = load_model(args)
    df_small_pop = load_big()
    count, df_y_mean = get_recommended_items(args, user_model)
    visual(df_small_pop, count, df_y_mean, args.read_message)
