# -*- coding: utf-8 -*-
# @Time    : 2022/9/30 20:04
# @Author  : Chongming GAO
# @FileName: show_bad_popularity_coat.py

# python run_worldModel_coat.py --cuda 0 --loss "pp" --message "pp" &
# python run_worldModel_coat.py --cuda 0 --loss "pair" --message "pair" &
# python run_worldModel_coat.py --cuda 0 --loss "point" --message "point" &
# python run_worldModel_coat.py --cuda 0 --loss "pointneg" --message "pointneg" &

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.neighbors import KernelDensity
from sklearn import preprocessing

import matplotlib
import torch
from numba import njit
from tqdm import tqdm
from multiprocessing import Pool

# matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

CODEPATH = os.path.dirname(__file__)
ROOTPATH = os.path.dirname(CODEPATH)
sys.path.extend([ROOTPATH])

from environments.coat.env.Coat import CoatEnv
from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
from environments.KuaiRec.env.KuaiEnv import KuaiEnv
from environments.YahooR3.env.Yahoo import YahooEnv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CoatEnv-v0")
    parser.add_argument("--user_model_name", type=str, default="DeepFM")
    parser.add_argument("--read_message", type=str, default="point")

    args = parser.parse_known_args()[0]
    return args


# @njit
def get_exposure_times(array, ans):
    last = np.array([-1, -1])
    cnt = 1
    for i in tqdm(range(len(array)), desc="counting users..."):
        # for i in range(len(array)):
        if all(array[i] == last):
            cnt += 1
        else:
            cnt = 1
            last = array[i]
        ans[i - cnt + 1:i + 1] = cnt
    return ans


def visual(df_data, df_pop, envname, lossname, field_train):
    fig = plt.figure(figsize=(24, 14))
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.3)
    axs = []

    def sort_fun(str):
        return float(str.strip("()").split(",")[0])

    ax1 = plt.subplot2grid((3, 4), (0, 0))
    sns.histplot(data=df_pop, x="count", bins=100, ax=ax1)

    ax2 = plt.subplot2grid((3, 4), (0, 1))
    sns.countplot(data=df_data, x="pop_group", order=sorted(df_data["pop_group"].unique(), key=sort_fun), ax=ax2)

    ax3 = plt.subplot2grid((3, 4), (0, 2))
    sns.histplot(data=df_data, x="kde", bins=100, ax=ax3)

    ax4 = plt.subplot2grid((3, 4), (0, 3))
    sns.countplot(data=df_data, x="kde_group", order=sorted(df_data["kde_group"].unique(), key=sort_fun), ax=ax4)

    # ax5 = plt.subplot2grid((3, 4), (1, 0))
    # sns.scatterplot(data=df_data, x="item_pop", y="kde", ax=ax5)

    ax6 = plt.subplot2grid((3, 4), (1, 1))
    sns.boxplot(data=df_data, x="kde_group", y="item_pop", order=sorted(df_data["kde_group"].unique(), key=sort_fun),
                ax=ax6)

    ax7 = plt.subplot2grid((3, 4), (1, 2))
    sns.boxplot(data=df_data, x="pop_group", y="kde", order=sorted(df_data["pop_group"].unique(), key=sort_fun), ax=ax7)

    ax8 = plt.subplot2grid((3, 4), (1, 3))
    sns.barplot(data=df_data, x="kde_group", y="error", hue="pop_group", ax=ax8,
                order=sorted(df_data["kde_group"].unique(), key=sort_fun),
                hue_order=sorted(df_data["pop_group"].unique(), key=sort_fun))

    # ax9 = plt.subplot2grid((3, 4), (2, 0))
    # sns.lineplot(data=df_data, x="kde", y="error", ax=ax9)

    ax10 = plt.subplot2grid((3, 4), (2, 1))
    sns.boxplot(data=df_data, x="kde_group", y="error", order=sorted(df_data["kde_group"].unique(), key=sort_fun),
                ax=ax10)

    # ax11 = plt.subplot2grid((3, 4), (2, 2))
    # sns.lineplot(data=df_data, x="item_pop", y="error", ax=ax11)

    ax12 = plt.subplot2grid((3, 4), (2, 3))
    sns.boxplot(data=df_data, x="pop_group", y="error", order=sorted(df_data["pop_group"].unique(), key=sort_fun),
                ax=ax12)

    plt.savefig(os.path.join(CODEPATH, f'all_{envname}_{lossname}_{field_train}.pdf'), bbox_inches='tight',
                pad_inches=0)
    plt.close(fig)


def draw(df_data, predicted_mat, df_pop, envname, lossname, is_train, df_frequency=None):
    df_data["y_pred"] = predicted_mat[df_data["user_id"], df_data["item_id"]] * df_data[yname].max()
    df_data["item_pop"] = df_pop["count"].loc[df_data["item_id"]].reset_index(drop=True)
    df_data["pop_group"] = df_pop["pop_group"].loc[df_data["item_id"]].reset_index(drop=True)
    df_data["error"] = np.abs(df_data["y_pred"] - df_data[yname])

    field_train = "train" if is_train else "test"

    # %% group kde
    kde_list = [df_data["kde"].min()] + [0, 0.2, 0.4, 0.6, 0.8] + [df_data["kde"].max()]
    for left, right in zip(kde_list[:-1], kde_list[1:]):
        df_data.loc[df_data["kde"] >= left, "kde_group"] = f"({left},{right})"

    visual(df_data, df_pop, envname, lossname, field_train)

    return df_frequency if is_train else None


@njit
def get_kde(x, data_array, res, bandwidth=0.1):
    # def gauss(x):
    #     return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * (x ** 2))
    # def gauss(x):
    #     return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x ** 2).sum(1, keepdims=True))

    N = len(data_array)

    constant = 1 / np.sqrt(2 * np.pi)
    for i in range(len(data_array)):
        temp = (x - data_array[i]) / bandwidth
        ans = constant * np.exp(-0.5 * (temp ** 2).sum(1))
        res += ans
    res /= (N * bandwidth)


def get_pop(df_train, popbin):
    df_popularity = df_train[["item_id", "user_id"]].groupby("item_id").agg(len)
    miss_id = list(set(range(df_train["item_id"].max() + 1)) - set(df_popularity.index))
    df_miss = pd.DataFrame({"id": miss_id, "user_id": 0})
    df_miss.set_index("id", inplace=True)
    df_pop = df_popularity.append(df_miss)
    df_pop = df_pop.sort_index()
    df_pop.rename(columns={"user_id": "count"}, inplace=True)

    # # for feat in df_train.columns[3:]:
    # #     df_feat_pop = df_train[[feat, "user_id"]].groupby(feat).agg(len)
    # #     print(df_feat_pop)
    #
    # feat = "age" # todo: for coat
    # df_feat_pop = df_train[[feat, "user_id"]].groupby(feat).agg(len)
    # print(df_feat_pop)

    # df_pop = df_pop.sort_values(by="count").reset_index(drop=True)

    bins = popbin + [df_pop.max()[0] + 1]
    pop_res = {}
    for left, right in zip(bins[:-1], bins[1:]):
        df_pop.loc[df_pop["count"].map(lambda x: x >= left and x < right), "pop_group"] = f"({left},{right})"
        # df_pop.loc[df_pop["count"].map(lambda x: x >= left and x < right), "pop_group"] = left
        pop_res[f"({left},{right})"] = sum(df_pop["count"].map(lambda x: x >= left and x < right))

    print(pop_res)

    sns.histplot(data=df_pop, x="count", bins=100)

    sns.histplot(data=df_pop["count"], bins=100)
    plt.show()
    plt.close()

    plt.savefig(os.path.join(CODEPATH, f'dist_pop_{envname}.pdf'), bbox_inches='tight', pad_inches=0)
    plt.close()

    return df_pop


def get_kde_all(df_train, df_test, train_emb, test_emb):
    # a = np.vstack([train_emb, np.expand_dims(train_emb[0, :], 0).repeat(1, axis=0)])

    bandwidth = 1.05 * np.std(train_emb) * (len(train_emb) ** (-1 / 5)) * train_emb.shape[1]
    # res = np.zeros([len(train_emb)])
    # t = time.time()
    # res_self = get_kde(train_emb, train_emb, res, bandwidth)
    # print(time.time() - t)
    # scaler = preprocessing.MinMaxScaler()
    # bb_self = scaler.fit_transform(np.exp(res_self).reshape(-1, 1))
    #
    # input_array = np.random.randn(2000)
    # bandwidth = 1.05 * np.std(input_array) * (len(input_array) ** (-1 / 5))
    # x_array = np.linspace(min(input_array), max(input_array), 50)
    # x_array = input_array
    # y_array = [get_kde(x_array, input_array, bandwidth) for i in range(x_array.shape[0])]

    # kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(input_array.reshape(-1,1))
    # res = np.exp(kde.score_samples(x_array.reshape(-1,1)))

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(train_emb)
    # t = time.time()
    # res = kde.score_samples(train_emb)
    # res_test = kde.score_samples(test_emb)
    # print(time.time() - t)

    t = time.time()
    n_cpu = os.cpu_count() - 2
    with Pool(n_cpu) as p:
        value_train_list = p.map(kde.score_samples, np.array_split(train_emb, n_cpu))
        value_test_list = p.map(kde.score_samples, np.array_split(test_emb, n_cpu))
    print("time", time.time() - t)

    kde_train = np.concatenate(value_train_list)
    kde_test = np.concatenate(value_test_list)

    # scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

    kde_train_reform = np.exp(kde_train).reshape(-1, 1)
    kde_test_reform = np.exp(kde_test).reshape(-1, 1)

    maxx = max(kde_train_reform)
    minn = min(kde_train_reform)

    kde_normed_train = (kde_train_reform - minn) / (maxx - minn)
    kde_mormed_test = (kde_test_reform - minn) / (maxx - minn)

    df_train["kde"] = kde_normed_train
    df_test["kde"] = kde_mormed_test

    sns.histplot(kde_normed_train)
    sns.histplot(kde_mormed_test)
    plt.savefig(os.path.join(CODEPATH, f'kde_{envname}_{lossname}.pdf'), bbox_inches='tight', pad_inches=0)
    plt.close()

    return df_train, df_test


def main(args, df_train, df_test, envname, lossname, yname, popbin):
    UM_SAVE_PATH = os.path.join(ROOTPATH, "saved_models", envname, args.user_model_name)
    MODEL_MAT_PATH = os.path.join(UM_SAVE_PATH, "mats", f"[{lossname}]_mat.pickle")

    USER_EMBEDDING_PATH = os.path.join(UM_SAVE_PATH, "embeddings", f"[{lossname}]_emb_user.pt")
    ITEM_EMBEDDING_PATH = os.path.join(UM_SAVE_PATH, "embeddings", f"[{lossname}]_emb_item.pt")
    USER_VAL_EMBEDDING_PATH = os.path.join(UM_SAVE_PATH, "embeddings", f"[{lossname}]_emb_user_val.pt")
    ITEM_VAL_EMBEDDING_PATH = os.path.join(UM_SAVE_PATH, "embeddings", f"[{lossname}]_emb_item_val.pt")

    user_train_embedding = torch.load(USER_EMBEDDING_PATH).detach().numpy()
    item_train_embedding = torch.load(ITEM_EMBEDDING_PATH).detach().numpy()
    # user_val_embedding = torch.load(USER_VAL_EMBEDDING_PATH).detach().numpy()
    # item_val_embedding = torch.load(ITEM_VAL_EMBEDDING_PATH).detach().numpy()

    train_user_emb = user_train_embedding[df_train["user_id"]]
    train_item_emb = item_train_embedding[df_train["item_id"]]
    train_emb = np.concatenate([train_user_emb, train_item_emb], axis=-1)
    test_user_emb = user_train_embedding[df_test["user_id"]]
    test_item_emb = item_train_embedding[df_test["item_id"]]
    test_emb = np.concatenate([test_user_emb, test_item_emb], axis=-1)

    df_pop = get_pop(df_train, popbin)


    df_train, df_test = get_kde_all(df_train, df_test, train_emb, test_emb)

    with open(MODEL_MAT_PATH, "rb") as file:
        predicted_mat = pickle.load(file)

    df_frequency = draw(df_train, predicted_mat, df_pop, envname, lossname, is_train=True)
    draw(df_test, predicted_mat, df_pop, envname, lossname, is_train=False, df_frequency=df_frequency)


def get_data(dataset):
    if dataset == "CoatEnv":
        df_train = CoatEnv.get_df_coat("train.ascii")[0]
        df_test = CoatEnv.get_df_coat("test.ascii")[0]
        env = "CoatEnv-v0"
        yname = "rating"
        popbin = [0, 10, 20, 40, 60, 80, 100, 200]

    if dataset == "YahooEnv":
        df_train = YahooEnv.get_df_yahoo("ydata-ymusic-rating-study-v1_0-train.txt")[0]
        df_test = YahooEnv.get_df_yahoo("ydata-ymusic-rating-study-v1_0-test.txt")[0]
        env = "YahooEnv-v0"
        yname = "rating"
        popbin = [0, 40, 60, 80, 100, 200]

    if dataset == "KuaiRandEnv":
        df_train = KuaiRandEnv.get_df_kuairand("train_processed.csv")[0]
        df_test = KuaiRandEnv.get_df_kuairand("test_processed.csv")[0]
        env = "KuaiRand-v0"
        yname = "is_click"
        popbin = [0, 10, 20, 40, 80, 150, 300]

    if dataset == "KuaiEnv":
        df_train = KuaiEnv.get_df_kuairec("big_matrix_processed.csv")[0]
        df_test = KuaiEnv.get_df_kuairec("small_matrix_processed.csv")[0]
        env = "KuaiEnv-v0"
        yname = "watch_ratio_normed"
        popbin = [0, 10, 20, 40, 60, 80, 100, 200]

    return df_train, df_test, env, yname, popbin


if __name__ == '__main__':
    datasets = [
        "CoatEnv",
        # "YahooEnv",
        # "KuaiRandEnv",
        # "KuaiEnv"
    ]

    losses = ["point", "pointneg", "pp", "pair"]
    # losses = ["2048-8", "4096-8", "10000-8"]
    # losses = ["10000-8"]
    # losses = ["point-10000-4",
    #           "point-10000-16",
    #           "point-10000-8",
    #           "pair-10000-4",
    #           "pair-10000-16",
    #           "pair-10000-8",
    #           "point-4096-4",
    #           "point-4096-16",
    #           "point-4096-8",
    #           "pair-4096-4",
    #           "pair-4096-16",
    #           "pair-4096-8",
    #           "point-2048-4",
    #           "point-2048-16",
    #           "point-2048-8",
    #           "pair-2048-4",
    #           "pair-2048-16",
    #           "pair-2048-8",
    #           "point-1024-16", "point-1024-16", "point-1024-16",
    #           "pair-1024-16", "pair-1024-16", "pair-1024-16",
    #           "pair-512-4", "pair-512-8", "pair-512-16",
    #           "point-512-4", "point-512-8", "point-512-16", ]

    args = get_args()
    for i in range(len(datasets)):
        df_train, df_test, envname, yname, popbin = get_data(datasets[i])
        for lossname in losses:
            if envname == "YahooEnv-v0":
                df_train = df_train.loc[df_train["user_id"] < 5400]
            main(args, df_train, df_test, envname, lossname, yname, popbin)
