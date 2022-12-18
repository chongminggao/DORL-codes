import argparse
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns

import torch

# matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

CODEPATH = os.path.dirname(__file__)
ROOTPATH = os.path.dirname(CODEPATH)
sys.path.extend([ROOTPATH])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CoatEnv-v0")
    parser.add_argument("--user_model_name", type=str, default="DeepFM")
    parser.add_argument("--read_message", type=str, default="point")

    args = parser.parse_known_args()[0]
    return args


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

    field_train = "Standard" if is_train else "Random"

    # %% group kde
    kde_list = [df_data["kde"].min()] + [0, 0.2, 0.4, 0.6, 0.8] + [df_data["kde"].max()]
    for left, right in zip(kde_list[:-1], kde_list[1:]):
        df_data.loc[df_data["kde"] >= left, "kde_group"] = f"({left},{right})"

    visual(df_data, df_pop, envname, lossname, field_train)

    return df_frequency if is_train else None


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
    plt.savefig(os.path.join(CODEPATH, f'dist_pop_{envname}.pdf'), bbox_inches='tight', pad_inches=0)
    plt.close()

    return df_pop


def get_percentage(cats, all_values, is_test=False, sorted_idx=None):
    res_cnt = Counter(cats)
    if is_test:
        sorted_cnt = [res_cnt[i] for i in sorted_idx]
    else:
        sorted_cnt = sorted(res_cnt.values(), reverse=True)
        sorted_cnt += (len(all_values) - len(sorted_cnt)) * [0]

    cumsum = np.cumsum(sorted_cnt)
    res = cumsum / cumsum[-1]

    if not is_test:
        sorted_idx = [x[0] for x in sorted(res_cnt.items(), key=lambda x: x[1], reverse=True)]
        sorted_idx += list(set(all_values) - set(sorted_idx))
        return res, sorted_idx
    else:
        return res


def draw(res_train, res_test, num, featname):
    df = pd.DataFrame({k + 1: [x, y] for k, (x, y) in enumerate(zip(res_train, res_test))})
    df["domain"] = ["Standard", "Random"]

    colors = sns.color_palette("muted", n_colors=num)
    fig = plt.figure(figsize=(6, 3))
    for i in range(num, 0, -1):
        # print(i)
        sns.barplot(x=i, y="domain", data=df, color=colors[i - 1])
    plt.xlabel(featname)
    plt.xticks([])
    plt.savefig(os.path.join(CODEPATH, f'feat_{envname}_{featname}.pdf'), bbox_inches='tight',
                pad_inches=0)
    plt.savefig(os.path.join(CODEPATH, f'feat_{envname}_{featname}.png'), bbox_inches='tight',
                pad_inches=0)
    plt.show()
    plt.close()


def get_distribution(threshold, item_features, multi_feat, envname):
    # if threshold > 0:
    feat_train = df_train.loc[df_train[yname] >= threshold, item_features[1:]]
    feat_test = df_test.loc[df_test[yname] >= threshold, item_features[1:]]
    # feat_train = df_train[item_features[1:]]
    # feat_test = df_test[item_features[1:]]

    feat_num = feat_train.nunique()

    if multi_feat:  # for kuairand and kuairec
        cats_train = feat_train.to_numpy().reshape(-1)
        pos_cat_train = cats_train[cats_train > 0]

        cats_test = feat_test.to_numpy().reshape(-1)
        pos_cat_test = cats_test[cats_test > 0]

        cat_set = np.unique(np.concatenate([pos_cat_train, pos_cat_test]))
        res_train, sorted_idx = get_percentage(pos_cat_train, cat_set)
        res_test = get_percentage(pos_cat_test, cat_set, is_test=True, sorted_idx=sorted_idx)
        draw(res_train, res_test, len(cat_set), "feat")
        return

    for featname, num in feat_num.items():
        cat_set = np.unique(np.concatenate([feat_train[featname], feat_test[featname]]))
        res_train, sorted_idx = get_percentage(feat_train[featname], cat_set)
        res_test = get_percentage(feat_test[featname], cat_set, is_test=True, sorted_idx=sorted_idx)
        draw(res_train, res_test, num, featname)


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

    with open(MODEL_MAT_PATH, "rb") as file:
        predicted_mat = pickle.load(file)

    df_frequency = draw(df_train, predicted_mat, df_pop, envname, lossname, is_train=True)
    draw(df_test, predicted_mat, df_pop, envname, lossname, is_train=False, df_frequency=df_frequency)


def get_data(dataset):
    if dataset == "CoatEnv":
        from environments.coat.env.Coat import CoatEnv
        df_train, _, _, _ = CoatEnv.get_df_coat("train.ascii")
        df_test = CoatEnv.get_df_coat("test.ascii")[0]
        envname = "CoatEnv-v0"
        yname = "rating"
        threshold = 4
        item_features = ['item_id', 'gender_i', "jackettype", 'color', 'onfrontpage']
        multi_feat = False
        popbin = [0, 10, 20, 40, 60, 80, 100, 200]

    if dataset == "YahooEnv":
        from environments.YahooR3.env.Yahoo import YahooEnv
        df_train = YahooEnv.get_df_yahoo("ydata-ymusic-rating-study-v1_0-train.txt")[0]
        df_test = YahooEnv.get_df_yahoo("ydata-ymusic-rating-study-v1_0-test.txt")[0]
        envname = "YahooEnv-v0"
        yname = "rating"
        threshold = 4
        item_features = ['item_id']
        multi_feat = False
        popbin = [0, 40, 60, 80, 100, 200]

    if dataset == "KuaiRandEnv":
        from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
        df_train, _, _, _ = KuaiRandEnv.get_df_kuairand("train_processed.csv")
        df_test = KuaiRandEnv.get_df_kuairand("test_processed.csv")[0]
        envname = "KuaiRand-v0"
        yname = "is_click"
        threshold = 1
        item_features = ["item_id"] + ["feat" + str(i) for i in range(3)]
        multi_feat = True
        popbin = [0, 10, 20, 40, 80, 150, 300]

    if dataset == "KuaiEnv":
        from environments.KuaiRec.env.KuaiEnv import KuaiEnv
        df_train, _, _, _ = KuaiEnv.get_df_kuairec("big_matrix_processed.csv")
        df_test = KuaiEnv.get_df_kuairec("small_matrix_processed.csv")[0]
        envname = "KuaiEnv-v0"
        yname = "watch_ratio_normed"
        threshold = 0.6
        item_features = ["item_id"] + ["feat" + str(i) for i in range(4)]
        multi_feat = True
        popbin = [0, 10, 20, 40, 60, 80, 100, 200]

    return df_train, df_test, envname, yname, threshold, item_features, multi_feat, popbin


if __name__ == '__main__':
    datasets = [
        "CoatEnv",
        # "YahooEnv",
        "KuaiRandEnv",
        "KuaiEnv"
    ]

    # losses = ["point", "pointneg", "pp", "pair"]

    args = get_args()
    for i in range(len(datasets)):
        df_train, df_test, envname, yname, threshold, item_features, multi_feat, popbin = get_data(datasets[i])
        # for lossname in losses:
        if envname == "YahooEnv-v0":
            df_train = df_train.loc[df_train["user_id"] < 5400]
        get_distribution(threshold, item_features, multi_feat, envname)
        # main(args, df_train, df_test, envname, lossname, yname, popbin)
