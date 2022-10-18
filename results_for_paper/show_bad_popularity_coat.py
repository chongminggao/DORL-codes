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

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
import torch
from numba import njit
from tqdm import tqdm

matplotlib.use('Agg')
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


def main(args, df_train, df_test, envname, lossname, yname):
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
    train_emb = np.concatenate([train_user_emb,train_item_emb],axis=-1)
    test_user_emb = user_train_embedding[df_test["user_id"]]
    test_item_emb = item_train_embedding[df_test["item_id"]]
    test_emb = np.concatenate([test_user_emb, test_item_emb], axis=-1)

    from scipy import stats
    from scipy.special import softmax
    from sklearn.neighbors import KernelDensity

    from sklearn import preprocessing



    a = np.vstack([train_emb, np.expand_dims(train_emb[0,:],0).repeat(1,axis=0)])
    kde = KernelDensity(kernel='gaussian', bandwidth=4).fit(train_emb)
    res = kde.score_samples(train_emb)
    # b = softmax(res)

    res_test = kde.score_samples(test_emb)
    # c = softmax(res_test)

    scaler = preprocessing.MinMaxScaler()
    bb = scaler.fit_transform(res.reshape(-1,1))

    cc = scaler.transform(res_test.reshape(-1,1))
    cc.mean()

    df_train["kde"] = bb
    df_test["kde"] = cc

    sns.histplot(bb)
    sns.histplot(cc)
    plt.savefig(os.path.join(CODEPATH, f'kde.pdf'), bbox_inches='tight', pad_inches=0)
    plt.close()




    with open(MODEL_MAT_PATH, "rb") as file:
        predicted_mat = pickle.load(file)

    df_popularity = df_train[["item_id", "user_id"]].groupby("item_id").agg(len)

    miss_id = list(set(range(df_train["item_id"].max() + 1)) - set(df_popularity.index))
    df_miss = pd.DataFrame({"id": miss_id, "user_id": 0})
    df_miss.set_index("id", inplace=True)
    df_pop = df_popularity.append(df_miss)
    df_pop = df_pop.sort_index()
    df_pop.rename(columns={"user_id": "count"}, inplace=True)

    # for feat in df_train.columns[3:]:
    #     df_feat_pop = df_train[[feat, "user_id"]].groupby(feat).agg(len)
    #     print(df_feat_pop)

    feat = "age"
    df_feat_pop = df_train[[feat, "user_id"]].groupby(feat).agg(len)
    print(df_feat_pop)




    # df_pop = df_pop.sort_values(by="count").reset_index(drop=True)

    bins = [0, 10, 20, 40, 60, 80, 100, 200] + [df_pop.max()[0] + 1]
    pop_res = {}
    for left, right in zip(bins[:-1], bins[1:]):
        df_pop.loc[df_pop["count"].map(lambda x: x >= left and x < right), "pop_group"] = f"({left},{right})"
        # df_pop.loc[df_pop["count"].map(lambda x: x >= left and x < right), "pop_group"] = left
        pop_res[f"({left},{right})"] = sum(df_pop["count"].map(lambda x: x >= left and x < right))

    print(pop_res)

    sns.histplot(data=df_pop, x="count", bins=100)
    plt.savefig(os.path.join(CODEPATH, f'dist_pop_{envname}.pdf'), bbox_inches='tight', pad_inches=0)
    plt.close()

    # df_train, _, _, _ = CoatEnv.get_df_coat("train.ascii")


    def visual(df_data, df_pop):
        fig = plt.figure(figsize=(24, 14))
        plt.subplots_adjust(wspace=0.3)
        plt.subplots_adjust(hspace=0.3)
        axs = []

        ax1 = plt.subplot2grid((2, 4), (0, 0))
        sns.histplot(data=df_pop, x="count", bins=100, ax=ax1)

        ax2 = plt.subplot2grid((2, 4), (0, 1))
        sns.countplot(data=df_data, x="pop_group", order=sorted(df_data["pop_group"].unique()), ax=ax2)

        ax3 = plt.subplot2grid((2, 4), (0, 2))
        sns.histplot(data=df_data, x="kde", ax=ax3)

        ax4 = plt.subplot2grid((2, 4), (0, 3))
        sns.countplot(data=df_data, x="kde_group", order=sorted(df_data["kde_group"].unique()), ax=ax4)

        ax5 = plt.subplot2grid((2, 4), (1, 0))
        sns.scatterplot(data=df_data, x="item_pop", y="kde", ax=ax5)

        ax6 = plt.subplot2grid((2, 4), (1, 1))
        sns.boxplot(data=df_data, x="kde_group", y="item_pop", order=sorted(df_data["kde_group"].unique()), ax=ax6)

        ax7 = plt.subplot2grid((2, 4), (1, 2))
        sns.boxplot(data=df_data, x="pop_group", y="kde", order=sorted(df_data["pop_group"].unique()), ax=ax7)

        plt.savefig(os.path.join(CODEPATH, f'all_{envname}.pdf'), bbox_inches='tight', pad_inches=0)
        plt.close()








    def draw(df_data, predicted_mat, df_pop, is_train, df_frequency=None):
        df_data["y_pred"] = predicted_mat[df_data["user_id"], df_data["item_id"]] * df_data[yname].max()
        df_data["item_pop"] = df_pop["count"].loc[df_data["item_id"]].reset_index(drop=True)
        df_data["pop_group"] = df_pop["pop_group"].loc[df_data["item_id"]].reset_index(drop=True)
        df_data["error"] = np.abs(df_data["y_pred"] - df_data[yname])

        # if is_train:
        #     df_frequency = df_data.groupby(["user_id", "item_id"])[yname].agg(len)
        # # df_data["expo_times"] = df_data.apply(lambda x: df_frequency.loc[x["user_id"], x["item_id"]], axis=1)
        # df_sorted = df_data.sort_values(["item_id", "user_id"])
        # df_sorted.reset_index(drop=True, inplace=True)
        #
        # array = df_sorted[["item_id", "user_id"]].to_numpy()
        # ans = np.zeros([len(array)], dtype=int)
        #
        # if is_train:
        #     df_sorted["expo_times"] = get_exposure_times(array, ans)
        #     df_sorted.loc[df_sorted[yname] == 0, "expo_times"] = 0  # for kuairand (point pair pp) loss
        # else:
        #     df_sorted["expo_times"] = df_sorted.apply(lambda x: df_frequency.loc[
        #         x["user_id"], x["item_id"]] if df_frequency.__contains__((x["user_id"], x["item_id"])) else 0, axis=1)
        # df_data = df_sorted

        field_train = "train" if is_train else "test"

        # %% visual distribution of the exposure times
        # sns.countplot(df_sorted["expo_times"])
        # plt.savefig(os.path.join(CODEPATH, f'dist_{field_train}_{envname}.pdf'), bbox_inches='tight', pad_inches=0)

        # %% group exposure times
        # times_list = [0, 1, 2, 3]
        # for times in times_list:
        #     df_sorted.loc[df_sorted["expo_times"] == times, "expo_group"] = f"{times}"
        #     df_sorted.loc[df_sorted["expo_times"] > times, "expo_group"] = f">{times}"

        # sns.boxplot(data=df_sorted, x="expo_group", y="error")
        # plt.savefig(os.path.join(CODEPATH, f'expo_{field_train}_{lossname}_{envname}.pdf'), bbox_inches='tight',
        #             pad_inches=0)
        # plt.close()

        # %% visual distribution of the kde
        sns.histplot(df_sorted["kde"])
        plt.savefig(os.path.join(CODEPATH, f'kde_{field_train}_{envname}.pdf'), bbox_inches='tight', pad_inches=0)

        # %% group kde
        kde_list = [df_data["kde"].min()] + [0, 0.2, 0.4, 0.6, 0.8] + [df_data["kde"].max()]
        for left, right in zip(kde_list[:-1], kde_list[1:]):
            # df_sorted.loc[df_sorted["kde"] == times, "kde_group"] = times
            df_data.loc[df_data["kde"] >= left, "kde_group"] = f"({left},{right})"

        sns.boxplot(data=df_sorted, x="kde_group", y="error")
        plt.savefig(os.path.join(CODEPATH, f'kde_group_{field_train}_{lossname}_{envname}.pdf'), bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        # df_sorted = df_sorted.loc[df_sorted["expo_times"] > 1]

        # sns.lineplot(x=df_sorted["kde_group"], y=df_sorted["error"],hue=df_sorted["pop_group"])
        sns.lineplot(data=df_sorted, x="kde_group", y="error", hue="pop_group", err_style="band")
        plt.savefig(os.path.join(CODEPATH, f'kde_line_{field_train}_{lossname}_{envname}.pdf'), bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        sns.lineplot(data=df_sorted, x="kde", y="error")
        plt.savefig(os.path.join(CODEPATH, f'curve_{field_train}_{lossname}_{envname}.pdf'), bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        sns.lineplot(data=df_sorted, x="item_pop", y="error")
        plt.savefig(os.path.join(CODEPATH, f'curve_{field_train}_{lossname}_{envname}.pdf'), bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        df_sorted["pop_group"] = df_sorted["item_pop"] // (df_sorted["item_pop"].max() / 6)
        sns.boxplot(data=df_sorted, x="pop_group", y="error")
        plt.savefig(os.path.join(CODEPATH, f'box_{field_train}_{lossname}_{envname}.pdf'), bbox_inches='tight',
                    pad_inches=0)
        # plt.show()
        plt.close()

        return df_frequency if is_train else None

    df_frequency = draw(df_train, predicted_mat, df_pop, is_train=True)
    draw(df_test, predicted_mat, df_pop, is_train=False, df_frequency=df_frequency)

    # df_frequency_train = df_train.groupby(["user_id", "item_id"])[yname].agg(len)
    # df_train["y_pred"] = predicted_mat[df_train["user_id"], df_train["item_id"]] * df_train[yname].max()
    # df_train["item_pop"] = df_pop.loc[df_train["item_id"]].reset_index(drop=True)
    # df_train["error"] = np.abs(df_train["y_pred"] - df_train[yname])
    #
    # df_train["expo_times"] = df_train.apply(lambda x: df_frequency_train.loc[x["user_id"], x["item_id"]], axis=1)
    # df_train.loc[df_train[yname] == 0, "expo_times"] = 0  # for kuairand (point pair pp) loss
    #
    # times_list = [0, 1, 2]
    # for times in times_list:
    #     df_train.loc[df_train["expo_times"] == times, "expo_group"] = f"{times}"
    # df_train.loc[df_train["expo_times"] > times, "expo_group"] = f">{times}"
    #
    #
    #
    # sns.boxplot(x=df_train["expo_group"], y=df_train["error"])
    # plt.savefig(os.path.join(CODEPATH, f'expo_train_{lossname}_{envname}.pdf'), bbox_inches='tight', pad_inches=0)
    # # plt.show()
    # plt.close()
    #
    # sns.lineplot(x=df_train["item_pop"], y=df_train["error"])
    # plt.savefig(os.path.join(CODEPATH, f'train_{lossname}_{envname}.pdf'), bbox_inches='tight', pad_inches=0)
    # # plt.show()
    # plt.close()
    #
    # df_train["pop_group"] = df_train["item_pop"] // (df_train["item_pop"].max() / 6)
    # sns.boxplot(x=df_train["pop_group"], y=df_train["error"])
    # plt.savefig(os.path.join(CODEPATH, f'box_train_{lossname}_{envname}.pdf'), bbox_inches='tight', pad_inches=0)
    # # plt.show()
    # plt.close()
    #
    # # %% Test
    #
    # df_frequency_test = df_test.groupby(["user_id", "item_id"])[yname].agg(len)
    # df_test["y_pred"] = predicted_mat[df_test["user_id"], df_test["item_id"]] * df_test[yname].max()
    # df_test["item_pop"] = df_pop.loc[df_test["item_id"]].reset_index(drop=True)
    # df_test["error"] = np.abs(df_test["y_pred"] - df_test[yname])
    #
    # df_test["expo_times"] = df_test.apply(lambda x: df_frequency_test.loc[x["user_id"], x["item_id"]], axis=1)
    # times_list = [0, 1, 2]
    # for times in times_list:
    #     df_test.loc[df_test["expo_times"] == times, "expo_group"] = f"{times}"
    # df_test.loc[df_test["expo_times"] > times, "expo_group"] = f">{times}"
    # df_test.loc[df_test[yname] == 0, "expo_times"] = 0  # for kuairand (point pair pp) loss
    # df_test = df_test.loc[df_test["expo_times"] > 1]
    #
    #
    # sns.boxplot(x=df_test["expo_group"], y=df_test["error"])
    # plt.savefig(os.path.join(CODEPATH, f'expo_test_{lossname}_{envname}.pdf'), bbox_inches='tight', pad_inches=0)
    # # plt.show()
    # plt.close()
    #
    # sns.lineplot(x=df_test["item_pop"], y=df_test["error"])
    # plt.savefig(os.path.join(CODEPATH, f'test_{lossname}_{envname}.pdf'), bbox_inches='tight', pad_inches=0)
    # # plt.show()
    # plt.close()
    #
    # df_test["pop_group"] = df_test["item_pop"] // (df_test["item_pop"].max() / 4)
    # sns.boxplot(x=df_test["pop_group"], y=df_test["error"])
    # plt.savefig(os.path.join(CODEPATH, f'box_test_{lossname}_{envname}.pdf'), bbox_inches='tight', pad_inches=0)
    # # plt.show()
    # plt.close()
    # a = 1


if __name__ == '__main__':
    from environments.coat.env.Coat import CoatEnv
    from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
    from environments.KuaiRec.env.KuaiEnv import KuaiEnv
    from environments.YahooR3.env.Yahoo import YahooEnv

    df_trains = [
        CoatEnv.get_df_coat("train.ascii")[0],
        # YahooEnv.get_df_yahoo("ydata-ymusic-rating-study-v1_0-train.txt")[0],
        # KuaiRandEnv.get_df_kuairand("train_processed.csv")[0],
        # KuaiEnv.get_df_kuairec("big_matrix_processed.csv")[0],

    ]

    df_tests = [
        CoatEnv.get_df_coat("test.ascii")[0],
        # YahooEnv.get_df_yahoo("ydata-ymusic-rating-study-v1_0-test.txt")[0],
        # KuaiRandEnv.get_df_kuairand("test_processed.csv")[0],
        # KuaiEnv.get_df_kuairec("small_matrix_processed.csv")[0],

    ]

    envs = [
        "CoatEnv-v0",
        # "YahooEnv-v0",
        # "KuaiRand-v0",
        # "KuaiEnv-v0",

    ]

    ynames = [
        "rating",
        # "rating",
        # "is_click",
        # "watch_ratio_normed",

    ]

    losses = ["point", "pointneg", "pp", "pair"]

    args = get_args()
    for i in range(len(ynames)):
        df_train = df_trains[i]
        df_test = df_tests[i]
        envname = envs[i]
        yname = ynames[i]
        # lossname = "pointneg"
        for lossname in losses:
            if envname == "YahooEnv-v0":
                df_train = df_train.loc[df_train["user_id"] < 5400]

            main(args, df_train, df_test, envname, lossname, yname)
