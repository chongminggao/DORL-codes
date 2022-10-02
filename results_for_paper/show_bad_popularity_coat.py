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
import numpy as np
import pandas as pd

from environments.coat.env.Coat import CoatEnv

CODEPATH = os.path.dirname(__file__)
ROOTPATH = os.path.dirname(CODEPATH)
DATAPATH = os.path.join(ROOTPATH, ".","environments","coat")



parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="CoatEnv-v0")
parser.add_argument("--user_model_name", type=str, default="DeepFM")
parser.add_argument("--read_message", type=str, default="pointneg")

args = parser.parse_known_args()[0]

UM_SAVE_PATH = os.path.join(ROOTPATH, "saved_models", args.env, args.user_model_name)
MODEL_MAT_PATH = os.path.join(UM_SAVE_PATH, "mats", f"[{args.read_message}]_mat.pickle")

with open(MODEL_MAT_PATH, "rb") as file:
    predicted_mat = pickle.load(file)

filename = "train.ascii"
filepath = os.path.join(DATAPATH, filename)
mat_train = pd.read_csv(filepath, sep="\s+", header=None)
isexposure = mat_train > 0
df_popularity = isexposure.sum()

df_train, _, _ = CoatEnv.get_df_coat("train.ascii")
df_frequency = df_train.groupby(["user_id", "item_id"])["rating"].agg(len)
df_train["y_pred"] = predicted_mat[df_train["user_id"],df_train["item_id"]] * 5
df_train["item_pop"] = df_popularity.loc[df_train["item_id"]].reset_index(drop=True)
df_train["bonus"] = np.abs(df_train["y_pred"] - df_train["rating"])


import seaborn as sns
import matplotlib.pyplot as plt
# sns.countplot(x=df_popularity)
# plt.show()

sns.lineplot(x=df_train["item_pop"], y=df_train["bonus"])
plt.show()

df_train["pop_group"] = df_train["item_pop"]//20
sns.boxplot(x=df_train["pop_group"], y=df_train["bonus"])
plt.show()

df_test, _, _ = CoatEnv.get_df_coat("test.ascii")
df_frequency = df_test.groupby(["user_id", "item_id"])["rating"].agg(len)
df_test["y_pred"] = predicted_mat[df_test["user_id"], df_test["item_id"]] * 5
df_test["item_pop"] = df_popularity.loc[df_test["item_id"]].reset_index(drop=True)
df_test["bonus"] = np.abs(df_test["y_pred"] - df_test["rating"])

sns.lineplot(x=df_test["item_pop"], y=df_test["bonus"])
plt.show()

df_test["pop_group"] = df_test["item_pop"] // 20
sns.boxplot(x=df_test["pop_group"], y=df_test["bonus"])
plt.show()
a = 1


