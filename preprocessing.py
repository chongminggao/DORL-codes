# -*- coding: utf-8 -*-


import os
import pandas as pd
import numpy as np
from tqdm import tqdm

CODEPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(CODEPATH, "environments", "KuaiRec", "data")

filename_big = os.path.join(DATAPATH, "big_matrix.csv")
df_big = pd.read_csv(filename_big, usecols=['user_id', 'video_id', 'timestamp', 'watch_ratio', 'video_duration'])
df_big['video_duration'] /= 1000

filename_small = os.path.join(DATAPATH, "small_matrix.csv")
df_small = pd.read_csv(filename_small, usecols=['user_id', 'video_id', 'timestamp', 'watch_ratio', 'video_duration'])
df_small['video_duration'] /= 1000

df = df_big.append(df_small)

all_video = set(df["video_id"])
max_y = df[["video_id", "watch_ratio"]].groupby("video_id").agg(max)
min_y = df[["video_id", "watch_ratio"]].groupby("video_id").agg(min)

df_big["watch_ratio_normed"] = 0.0
df_small["watch_ratio_normed"] = 0.0
for v in tqdm(all_video):
    df_big["watch_ratio_normed"][df_big["video_id"] == v] = \
        (df_big[df_big["video_id"] == v]["watch_ratio"] - min_y.loc[v][0]) / (max_y.loc[v][0] - min_y.loc[v][0])
    df_small["watch_ratio_normed"][df_small["video_id"] == v] = \
        (df_small[df_small["video_id"] == v]["watch_ratio"] - min_y.loc[v][0]) / (max_y.loc[v][0] - min_y.loc[v][0])

df_big["watch_ratio_normed"][df_big["watch_ratio_normed"].isna()] = 0
df_small["watch_ratio_normed"][df_small["watch_ratio_normed"].isna()] = 0

df_big.to_csv(filename_big, index=False)
df_small.to_csv(filename_small, index=False)
