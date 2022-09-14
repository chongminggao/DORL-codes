# -*- coding: utf-8 -*-
# @Time    : 2021/9/11 4:24 下午
# @Author  : Chongming GAO
# @FileName: util.py

import os
import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm


def compute_action_distance(action: np.ndarray, actions_hist: np.ndarray,
                            env_name="VirtualTB-v0", realenv=None):  # for kuaishou data
    if env_name == "VirtualTB-v0":
        a = action - actions_hist
        if len(a.shape) > 1:
            dist = np.linalg.norm(a, axis=1)
        else:
            dist = np.linalg.norm(a)
    elif env_name == "KuaiEnv-v0":
        # df_video_env = realenv.df_video_env
        # list_feat = realenv.list_feat
        # item_index = realenv.lbe_video.inverse_transform([action])
        # item_index_hist = realenv.lbe_video.inverse_transform(actions_hist)
        df_dist_small = realenv.df_dist_small

        dist = df_dist_small.iloc[action, actions_hist].to_numpy()

    return dist


def compute_exposure(t_diff: np.ndarray, dist: np.ndarray, tau):
    if tau <= 0:
        res = 0
        return res
    res = np.sum(np.exp(- t_diff * dist / tau))
    return res


def softplus_np(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def clip0(x):
    return np.amax(x, 0)


@njit
def compute_exposure_each_user(start_index: int,
                               distance_mat: np.ndarray,
                               timestamp: np.ndarray,
                               exposure_all: np.ndarray,
                               index_u: np.ndarray,
                               video_u: np.ndarray,
                               tau: float
                               ):
    for i in range(1, len(index_u)):
        video = video_u[i]
        t_diff = timestamp[index_u[i]] - timestamp[start_index:index_u[i]]
        t_diff[t_diff == 0] = 1  # important!
        # dist_hist = np.fromiter(map(lambda x: distance_mat[x, video], video_u[:i]), np.float)

        dist_hist = np.zeros(i)
        for j in range(i):
            video_j = video_u[j]
            dist_hist[j] = distance_mat[video_j, video]

        exposure = np.sum(np.exp(- t_diff * dist_hist / tau))
        exposure_all[start_index + i] = exposure


def get_similarity_mat(list_feat, DATAPATH):
    similarity_mat_path = os.path.join(DATAPATH, "similarity_mat_video.csv")
    if os.path.isfile(similarity_mat_path):
        # with open(similarity_mat_path, 'rb') as f:
        #     similarity_mat = np.load(f, allow_pickle=True, fix_imports=True)
        print(f"loading similarity matrix... from {similarity_mat_path}")
        df_sim = pd.read_csv(similarity_mat_path, index_col=0)
        df_sim.columns = df_sim.columns.astype(int)
        print("loading completed.")
        similarity_mat = df_sim.to_numpy()
    else:
        series_feat_list = pd.Series(list_feat)
        df_feat_list = series_feat_list.to_frame("categories")
        df_feat_list.index.name = "video_id"

        similarity_mat = np.zeros([len(df_feat_list), len(df_feat_list)])
        print("Compute the similarity matrix (for the first time and will be saved for later usage)")
        for i in tqdm(range(len(df_feat_list)), desc="Computing..."):
            for j in range(i):
                similarity_mat[i, j] = similarity_mat[j, i]
            for j in range(i, len(df_feat_list)):
                similarity_mat[i, j] = len(set(series_feat_list[i]).intersection(set(series_feat_list[j]))) / len(
                    set(series_feat_list[i]).union(set(series_feat_list[j])))

        df_sim = pd.DataFrame(similarity_mat)
        df_sim.to_csv(similarity_mat_path)

    return similarity_mat


def get_distance_mat(list_feat, sub_index_list, DATAPATH):
    if sub_index_list is not None:
        distance_mat_small_path = os.path.join(DATAPATH, "distance_mat_video_small.csv")
        if os.path.isfile(distance_mat_small_path):
            print("loading small distance matrix...")
            df_dist_small = pd.read_csv(distance_mat_small_path, index_col=0)
            df_dist_small.columns = df_dist_small.columns.astype(int)
            print("loading completed.")
        else:
            similarity_mat = get_similarity_mat(list_feat, DATAPATH)
            df_sim = pd.DataFrame(similarity_mat)
            df_sim_small = df_sim.loc[sub_index_list, sub_index_list]

            df_dist_small = 1.0 / df_sim_small

            df_dist_small.to_csv(distance_mat_small_path)

        return df_dist_small

    return None




def compute_exposure_effect_kuaiRec(df_x, timestamp, list_feat, tau, MODEL_SAVE_PATH, DATAPATH):
    exposure_path = os.path.join(MODEL_SAVE_PATH, "..", "saved_exposure", "exposure_pos_{:.1f}.csv".format(tau))

    if os.path.isfile(exposure_path):
        print("loading saved exposure scores: ", exposure_path)
        exposure_pos_df = pd.read_csv(exposure_path)
        exposure_pos = exposure_pos_df.to_numpy()
        return exposure_pos

    similarity_mat = get_similarity_mat(list_feat, DATAPATH)

    distance_mat = 1 / similarity_mat

    exposure_pos = np.zeros([len(df_x), 1])

    user_list = df_x["user_id"].unique()

    timestamp = timestamp.to_numpy()

    print("Compute the exposure effect (for the first time and will be saved for later usage)")
    for user in tqdm(user_list, desc="Computing exposure effect of historical data"):
        df_user = df_x[df_x['user_id'] == user]
        start_index = df_user.index[0]
        index_u = df_user.index.to_numpy()
        video_u = df_user['video_id'].to_numpy()
        compute_exposure_each_user(start_index, distance_mat, timestamp, exposure_pos,
                                   index_u, video_u, tau)

    exposure_pos_df = pd.DataFrame(exposure_pos)

    if not os.path.exists(os.path.dirname(exposure_path)):
        os.mkdir(os.path.dirname(exposure_path))
    exposure_pos_df.to_csv(exposure_path, index=False)

    return exposure_pos

# @njit
# def find_negative(user_ids, video_ids, mat_small, mat_big, mat_negative, max_item):
#     for i in range(len(user_ids)):
#         user, item = user_ids[i], video_ids[i]
#
#         neg = item + 1
#         while neg <= max_item:
#             if neg == 1225:  # 1225 is an absent video_id
#                 neg = 1226
#             if mat_small[user, neg] or mat_big[user, neg] or mat_negative[user, neg]:
#                 neg += 1
#             else:
#                 mat_negative[user, neg] = True
#                 break
#         else:
#             neg = item - 1
#             while neg >= 0:
#                 if neg == 1225:  # 1225 is an absent video_id
#                     neg = 1224
#                 if mat_small[user, neg] or mat_big[user, neg] or mat_negative[user, neg]:
#                     neg -= 1
#                 else:
#                     mat_negative[user, neg] = True
#                     break
