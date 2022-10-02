# -*- coding: utf-8 -*-
# @Time    : 2021/10/1 3:03 下午
# @Author  : Chongming GAO
# @FileName: kuaiEnv.py
import json
import os
import pickle

import gym
import torch
from gym import spaces
from numba import njit
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from collections import Counter
import itertools

import pandas as pd
import numpy as np
import random

from tqdm import tqdm


CODEPATH = os.path.dirname(__file__)
ROOTPATH = os.path.dirname(CODEPATH)
DATAPATH = os.path.join(ROOTPATH, "data")


class KuaiRandEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, mat=None, lbe_user=None, lbe_video=None, list_feat=None, df_video_env=None, df_dist_small=None,
                 num_leave_compute=5, leave_threshold=1, max_turn=100):

        self.max_turn = max_turn

        if mat is not None:
            self.mat = mat
            self.lbe_user = lbe_user
            self.lbe_video = lbe_video
            self.list_feat = list_feat
            self.df_video_env = df_video_env
            self.df_dist_small = df_dist_small
        else:
            self.mat, self.lbe_user, self.lbe_video, self.list_feat, self.df_video_env, self.df_dist_small = self.load_mat()

        self.list_feat_small = list(map(lambda x: self.list_feat[x], self.lbe_video.classes_))

        self.observation_space = spaces.Box(low=0, high=len(self.mat) - 1, shape=(1,), dtype=np.int32)
        self.action_space = spaces.Box(low=0, high=self.mat.shape[1] - 1, shape=(1,), dtype=np.int32)

        self.num_leave_compute = num_leave_compute
        self.leave_threshold = leave_threshold

        self.reset()

    @staticmethod
    def load_mat(filepath_GT):
        df_mat = pd.read_csv(filepath_GT, header=0, nrows=1000000)

        # mat_distance = get_distance_mat(mat)

        num_item = mat.shape[1]
        distance = np.zeros([num_item, num_item])
        mat_distance = get_distance_mat1(mat, distance)

        df_item = CoatEnv.load_item_feat()

        # dist_cat = np.zeros_like(mat_distance)
        # for i in range(len(dist_cat)):
        #     for j in range(len(dist_cat)):
        #         sim = (sum(df_item.loc[i] - df_item.loc[j] == 0) / len(df_item.columns))
        #         dist_cat[i,j] = 6 if sim == 0 else 1 / sim
        #
        #
        # dist_cat[np.isinf(dist_cat)] = 6
        # dist_cat = dist_cat * 3
        # df = pd.DataFrame(zip(mat_distance.reshape([-1]), dist_cat.reshape([-1])), columns=["dist","category"])
        #
        # df.groupby("category").agg(np.mean)
        #
        # import seaborn as sns
        # sns.boxplot(x = df["category"], y=df["dist"])
        # from matplotlib import pyplot as plt
        # plt.show()

        # import seaborn as sns
        # sns.histplot(data=mat_distance.reshape([-1]))
        # from matplotlib import pyplot as plt
        # plt.show()

        return mat, df_item, mat_distance

    @staticmethod
    def load_user_feat():
        print("load user features")
        filepath = os.path.join(DATAPATH, 'user_features_pure.csv')
        df_user = pd.read_csv(filepath, usecols=['user_id', 'user_active_degree',
                                                 'is_live_streamer', 'is_video_author', 'follow_user_num_range',
                                                 'fans_user_num_range', 'friend_user_num_range',
                                                 'register_days_range'] + [f'onehot_feat{x}' for x in range(18)]
                              )
        for col in ['user_active_degree',
                    'is_live_streamer', 'is_video_author', 'follow_user_num_range',
                    'fans_user_num_range', 'friend_user_num_range', 'register_days_range']:

            df_user[col] = df_user[col].map(lambda x: chr(0) if x == 'UNKNOWN' else x)
            lbe = LabelEncoder()
            df_user[col] = lbe.fit_transform(df_user[col])
            # print(lbe.classes_)
            if chr(0) in lbe.classes_.tolist() or -124 in lbe.classes_.tolist():
                assert lbe.classes_[0] in {-124, chr(0)}
                # do not add one
            else:
                df_user[col] += 1
        for col in [f'onehot_feat{x}' for x in range(18)]:
            df_user.loc[df_user[col].isna(), col] = -124
            lbe = LabelEncoder()
            df_user[col] = lbe.fit_transform(df_user[col])
            # print(lbe.classes_)
            if chr(0) in lbe.classes_.tolist() or -124 in lbe.classes_.tolist():
                assert lbe.classes_[0] in {-124, chr(0)}
                # do not add one
            else:
                df_user[col] += 1

        df_user = df_user.set_index("user_id")
        return df_user

    @staticmethod
    def get_df_kuairand(name, is_sort=True):
        filename = os.path.join(DATAPATH, name)
        df_data = pd.read_csv(filename,
                              usecols=['user_id', 'item_id', 'time_ms', 'is_like', 'is_click', 'long_view',
                                       'duration_normed', "watch_ratio_normed"])

        # df_data['watch_ratio'] = df_data["play_time_ms"] / df_data["duration_ms"]
        # df_data.loc[df_data['watch_ratio'].isin([np.inf, np.nan]), 'watch_ratio'] = 0
        # df_data.loc[df_data['watch_ratio'] > 5, 'watch_ratio'] = 5
        # df_data['duration_01'] = df_data['duration_ms'] / 1e5
        # df_data.rename(columns={"time_ms": "timestamp"}, inplace=True)
        # df_data["timestamp"] /= 1e3

        # load feature info
        list_feat, df_feat = KuaiRandEnv.load_category()
        df_data = df_data.join(df_feat, on=['item_id'], how="left")

        df_item = KuaiRandEnv.load_item_feat()

        # load user info
        df_user = KuaiRandEnv.load_user_feat()
        df_data = df_data.join(df_user, on=['user_id'], how="left")

        # get user sequences
        if is_sort:
            df_data.sort_values(["user_id", "time_ms"], inplace=True)
            df_data.reset_index(drop=True, inplace=True)

        return df_data, df_user, df_item, list_feat

    @staticmethod
    def load_category():
        print("load item feature")
        filepath = os.path.join(DATAPATH, 'video_features_basic_pure.csv')
        df_item = pd.read_csv(filepath, usecols=["tag"], dtype=str)
        ind = df_item['tag'].isna()
        df_item['tag'].loc[~ind] = df_item['tag'].loc[~ind].map(lambda x: eval(f"[{x}]"))
        df_item['tag'].loc[ind] = [[-1]] * ind.sum()

        list_feat = df_item['tag'].to_list()

        df_feat = pd.DataFrame(list_feat, columns=['feat0', 'feat1', 'feat2'], dtype=int)
        df_feat.index.name = "item_id"
        df_feat[df_feat.isna()] = -1
        df_feat = df_feat + 1
        df_feat = df_feat.astype(int)

        return list_feat, df_feat

    @staticmethod
    def load_item_feat(only_small=False):
        list_feat, df_feat = KuaiRandEnv.load_category()
        video_mean_duration = KuaiRandEnv.load_video_duration()
        df_item = df_feat.join(video_mean_duration, on=['item_id'], how="left")

        return df_item

    @staticmethod
    def load_video_duration():
        duration_path = os.path.join(DATAPATH, "video_duration_normed.csv")
        if os.path.isfile(duration_path):
            video_mean_duration = pd.read_csv(duration_path, header=0)["duration_normed"]
        else:
            small_path = os.path.join(DATAPATH, "test_processed.csv")
            small_duration = pd.read_csv(small_path, header=0, usecols=["item_id", 'duration_normed'])
            big_path = os.path.join(DATAPATH, "train_processed.csv")
            big_duration = pd.read_csv(big_path, header=0, usecols=["item_id", 'duration_normed'])
            duration_all = small_duration.append(big_duration)
            video_mean_duration = duration_all.groupby("item_id").agg(lambda x: sum(list(x)) / len(x))[
                "duration_normed"]
            video_mean_duration.to_csv(duration_path, index=False)

        video_mean_duration.index.name = "item_id"
        return video_mean_duration

    @staticmethod
    def get_similarity_mat(list_feat):
        similarity_mat_path = os.path.join(DATAPATH, "similarity_mat_video.csv")
        if os.path.isfile(similarity_mat_path):
            # with open(similarity_mat_path, 'rb') as f:
            #     similarity_mat = np.load(f, allow_pickle=True, fix_imports=True)
            print("loading similarity matrix...")
            df_sim = pd.read_csv(similarity_mat_path, index_col=0)
            df_sim.columns = df_sim.columns.astype(int)
            print("loading completed.")
            similarity_mat = df_sim.to_numpy()
        else:
            series_feat_list = pd.Series(list_feat)
            df_feat_list = series_feat_list.to_frame("categories")
            df_feat_list.index.name = "item_id"

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

    @staticmethod
    def get_distance_mat(list_feat, sub_index_list):
        if sub_index_list is not None:
            distance_mat_small_path = os.path.join(DATAPATH, "distance_mat_video_small.csv")
            if os.path.isfile(distance_mat_small_path):
                print("loading small distance matrix...")
                df_dist_small = pd.read_csv(distance_mat_small_path, index_col=0)
                df_dist_small.columns = df_dist_small.columns.astype(int)
                print("loading completed.")
            else:
                similarity_mat = KuaiEnv.get_similarity_mat(list_feat)
                df_sim = pd.DataFrame(similarity_mat)
                df_sim_small = df_sim.loc[sub_index_list, sub_index_list]

                df_dist_small = 1.0 / df_sim_small

                df_dist_small.to_csv(distance_mat_small_path)

            return df_dist_small

        return None

    @property
    def state(self):
        if self.action is None:
            res = self.cur_user
        else:
            res = self.action
        return np.array([res])

    def __user_generator(self):
        user = random.randint(0, len(self.mat) - 1)
        # # todo for debug
        # user = 0
        return user

    def step(self, action):
        # action = int(action)

        # Action: tensor with shape (32, )
        self.action = action
        t = self.total_turn
        done = self._determine_whether_to_leave(t, action)
        if t >= (self.max_turn - 1):
            done = True
        self._add_action_to_history(t, action)

        reward = self.mat[self.cur_user, action]

        self.cum_reward += reward
        self.total_turn += 1

        if done:
            self.cur_user = self.__user_generator()

        return self.state, reward, done, {'cum_reward': self.cum_reward}

    def reset(self):
        self.cum_reward = 0
        self.total_turn = 0
        self.cur_user = self.__user_generator()

        self.action = None  # Add by Chongming
        self._reset_history()

        return self.state

    def render(self, mode='human', close=False):
        history_action = self.history_action
        category = {k: self.list_feat_small[v] for k, v in history_action.items()}
        # category_debug = {k:self.list_feat[v] for k,v in history_action.items()}
        # return history_action, category, category_debug
        return self.cur_user, history_action, category

    def _determine_whether_to_leave(self, t, action):
        # self.list_feat[action]
        if t == 0:
            return False
        window_actions = self.sequence_action[t - self.num_leave_compute:t]
        hist_categories_each = list(map(lambda x: self.list_feat_small[x], window_actions))

        # hist_set = set.union(*list(map(lambda x: self.list_feat[x], self.sequence_action[t - self.num_leave_compute:t-1])))

        hist_categories = list(itertools.chain(*hist_categories_each))
        hist_dict = Counter(hist_categories)
        category_a = self.list_feat_small[action]
        for c in category_a:
            if hist_dict[c] > self.leave_threshold:
                return True

        # if action in window_actions:
        #     return True

        return False

    def _reset_history(self):
        self.history_action = {}
        self.sequence_action = []
        self.max_history = 0

    def _add_action_to_history(self, t, action):

        self.sequence_action.append(action)
        self.history_action[t] = action

        assert self.max_history == t
        self.max_history += 1


# @njit
# def find_negative(user_ids, item_ids, mat_train, df_negative):
#     for i in range(len(user_ids)):
#         user, item = user_ids[i], item_ids[i]
#         value = mat_train[user, item]
#
#         neg = item + 1
#         # neg_v = mat_train[user, neg]
#
#         while neg < mat_train.shape[1]:
#             neg_v = mat_train[user, neg]
#             if neg_v >= value:
#                 neg += 1
#             else:
#                 df_negative[i, 0] = user
#                 df_negative[i, 1] = neg
#                 break
#         else:
#             neg = item - 1
#             while neg >= 0:
#                 neg_v = mat_train[user, neg]
#                 if neg_v >= value:
#                     neg -= 1
#                 else:
#                     df_negative[i, 0] = user
#                     df_negative[i, 1] = neg
#                     break

# def negative_sampling(df_train, df_feat, df_user, y_name):
#     print("negative sampling...")
#     mat_train = csr_matrix((df_train[y_name], (df_train["user_id"], df_train["item_id"])),
#                            shape=(df_train['user_id'].max() + 1, df_train['item_id'].max() + 1)).toarray()
#     df_negative = np.zeros([len(df_train), 2])
#
#     user_ids = df_train["user_id"].to_numpy()
#     item_ids = df_train["item_id"].to_numpy()
#
#     if y_name == "watch_ratio_normed":
#         find_negative(user_ids, item_ids, mat_train, df_negative, is_rand=True)
#     else:
#         df_train[y_name]
#
#
#
#     df_negative = pd.DataFrame(df_negative, columns=["user_id", "item_id"], dtype=int)
#
#     df_negative = df_negative.join(df_feat, on=['item_id'], how="left")
#     df_negative = df_negative.join(df_user, on=['user_id'], how="left")
#     df_duration = df_train[["item_id", "duration_normed"]].groupby("item_id").agg(np.mean)
#     df_negative = df_negative.join(df_duration, on=['item_id'], how="left")
#     df_negative.loc[df_negative["duration_normed"].isna(), "duration_normed"] = 0
#     return df_train, df_negative


def construct_complete_val_x(dataset_val, user_features, item_features):
    df_item_env = dataset_val.df_item_val
    df_user = KuaiRandEnv.load_user_feat()

    user_ids = np.arange(dataset_val.x_columns[dataset_val.user_col].vocabulary_size)
    # user_ids = np.arange(1000)
    item_ids = np.arange(dataset_val.x_columns[dataset_val.item_col].vocabulary_size)
    df_user_complete = pd.DataFrame(
        df_user.loc[user_ids].reset_index()[user_features].to_numpy().repeat(len(item_ids), axis=0),
        columns=df_user.reset_index()[user_features].columns)
    df_item_complete = pd.DataFrame(np.tile(df_item_env.reset_index()[item_features], (len(user_ids), 1)),
                                    columns=df_item_env.reset_index()[item_features].columns)

    # np.tile(np.concatenate([np.expand_dims(df_item_env.index.to_numpy(), df_item_env.to_numpy()], axis=1), (2, 1))

    df_x_complete = pd.concat([df_user_complete, df_item_complete], axis=1)
    return df_x_complete

def compute_normed_reward_for_all(user_model, dataset_val, user_features, item_features):
    df_x_complete = construct_complete_val_x(dataset_val, user_features, item_features)
    n_user, n_item = df_x_complete[["user_id", "item_id"]].nunique()
    predict_mat = np.zeros((n_user, n_item))

    for i, user in tqdm(enumerate(range(n_user)), total=n_user, desc="predict all users' rewards on all items"):
        ui = torch.tensor(df_x_complete[df_x_complete["user_id"] == user].to_numpy(),dtype=torch.float, device=user_model.device, requires_grad=False)
        reward_u = user_model.forward(ui).detach().squeeze().cpu().numpy()
        predict_mat[i] = reward_u

    minn = predict_mat.min()
    maxx = predict_mat.max()

    normed_mat = (predict_mat - minn) / (maxx - minn)

    return normed_mat