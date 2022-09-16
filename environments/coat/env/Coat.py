# -*- coding: utf-8 -*-
# @Time    : 2022/9/15 22:40
# @Author  : Chongming GAO
# @FileName: Coat.py


import os

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
DATAPATH = ROOTPATH


class CoatEnv(gym.Env):
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

        # smallmat shape: (1411, 3327)

        self.observation_space = spaces.Box(low=0, high=len(self.mat) - 1, shape=(1,), dtype=np.int32)
        self.action_space = spaces.Box(low=0, high=self.mat.shape[1] - 1, shape=(1,), dtype=np.int32)

        self.num_leave_compute = num_leave_compute
        self.leave_threshold = leave_threshold

        self.reset()

    @staticmethod
    def load_user_feat():
        filename_user = os.path.join(DATAPATH, "user_item_features", "user_features.ascii")
        mat_user = pd.read_csv(filename_user, sep="\s+", header=None, dtype=str)

        feat_cols_user = [2, 6, 3, 3]
        feat_cols_user = [0] + list(np.cumsum(feat_cols_user))

        df_user = pd.DataFrame([], columns=['gender_u', 'age', 'location', 'fashioninterest'], dtype=int)
        for k, (left, right) in enumerate(zip(feat_cols_user[:-1], feat_cols_user[1:])):
            col = mat_user[range(left, right)].apply(lambda x: "".join(x), axis=1)
            col_int = col.map(lambda x: np.log2(int(x[::-1], 2)))
            assert not any(col_int % 1)
            df_user[df_user.columns[k]] = col_int.astype(dtype=int)
        df_user.index.name = "user_id"
        return df_user

    @staticmethod
    def load_item_feat():
        filename_item = os.path.join(DATAPATH, "user_item_features", "item_features.ascii")
        mat_item = pd.read_csv(filename_item, sep="\s+", header=None, dtype=str)

        feat_cols_item = [0, 2, 18, 31, 33]

        df_item = pd.DataFrame([], columns=['gender_i', 'jackettype', 'color', 'onfrontpage'], dtype=int)
        for k, (left, right) in enumerate(zip(feat_cols_item[:-1], feat_cols_item[1:])):
            col = mat_item[range(left, right)].apply(lambda x: "".join(x), axis=1)
            col_int = col.map(lambda x: np.log2(int(x[::-1], 2)))
            assert not any(col_int % 1)
            df_item[df_item.columns[k]] = col_int.astype(dtype=int)
        df_item.index.name = "item_id"

        return df_item

    @staticmethod
    def compute_normed_reward(user_model, lbe_user, lbe_video, df_video_env):
        # filename = "normed_reward.pickle"
        # filepath = os.path.join(DATAPATH, filename)

        # if os.path.isfile(filepath):
        #     with open(filepath, "rb") as file:
        #         normed_mat = pickle.load(file)
        #     return normed_mat

        n_user = len(lbe_user.classes_)
        n_item = len(lbe_video.classes_)

        item_info = df_video_env.loc[lbe_video.classes_]
        item_info["item_id"] = item_info.index
        item_info = item_info[["item_id", "feat0", "feat1", "feat2", "feat3", "video_duration"]]
        item_np = item_info.to_numpy()

        predict_mat = np.zeros((n_user, n_item))

        for i, user in tqdm(enumerate(lbe_user.classes_), total=n_user, desc="predict all users' rewards on all items"):
            ui = torch.tensor(np.concatenate((np.ones((n_item, 1)) * user, item_np), axis=1),
                              dtype=torch.float, device=user_model.device, requires_grad=False)
            reward_u = user_model.forward(ui).detach().squeeze().cpu().numpy()
            predict_mat[i] = reward_u

        minn = predict_mat.min()
        maxx = predict_mat.max()

        normed_mat = (predict_mat - minn) / (maxx - minn)

        return normed_mat

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


@njit
def find_negative(user_ids, item_ids, mat_train, df_negative, K=1):
    for i in range(len(user_ids)):
        user, item = user_ids[i], item_ids[i]
        value = mat_train[user, item]

        neg = item + 1
        # neg_v = mat_train[user, neg]

        while neg < mat_train.shape[1]:
            neg_v = mat_train[user, neg]
            if neg_v > 0:
                neg += 1
            else:
                df_negative[i, 0] = user
                df_negative[i, 1] = neg
                break
        else:
            neg = item - 1
            while neg >= 0:
                neg_v = mat_train[user, neg]
                if neg_v > 0:
                    neg -= 1
                else:
                    df_negative[i, 0] = user
                    df_negative[i, 1] = neg
                    break


def negative_sampling(df_train, df_item, df_user):
    print("negative sampling...")
    mat_train = csr_matrix((df_train["rating"], (df_train["user_id"], df_train["item_id"])),
                           shape=(df_train['user_id'].max() + 1, df_train['item_id'].max() + 1)).toarray()
    df_negative = np.zeros([len(df_train), 2])

    user_ids = df_train["user_id"].to_numpy()
    item_ids = df_train["item_id"].to_numpy()

    find_negative(user_ids, item_ids, mat_train, df_negative)
    df_negative = pd.DataFrame(df_negative, columns=["user_id", "item_id"], dtype=int)

    df_negative = df_negative.join(df_item, on=['item_id'], how="left")
    df_negative = df_negative.join(df_user, on=['user_id'], how="left")

    # df_negative.loc[df_negative["duration_ms"].isna(), "duration_ms"] = 0
    return df_negative