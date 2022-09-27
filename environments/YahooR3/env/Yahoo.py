# -*- coding: utf-8 -*-
# @Time    : 2022/9/19 22:19
# @Author  : Chongming GAO
# @FileName: Yahoo.py

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


class YahooEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, mat=None, mat_distance=None,
                 num_leave_compute=5, leave_threshold=1, max_turn=100):

        self.max_turn = max_turn

        if mat is not None:
            self.mat = mat
            self.mat_distance = mat_distance
        else:
            self.mat, self.mat_distance = self.load_mat()

        # smallmat shape: (1411, 3327)

        self.observation_space = spaces.Box(low=0, high=len(self.mat) - 1, shape=(1,), dtype=np.int32)
        self.action_space = spaces.Box(low=0, high=self.mat.shape[1] - 1, shape=(1,), dtype=np.int32)

        self.num_leave_compute = num_leave_compute
        self.leave_threshold = leave_threshold

        self.reset()

    @staticmethod
    def load_mat():
        filename_GT = os.path.join(DATAPATH, "..", "RL4Rec", "data", "yahoo_pseudoGT_ratingM.ascii")
        mat = pd.read_csv(filename_GT, sep="\s+", header=None, dtype=str).to_numpy(dtype=int)

        num_item = mat.shape[1]
        distance = np.zeros([num_item, num_item])
        mat_distance = get_distance_mat(mat,distance)

        # import seaborn as sns
        # sns.histplot(data=mat_distance.reshape([-1]))
        # from matplotlib import pyplot as plt
        # plt.show()

        mat = mat[:5400,:]

        return mat, mat_distance


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
        # history_action = self.history_action
        # category = {k: self.list_feat_small[v] for k, v in history_action.items()}
        # category_debug = {k:self.list_feat[v] for k,v in history_action.items()}
        # return history_action, category, category_debug
        # return self.cur_user, history_action, category
        pass

    def _determine_whether_to_leave(self, t, action):
        if t == 0:
            return False
        window_actions = self.sequence_action[t - self.num_leave_compute:t]
        dist_list = np.array([self.mat_distance[action, x] for x in window_actions])
        if any(dist_list < self.leave_threshold):
            return True

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


@njit
def get_distance_mat(mat, distance):
    matt = np.transpose(mat)
    for item_i in range(len(distance)):
        vec_i = matt[item_i]
        for item_j in range(len(distance)):
            vec_j = matt[item_j]
            dist = ((vec_i-vec_j)**2).sum()**0.5
            distance[item_i, item_j] = dist
    return distance

def negative_sampling(df_train):
    print("negative sampling...")
    mat_train = csr_matrix((df_train["rating"], (df_train["user_id"], df_train["item_id"])),
                           shape=(df_train['user_id'].max() + 1, df_train['item_id'].max() + 1)).toarray()
    df_negative = np.zeros([len(df_train), 2])

    user_ids = df_train["user_id"].to_numpy()
    item_ids = df_train["item_id"].to_numpy()

    find_negative(user_ids, item_ids, mat_train, df_negative)
    df_negative = pd.DataFrame(df_negative, columns=["user_id", "item_id"], dtype=int)

    # df_negative.loc[df_negative["duration_ms"].isna(), "duration_ms"] = 0
    return df_negative