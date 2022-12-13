import datetime
import json
import os
import random
import time
from collections import defaultdict


import numpy as np
import pandas as pd
import torch

import sys

from tqdm import tqdm

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from core.configs import get_features, get_training_data, get_true_env, get_val_data, get_common_args

from tianshou.data import VectorReplayBuffer, Batch
from tianshou.env import DummyVectorEnv

from util.utils import create_dir
import logzero
from logzero import logger


def prepare_dir_log(args):
    # %% 1. Create dirs
    MODEL_SAVE_PATH = os.path.join(".", "saved_models", args.env, args.model_name)
    create_dirs = [os.path.join(".", "saved_models"),
                   os.path.join(".", "saved_models", args.env),
                   MODEL_SAVE_PATH,
                   os.path.join(MODEL_SAVE_PATH, "logs")]
    create_dir(create_dirs)

    nowtime = datetime.datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d-%H_%M_%S")
    logger_path = os.path.join(MODEL_SAVE_PATH, "logs", "[{}]_{}.log".format(args.message, nowtime))
    logzero.logfile(logger_path)
    logger.info(json.dumps(vars(args), indent=2))

    return MODEL_SAVE_PATH, logger_path


def construct_buffer_from_offline_data(args, df_train, env):
    num_bins = args.test_num

    df_user_num = df_train[["user_id", "item_id"]].groupby("user_id").agg(len)

    if args.env == 'KuaiEnv-v0':
        assert hasattr(env, "lbe_user")
        df_user_num_mapped = df_user_num.loc[env.lbe_user.classes_]
        df_user_num_mapped = df_user_num_mapped.reset_index(drop=True)
        assert len(env.mat) == len(df_user_num_mapped)
    elif args.env == 'YahooEnv-v0':
        df_user_num_mapped = df_user_num.iloc[:len(env.mat)]
    else:  # KuaiRand-v0 and CoatEnv-v0
        df_user_num_mapped = df_user_num

    df_user_num_sorted = df_user_num_mapped.sort_values("item_id", ascending=False)

    bins = np.zeros([num_bins])
    bins_ind = defaultdict(set)
    for user, num in df_user_num_sorted.reset_index().to_numpy():
        ind = bins.argmin()
        bins_ind[ind].add(user)
        bins[ind] += num
        np.zeros([num_bins])

    max_size = max(bins)
    buffer_size = max_size * num_bins
    buffer = VectorReplayBuffer(buffer_size, num_bins)

    # env, env_task_class, kwargs_um = get_true_env(args)
    env.max_turn = max_size

    if args.env == 'KuaiEnv-v0':
        assert hasattr(env, "lbe_item")
        df_numpy = df_train[["user_id", "item_id", args.yfeat]].to_numpy()
        indices = [False] * len(df_numpy)
        for k, (user,item,yfeat) in tqdm(enumerate(df_numpy), total=len(df_numpy)):
            if int(item) in env.lbe_item.classes_:
                indices[k] = True
        df_filtered = df_train[["user_id", "item_id", args.yfeat]].loc[indices]
        df_filtered["user_id"] = 0
        # df_user_items = df_filtered.groupby("user_id").agg(list)

        num_each = np.ceil(len(df_filtered) / num_bins)
        buffer_size = num_each * num_bins
        buffer = VectorReplayBuffer(buffer_size, num_bins)

        ind_pair = zip(np.arange(0,buffer_size,num_each), np.arange(num_each,buffer_size + num_each,num_each))
        for left, right in ind_pair:









    else:
        # a = df_tuple.apply(lambda x: x["item_id"] in env.lbe_item.classes_ and x["user_id"] in env.lbe_user.classes_, axis=1)

        df_user_items = df_train[["user_id", "item_id", args.yfeat]].groupby("user_id").agg(list)

    # if hasattr(env, "lbe_user"):
    #     df_user_items_mapped = df_user_items.loc[env.lbe_user.classes_].reset_index(drop=True)
    # else:
    #     df_user_items_mapped = df_user_items
    for indices, users in tqdm(bins_ind.items(), total=len(bins_ind), desc="preparing offline data into buffer..."):
        for user in users:
            items = [-1] + df_user_items.loc[user][0]
            rewards = df_user_items.loc[user][1]
            np_ui_pair = np.vstack([np.ones_like(items) * user, items]).T

            env.reset()
            env.cur_user = user
            dones = np.zeros(len(rewards), dtype=bool)

            for k, item in enumerate(items[1:]):
                obs_next, rew, done, info = env.step(item)
                if done:
                    env.reset()
                    env.cur_user = user
                dones[k] = done
                dones[-1] = True
                # print(env.cur_user, obs_next, rew, done, info)

            batch = Batch(obs=np_ui_pair[:-1], obs_next=np_ui_pair[1:], act=items[1:],
                          policy={}, info={}, rew=rewards, done=dones)

            ptr, ep_rew, ep_len, ep_idx = buffer.add(batch, buffer_ids=np.ones([len(batch)], dtype=int) * indices)

    return buffer


def prepare_buffer_via_offline_data(args):
    df_train, df_user, df_item, list_feat = get_training_data(args.env)
    # df_val, df_user_val, df_item_val, list_feat = get_val_data(args.env)
    # df_train = df_train.head(10000)
    if "time_ms" in df_train.columns:
        df_train.rename(columns={"time_ms": "timestamp"}, inplace=True)
        df_train = df_train.sort_values(["user_id", "timestamp"])
    if not "timestamp" in df_train.columns:
        df_train = df_train.sort_values(["user_id"])

    df_train[["user_id", "item_id"]].to_numpy()

    env, env_task_class, kwargs_um = get_true_env(args)
    buffer = construct_buffer_from_offline_data(args, df_train, env)
    env.max_turn = args.max_turn

    test_envs = DummyVectorEnv(
        [lambda: env_task_class(**kwargs_um) for _ in range(args.test_num)])

    args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    # seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    return env, buffer, test_envs
