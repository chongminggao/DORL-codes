# -*- coding: utf-8 -*-
# @Time    : 2022/7/19 15:47
# @Author  : Chongming GAO
# @FileName: mytestRL.py

import argparse
import functools
import datetime

import json
import os
import pickle
import random
import time

import torch
from torch import nn

import sys
sys.path.extend(["./src","./src/DeepCTR-Torch"])

from core.inputs import SparseFeatP, input_from_feature_columns
from core.user_model_pairwise import UserModel_Pairwise
from core.util import negative_sampling, compute_exposure_effect_kuaiRec
from deepctr_torch.inputs import DenseFeat, build_input_features, combined_dnn_input
import pandas as pd
import numpy as np

from core.user_model import StaticDataset

import logzero
from logzero import logger

from environments.KuaiRec.env.KuaiEnv import KuaiEnv
from evaluation import test_kuaishou
# from util.upload import my_upload
from util.utils import create_dir, LoggerCallback_Update

CODEPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(CODEPATH, "environments", "KuaiRec", "data")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action="store_true")
    parser.add_argument("--env", type=str, default='KuaiEnv-v0')

    # recommendation related:
    # parser.add_argument('--not_softmax', action="store_false")
    parser.add_argument('--is_softmax', dest='is_softmax', action='store_true')
    parser.add_argument('--not_softmax', dest='is_softmax', action='store_false')
    parser.set_defaults(is_softmax=True)

    parser.add_argument('--l2_reg_dnn', default=0.1, type=float)
    parser.add_argument('--lambda_ab', default=10, type=float)

    parser.add_argument('--epsilon', default=0, type=float)
    parser.add_argument('--is_ucb', dest='is_ucb', action='store_true')
    parser.add_argument('--no_ucb', dest='is_ucb', action='store_false')
    parser.set_defaults(is_ucb=False)

    parser.add_argument('--use_pairwise', dest='use_pairwise', action='store_true')
    parser.add_argument('--no_use_pairwise', dest='use_pairwise', action='store_false')
    parser.set_defaults(use_pairwise=True)

    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--entity_dim", type=int, default=16)
    parser.add_argument("--user_model_name", type=str, default="DeepFM")
    parser.add_argument('--dnn', default=(64, 64), type=int, nargs="+")
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--cuda', default=1, type=int)
    # # env:
    parser.add_argument('--leave_threshold', default=1, type=float)
    parser.add_argument('--num_leave_compute', default=5, type=int)
    # exposure parameters:
    parser.add_argument('--tau', default=0, type=float)

    parser.add_argument('--is_ab', dest='is_ab', action='store_true')
    parser.add_argument('--no_ab', dest='is_ab', action='store_false')
    parser.set_defaults(is_ab=False)
    parser.add_argument("--message", type=str, default="UserModel1")

    args = parser.parse_known_args()[0]
    return args


def load_dataset_kuaishou(user_features, item_features, reward_features, tau, entity_dim, feature_dim, MODEL_SAVE_PATH):
    filename = os.path.join(DATAPATH, "big_matrix.csv")
    df_big = pd.read_csv(filename, usecols=['user_id', 'video_id', 'timestamp', 'watch_ratio_normed', 'video_duration'])
    df_big['video_duration'] /= 1000

    # load feature info
    list_feat, df_feat = KuaiEnv.load_category()

    df_big = df_big.join(df_feat, on=['video_id'], how="left")
    df_big.loc[df_big['watch_ratio_normed'] > 5, 'watch_ratio_normed'] = 5

    # user_features = ["user_id"]
    # item_features = ["video_id"] + ["feat" + str(i) for i in range(4)] + ["video_duration"]
    # reward_features = ["watch_ratio_normed"]

    df_x, df_y = df_big[user_features + item_features], df_big[reward_features]

    x_columns = [SparseFeatP("user_id", df_big['user_id'].max() + 1, embedding_dim=entity_dim)] + \
                [SparseFeatP("video_id", df_big['video_id'].max() + 1, embedding_dim=entity_dim)] + \
                [SparseFeatP("feat{}".format(i),
                             df_feat.max().max() + 1,
                             embedding_dim=feature_dim,
                             embedding_name="feat",  # Share the same feature!
                             padding_idx=0  # using padding_idx in embedding!
                             ) for i in range(4)] + \
                [DenseFeat("video_duration", 1)]

    ab_columns = [SparseFeatP("alpha_u", df_big['user_id'].max() + 1, embedding_dim=1)] + \
                 [SparseFeatP("beta_i", df_big['video_id'].max() + 1, embedding_dim=1)]

    y_columns = [DenseFeat("y", 1)]

    df_negative = negative_sampling(df_big, df_feat, DATAPATH)
    df_x_neg, df_y_neg = df_negative[user_features + item_features], df_negative[reward_features]

    df_x_neg = df_x_neg.rename(columns={k: k + "_neg" for k in df_x_neg.columns.to_numpy()})

    df_x_all = pd.concat([df_x, df_x_neg], axis=1)

    if tau == 0:
        exposure_pos = np.zeros([len(df_x_all), 1])
    else:
        timestamp = df_big['timestamp']
        exposure_pos = compute_exposure_effect_kuaiRec(df_x, timestamp, list_feat, tau, MODEL_SAVE_PATH, DATAPATH)

    dataset = StaticDataset(x_columns, y_columns, num_workers=4)
    dataset.compile_dataset(df_x_all, df_y, exposure_pos)

    return dataset, x_columns, y_columns, ab_columns

def load_static_validate_data_kuairec(user_features, item_features, reward_features, entity_dim, feature_dim, DATAPATH):
    filename = os.path.join(DATAPATH, "small_matrix.csv")
    df_small = pd.read_csv(filename, usecols=['user_id', 'video_id', 'watch_ratio_normed', 'video_duration'])
    df_small['video_duration'] /= 1000


    list_feat, df_feat = KuaiEnv.load_category()

    df_small = df_small.join(df_feat, on=['video_id'], how="left")
    df_small.loc[df_small['watch_ratio_normed'] > 5, 'watch_ratio_normed'] = 5

    # user_features = ["user_id"]
    # item_features = ["video_id"] + ["feat" + str(i) for i in range(4)] + ["video_duration"]
    # reward_features = ["watch_ratio_normed"]

    col_names = user_features + item_features + reward_features

    df_x, df_y = df_small[user_features + item_features], df_small[reward_features]

    x_columns = [SparseFeatP("user_id", df_small['user_id'].max() + 1, embedding_dim=entity_dim)] + \
                [SparseFeatP("video_id", df_small['video_id'].max() + 1, embedding_dim=entity_dim)] + \
                [SparseFeatP("feat{}".format(i),
                             df_feat.max().max() + 1,
                             embedding_dim=feature_dim,
                             embedding_name="feat",  # Share the same feature!
                             padding_idx=0  # using padding_idx in embedding!
                             ) for i in range(4)] + \
                [DenseFeat("video_duration", 1)]

    y_columns = [DenseFeat("y", 1)]

    video_mean_duration = KuaiEnv.load_video_duration()

    dataset_val = StaticDataset(x_columns, y_columns, num_workers=4)
    dataset_val.compile_dataset(df_x, df_y)

    # Setup!
    item_list = df_small["video_id"].unique()
    df_item_env = df_feat.loc[item_list]
    df_item_env = df_item_env.join(video_mean_duration, on=['video_id'], how="left")

    dataset_val.set_env_items(df_item_env)

    return dataset_val

def main(args):
    args.entity_dim = args.feature_dim
    # %% 1. Create dirs
    MODEL_SAVE_PATH = os.path.join(".", "saved_models", args.env, args.user_model_name)

    create_dirs = [os.path.join(".", "saved_models"),
                   os.path.join(".", "saved_models", args.env),
                   MODEL_SAVE_PATH,
                   os.path.join(MODEL_SAVE_PATH, "logs"),
                   os.path.join(MODEL_SAVE_PATH, "mats"),
                   os.path.join(MODEL_SAVE_PATH, "embeddings"),
                   os.path.join(MODEL_SAVE_PATH, "params"),
                   os.path.join(MODEL_SAVE_PATH, "models")]
    create_dir(create_dirs)

    nowtime = datetime.datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d-%H_%M_%S")
    logger_path = os.path.join(MODEL_SAVE_PATH, "logs", "[{}]_{}.log".format(args.message, nowtime))
    logzero.logfile(logger_path)
    logger.info(json.dumps(vars(args), indent=2))

    # %% 2. Prepare Envs
    mat, lbe_user, lbe_video, list_feat, df_video_env, df_dist_small = KuaiEnv.load_mat()
    # register(
    #     id=args.env,  # 'KuaiEnv-v0',
    #     entry_point='environments.KuaiRec.env.KuaiEnv:KuaiEnv',
    #     kwargs={"mat": mat,
    #             "lbe_user": lbe_user,
    #             "lbe_video": lbe_video,
    #             "num_leave_compute": args.num_leave_compute,
    #             "leave_threshold": args.leave_threshold,
    #             "list_feat": list_feat,
    #             "df_video_env": df_video_env,
    #             "df_dist_small": df_dist_small}
    # )
    # env = gym.make(args.env)

    kwargs = {"mat": mat,
              "lbe_user": lbe_user,
              "lbe_video": lbe_video,
              "num_leave_compute": args.num_leave_compute,
              "leave_threshold": args.leave_threshold,
              "list_feat": list_feat,
              "df_video_env": df_video_env,
              "df_dist_small": df_dist_small}
    env = KuaiEnv(**kwargs)

    # %% 3. Prepare dataset
    user_features = ["user_id"]
    item_features = ["video_id"] + ["feat" + str(i) for i in range(4)] + ["video_duration"]
    reward_features = ["watch_ratio_normed"]
    static_dataset, x_columns, y_columns, ab_columns = load_dataset_kuaishou(user_features, item_features,
                                                                             reward_features,
                                                                             args.tau, args.entity_dim,
                                                                             args.feature_dim,
                                                                             MODEL_SAVE_PATH)
    if not args.is_ab:
        ab_columns = None

    dataset_val = load_static_validate_data_kuairec(user_features, item_features, reward_features,
                                                     args.entity_dim, args.feature_dim, DATAPATH)

    # %% 4. Setup model
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    SEED = 2022
    np.random.seed(SEED)
    random.seed(SEED)

    task = "regression"
    task_logit_dim = 1
    model = UserModel_Pairwise(x_columns, y_columns, task, task_logit_dim,
                               dnn_hidden_units=args.dnn, seed=SEED, l2_reg_dnn=args.l2_reg_dnn,
                               device=device, ab_columns=ab_columns)

    model.compile(optimizer="adam",
                  # loss_dict=task_loss_dict,
                  loss_func=loss_kuaishou_pairwise if args.use_pairwise else loss_kuaishou_pointwise,
                  metric_fun={"mae": lambda y, y_predict: nn.functional.l1_loss(torch.from_numpy(y),
                                                                                torch.from_numpy(y_predict)).numpy(),
                              "mse": lambda y, y_predict: nn.functional.mse_loss(torch.from_numpy(y),
                                                                                 torch.from_numpy(y_predict)).numpy()},
                  metrics=None)  # No evaluation step at offline stage

    model.compile_RL_test(
        functools.partial(test_kuaishou, env=env, dataset_val=dataset_val, is_softmax=args.is_softmax,
                          epsilon=args.epsilon, is_ucb=args.is_ucb))

    # %% 5. Learn model
    history = model.fit_data(static_dataset, dataset_val,
                             batch_size=args.batch_size, epochs=args.epoch,
                             callbacks=[[LoggerCallback_Update(logger_path)]])
    logger.info(history.history)

    # %% 6. Save model

    MODEL_MAT_PATH = os.path.join(MODEL_SAVE_PATH, "mats", f"[{args.message}]_mat.pickle")
    MODEL_PARAMS_PATH = os.path.join(MODEL_SAVE_PATH, "params", f"[{args.message}]_params.pickle")
    MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "models", f"[{args.message}]_model.pt")
    MODEL_EMBEDDING_PATH = os.path.join(MODEL_SAVE_PATH, "embeddings", f"[{args.message}]_emb.pt")
    USER_EMBEDDING_PATH = os.path.join(MODEL_SAVE_PATH, "embeddings", f"[{args.message}]_emb_user.pt")
    ITEM_EMBEDDING_PATH = os.path.join(MODEL_SAVE_PATH, "embeddings", f"[{args.message}]_emb_item.pt")

    # (1) Compute and save Mat
    normed_mat = KuaiEnv.compute_normed_reward(model, lbe_user, lbe_video, df_video_env)
    # mat_save_path = os.path.join(MODEL_SAVE_PATH, "normed_mat-{}.pickle".format(args.message))
    with open(MODEL_MAT_PATH, "wb") as f:
        pickle.dump(normed_mat, f)

    # (2) Save params
    model_parameters = {"feature_columns": x_columns, "y_columns": y_columns, "task": task,
                        "task_logit_dim": task_logit_dim, "dnn_hidden_units": args.dnn, "seed": SEED, "device": device,
                        "ab_columns": ab_columns}

    # model_parameter_path = os.path.join(MODEL_SAVE_PATH,
    #                                     "{}_params_{}.pickle".format(args.user_model_name, args.message))
    with open(MODEL_PARAMS_PATH, "wb") as output_file:
        pickle.dump(model_parameters, output_file)

    # (3) Save Model
    #  To cpu
    model = model.cpu()
    model.linear_model.device = "cpu"
    model.linear.device = "cpu"
    # for linear_model in user_model.linear_model_task:
    #     linear_model.device = "cpu"

    # model_save_path = os.path.join(MODEL_SAVE_PATH, "{}_{}.pt".format(args.user_model_name, args.message))
    torch.save(model.state_dict(), MODEL_PATH)

    # (4) Save Embedding
    torch.save(model.embedding_dict.state_dict(), MODEL_EMBEDDING_PATH)

    num_user = static_dataset.x_columns[0].vocabulary_size
    num_item = static_dataset.x_columns[1].vocabulary_size

    user_columns = x_columns[:1]
    item_columns = x_columns[1:]

    # Get item representation
    list_feat, df_feat = KuaiEnv.load_category()
    df_item = pd.DataFrame(range(num_item), columns=["video_id"])
    df_item = df_item.join(df_feat, on=['video_id'], how="left")
    video_mean_duration = KuaiEnv.load_video_duration()
    df_item = df_item.join(video_mean_duration, on=['video_id'], how="left")

    feature_index_item = build_input_features(item_columns)
    tensor_item = torch.FloatTensor(df_item.to_numpy())
    sparse_embedding_list, dense_value_list = input_from_feature_columns(tensor_item, item_columns,
                                                                         model.embedding_dict,
                                                                         feature_index=feature_index_item,
                                                                         support_dense=True, device='cpu')
    representation_item = combined_dnn_input(sparse_embedding_list, dense_value_list)

    # Get user representation
    df_user = pd.DataFrame(range(num_user), columns=["user_id"])

    feature_index_user = build_input_features(user_columns)
    tensor_user = torch.FloatTensor(df_user.to_numpy())
    sparse_embedding_list_user, dense_value_list_user = input_from_feature_columns(tensor_user, user_columns,
                                                                                   model.embedding_dict,
                                                                                   feature_index=feature_index_user,
                                                                                   support_dense=True, device='cpu')
    representation_user = combined_dnn_input(sparse_embedding_list_user, dense_value_list_user)

    torch.save(representation_item, ITEM_EMBEDDING_PATH)
    torch.save(representation_user, USER_EMBEDDING_PATH)



    # %% 7. Upload logs

    REMOTE_ROOT = "/root/Counterfactual_IRS"
    LOCAL_PATH = logger_path
    REMOTE_PATH = os.path.join(REMOTE_ROOT, os.path.dirname(LOCAL_PATH))

    # my_upload(LOCAL_PATH, REMOTE_PATH, REMOTE_ROOT)


sigmoid = nn.Sigmoid()

def loss_kuaishou_pointwise(y, y_deepfm_pos, y_deepfm_neg, exposure, alpha_u=None, beta_i=None):
    if alpha_u is not None:
        exposure_new = exposure * alpha_u * beta_i
        loss_ab = ((alpha_u - 1) ** 2).mean() + ((beta_i - 1) ** 2).mean()
    else:
        exposure_new = exposure
        loss_ab = 0

    y_exposure = 1 / (1 + exposure_new) * y_deepfm_pos
    loss_y = ((y_exposure - y) ** 2).mean()

    bpr_click = 0

    loss = loss_y + args.lambda_ab * loss_ab

    return loss

def loss_kuaishou_pairwise(y, y_deepfm_pos, y_deepfm_neg, exposure, alpha_u=None, beta_i=None):
    if alpha_u is not None:
        exposure_new = exposure * alpha_u * beta_i
        loss_ab = ((alpha_u - 1) ** 2).mean() + ((beta_i - 1) ** 2).mean()
    else:
        exposure_new = exposure
        loss_ab = 0

    y_exposure = 1 / (1 + exposure_new) * y_deepfm_pos

    loss_y = ((y_exposure - y) ** 2).mean()
    bpr_click = - sigmoid(y_deepfm_pos - y_deepfm_neg).log().mean()

    loss = loss_y + bpr_click + args.lambda_ab * loss_ab

    return loss


if __name__ == '__main__':
    args = get_args()
    main(args)
