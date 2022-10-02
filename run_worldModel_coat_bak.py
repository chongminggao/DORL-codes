# -*- coding: utf-8 -*-
import argparse
import datetime
import functools

import json
import os
import pickle
import random
import time

import torch
from torch import nn

import sys


from environments.coat.env.Coat import CoatEnv, construct_complete_val_x, compute_normed_reward_for_all

sys.path.extend(["./src", "./src/DeepCTR-Torch"])
from core.evaluation.metrics import get_ranking_results
from core.inputs import SparseFeatP, input_from_feature_columns
from core.user_model_pairwise import UserModel_Pairwise
from core.util import compute_exposure_effect_kuaiRec
from deepctr_torch.inputs import DenseFeat, build_input_features, combined_dnn_input
import pandas as pd
import numpy as np

from core.user_model import StaticDataset

import logzero
from logzero import logger

from environments.coat.env.Coat import CoatEnv, negative_sampling
# from util.upload import my_upload
from util.utils import create_dir, LoggerCallback_Update

CODEPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(CODEPATH, "environments", "coat")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action="store_true")
    parser.add_argument("--env", type=str, default='CoatEnv-v0')
    parser.add_argument("--yfeat", type=str, default='rating')
    parser.add_argument("--optimizer", type=str, default='adam')

    # recommendation related:
    # parser.add_argument('--not_softmax', action="store_false")
    parser.add_argument('--is_softmax', dest='is_softmax', action='store_true')
    parser.add_argument('--no_softmax', dest='is_softmax', action='store_false')
    parser.set_defaults(is_softmax=True)

    parser.add_argument('--rating_threshold', default=4, type=float)
    parser.add_argument('--rankingK', default=(20, 10, 5), type=int, nargs="+")

    parser.add_argument('--is_binarize', dest='is_binarize', action='store_true')
    parser.add_argument('--no_binarize', dest='is_binarize', action='store_false')
    parser.set_defaults(is_binarize=True)

    parser.add_argument('--is_all_item_ranking', dest='is_all_item_ranking', action='store_true')
    parser.add_argument('--no_all_item_ranking', dest='is_all_item_ranking', action='store_false')
    parser.set_defaults(all_item_ranking=False)

    parser.add_argument('--is_userinfo', dest='is_userinfo', action='store_true')
    parser.add_argument('--no_userinfo', dest='is_userinfo', action='store_false')
    parser.set_defaults(is_userinfo=True)

    parser.add_argument('--l2_reg_dnn', default=0.1, type=float)
    parser.add_argument('--lambda_ab', default=10, type=float)

    parser.add_argument('--epsilon', default=0, type=float)
    parser.add_argument('--is_ucb', dest='is_ucb', action='store_true')
    parser.add_argument('--no_ucb', dest='is_ucb', action='store_false')
    parser.set_defaults(is_ucb=False)


    parser.add_argument("--loss", type=str, default='pointneg')

    parser.add_argument("--feature_dim", type=int, default=8)
    parser.add_argument("--entity_dim", type=int, default=8)
    parser.add_argument("--user_model_name", type=str, default="DeepFM")
    parser.add_argument('--dnn', default=(64, 64), type=int, nargs="+")
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--cuda', default=1, type=int)
    # # env:
    parser.add_argument('--leave_threshold', default=1, type=float)
    parser.add_argument('--num_leave_compute', default=5, type=int)
    # exposure parameters:
    parser.add_argument('--tau', default=0, type=float)

    parser.add_argument('--is_ab', dest='is_ab', action='store_true')
    parser.add_argument('--no_ab', dest='is_ab', action='store_false')
    parser.set_defaults(is_ab=False)
    parser.add_argument("--message", type=str, default="point")

    args = parser.parse_known_args()[0]
    return args


def load_dataset_coat(user_features, item_features, reward_features, tau, entity_dim, feature_dim, MODEL_SAVE_PATH):
    df_train, df_user, df_item, list_feat = CoatEnv.get_df_coat("train.ascii")
    df_x = df_train[user_features + item_features]
    df_y = df_train[reward_features]


    x_columns = [SparseFeatP("user_id", df_train['user_id'].max() + 1, embedding_dim=entity_dim)] + \
                [SparseFeatP(col, df_user[col].max() + 1, embedding_dim=feature_dim) for col in user_features[1:]] + \
                [SparseFeatP("item_id", df_train['item_id'].max() + 1, embedding_dim=entity_dim)] + \
                [SparseFeatP(col, df_item[col].max() + 1, embedding_dim=feature_dim) for col in item_features[1:]]

    ab_columns = [SparseFeatP("alpha_u", df_train['user_id'].max() + 1, embedding_dim=1)] + \
                 [SparseFeatP("beta_i", df_train['item_id'].max() + 1, embedding_dim=1)]

    y_columns = [DenseFeat("y", 1)]

    df_negative = negative_sampling(df_train, df_item, df_user)

    df_x_neg = df_negative[user_features + item_features]

    df_x_neg = df_x_neg.rename(columns={k: k + "_neg" for k in df_x_neg.columns.to_numpy()})

    df_x_all = pd.concat([df_x, df_x_neg], axis=1)

    if tau == 0:
        exposure_pos = np.zeros([len(df_x_all), 1])
    else:
        timestamp = df_train['timestamp']
        exposure_pos = compute_exposure_effect_kuaiRec(df_x, timestamp, list_feat, tau, MODEL_SAVE_PATH, DATAPATH)

    dataset = StaticDataset(x_columns, y_columns, num_workers=4)
    dataset.compile_dataset(df_x_all, df_y, exposure_pos)

    return dataset, x_columns, y_columns, ab_columns


def load_static_validate_data_coat(user_features, item_features, reward_features, entity_dim, feature_dim,
                                       DATAPATH):
    df_val, df_user, df_item, list_feat = CoatEnv.get_df_coat("test.ascii")
    df_x = df_val[user_features + item_features]
    df_y = df_val[reward_features]

    # return df_train, df_user, df_item

    x_columns = [SparseFeatP("user_id", df_val['user_id'].max() + 1, embedding_dim=entity_dim)] + \
                [SparseFeatP(col, df_user[col].max() + 1, embedding_dim=feature_dim) for col in user_features[1:]] + \
                [SparseFeatP("item_id", df_val['item_id'].max() + 1, embedding_dim=entity_dim)] + \
                [SparseFeatP(col, df_item[col].max() + 1, embedding_dim=feature_dim) for col in item_features[1:]]

    ab_columns = [SparseFeatP("alpha_u", df_val['user_id'].max() + 1, embedding_dim=1)] + \
                 [SparseFeatP("beta_i", df_val['item_id'].max() + 1, embedding_dim=1)]

    y_columns = [DenseFeat("y", 1)]

    dataset_val = StaticDataset(x_columns, y_columns, num_workers=4)
    dataset_val.compile_dataset(df_x, df_y)


    dataset_val.set_df_item_val(df_item)

    if not any(df_y.to_numpy() % 1):
        # make sure the label is binary
        df_binary = pd.concat([df_val[["user_id", "item_id"]], df_y], axis=1)
        df_ones = df_binary.loc[df_binary[reward_features[0]] > 0]
        ground_truth = df_ones[["user_id", "item_id"] + reward_features].groupby("user_id").agg(list)
        ground_truth.rename(columns={"item_id": "item_id", reward_features[0]: "y"}, inplace=True)


        threshold = args.rating_threshold
        index = ground_truth["y"].map(lambda x: [True if i >= threshold else False for i in x])
        df_temp = pd.DataFrame(index)
        df_temp.rename(columns={"y": "ind"}, inplace=True)
        df_temp["y"] = ground_truth["y"]
        df_temp["true_id"] = ground_truth["item_id"]
        df_true_id = df_temp.apply(lambda x: np.array(x["true_id"])[x["ind"]].tolist(), axis=1)
        df_true_y = df_temp.apply(lambda x: np.array(x["y"])[x["ind"]].tolist(), axis=1)

        if args.is_binarize:
            df_true_y = df_true_y.map(lambda x: [1]*len(x))

        ground_truth_revise = pd.concat([df_true_id,df_true_y], axis=1)
        ground_truth_revise.rename(columns={0:"item_id",1:"y"},inplace=True)
        dataset_val.set_ground_truth(ground_truth_revise)


        assert dataset_val.x_columns[0].name == "user_id"
        dataset_val.set_user_col(0)
        assert dataset_val.x_columns[len(user_features)].name == "item_id"
        dataset_val.set_item_col(len(user_features))


        if args.all_item_ranking:
            dataset_val.set_all_item_ranking_in_evaluation(args.all_item_ranking)
            # user_ids = np.arange(dataset_val.x_columns[dataset_val.user_col].vocabulary_size)
            # # user_ids = np.arange(1000)
            # item_ids = np.arange(dataset_val.x_columns[dataset_val.item_col].vocabulary_size)
            # df_user_complete = pd.DataFrame(
            #     df_user.loc[user_ids].reset_index()[user_features].to_numpy().repeat(len(item_ids), axis=0),
            #     columns=df_user.reset_index()[user_features].columns)
            # df_item_complete = pd.DataFrame(np.tile(df_item.reset_index()[item_features], (len(user_ids), 1)),
            #                                 columns=df_item.reset_index()[item_features].columns)
            #
            # # np.tile(np.concatenate([np.expand_dims(df_item_env.index.to_numpy(), df_item_env.to_numpy()], axis=1), (2, 1))
            #
            # df_x_complete = pd.concat([df_user_complete, df_item_complete], axis=1)

            df_x_complete = construct_complete_val_x(dataset_val, user_features, item_features)
            df_y_complete = pd.DataFrame(np.zeros(len(df_x_complete)), columns=df_y.columns)

            dataset_complete = StaticDataset(x_columns, y_columns, num_workers=4)
            dataset_complete.compile_dataset(df_x_complete, df_y_complete)
            dataset_val.set_dataset_complete(dataset_complete)

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

    # %% 3. Prepare dataset
    user_features = ["user_id", 'gender_u', 'age', 'location', 'fashioninterest']
    item_features = ['item_id', 'gender_i', "jackettype", 'color', 'onfrontpage']
    reward_features = [args.yfeat]

    static_dataset, x_columns, y_columns, ab_columns = load_dataset_coat(user_features, item_features,
                                                                         reward_features,
                                                                         args.tau, args.entity_dim,
                                                                         args.feature_dim,
                                                                         MODEL_SAVE_PATH)

    if not args.is_ab:
        ab_columns = None

    dataset_val = load_static_validate_data_coat(user_features, item_features, reward_features, args.entity_dim,
                                                     args.feature_dim, DATAPATH)

    # %% 4. Setup model
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    SEED = 2022
    np.random.seed(SEED)
    random.seed(SEED)

    task = "regression"
    task_logit_dim = 1
    model = UserModel_Pairwise(x_columns, y_columns, task, task_logit_dim,
                               dnn_hidden_units=args.dnn, seed=SEED, l2_reg_dnn=args.l2_reg_dnn,
                               device=device, ab_columns=ab_columns, init_std=0.001)

    if args.loss == "pair":
        loss_fun = loss_pairwise
    if args.loss == "point":
        loss_fun = loss_pointwise
    if args.loss == "pointneg":
        loss_fun = loss_pointwise_negative
    if args.loss == "pointpair" or args.loss == "pairpoint" or args.loss == "pp":
        loss_fun = loss_pairwise_pointwise

    model.compile(optimizer=args.optimizer,
                  # loss_dict=task_loss_dict,
                  loss_func=loss_fun,
                  metric_fun={"MAE": lambda y, y_predict: nn.functional.l1_loss(torch.from_numpy(y).type(torch.float),
                                                                                torch.from_numpy(y_predict)).numpy(),
                              "MSE": lambda y, y_predict: nn.functional.mse_loss(torch.from_numpy(y).type(torch.float),
                                                                                 torch.from_numpy(y_predict)).numpy(),
                              "RMSE": lambda y, y_predict: nn.functional.mse_loss(torch.from_numpy(y).type(torch.float),
                                                                                  torch.from_numpy(
                                                                                      y_predict)).numpy() ** 0.5
                              },
                  metric_fun_ranking=functools.partial(get_ranking_results, K=args.rankingK,
                                                       metrics=["Recall", "Precision", "NDCG", "HT", "MAP", "MRR"]),
                  metrics=None)  # No evaluation step at offline stage

    # model.compile_RL_test(
    #     functools.partial(test_kuaishou, env=env, dataset_val=dataset_val, is_softmax=args.is_softmax,
    #                       epsilon=args.epsilon, is_ucb=args.is_ucb))

    # %% 5. Learn model
    history = model.fit_data(static_dataset, dataset_val,
                             batch_size=args.batch_size, epochs=args.epoch, shuffle=True,
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
    # normed_mat = CoatEnv.compute_normed_reward(model)
    normed_mat = compute_normed_reward_for_all(model, dataset_val, user_features, item_features)
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
    num_item = static_dataset.x_columns[len(user_features)].vocabulary_size

    user_columns = x_columns[:len(user_features)]
    item_columns = x_columns[len(user_features):]

    # Get item representation
    # list_feat, df_item = CoatEnv.load_category()
    # df_item = pd.DataFrame(range(num_item), columns=["item_id"])
    # df_item = df_item.join(df_item, on=['item_id'], how="left")
    # video_mean_duration = KuaiEnv.load_video_duration()
    # df_item = df_item.join(video_mean_duration, on=['item_id'], how="left")

    df_item = dataset_val.df_item_val
    df_item["item_id"] = df_item.index
    df_item = df_item[[column.name for column in item_columns]]

    feature_index_item = build_input_features(item_columns)
    tensor_item = torch.FloatTensor(df_item.to_numpy())
    sparse_embedding_list, dense_value_list = input_from_feature_columns(tensor_item, item_columns,
                                                                         model.embedding_dict,
                                                                         feature_index=feature_index_item,
                                                                         support_dense=True, device='cpu')
    representation_item = combined_dnn_input(sparse_embedding_list, dense_value_list)

    # Get user representation
    df_user = CoatEnv.load_user_feat()
    df_user["user_id"] = df_user.index
    df_user = df_user[[column.name for column in user_columns]]

    # # df_user = pd.DataFrame(range(num_user), columns=["user_id"])

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


def loss_pointwise_negative(y, y_deepfm_pos, y_deepfm_neg, exposure, alpha_u=None, beta_i=None):
    if alpha_u is not None:
        exposure_new = exposure * alpha_u * beta_i
        loss_ab = ((alpha_u - 1) ** 2).mean() + ((beta_i - 1) ** 2).mean()
    else:
        exposure_new = exposure
        loss_ab = 0

    y_exposure = 1 / (1 + exposure_new) * y_deepfm_pos
    loss_y = ((y_exposure - y) ** 2).sum()
    loss_y_neg = ((y_deepfm_neg - 0) ** 2).sum()

    loss = loss_y + loss_y_neg + args.lambda_ab * loss_ab
    return loss

def loss_pointwise(y, y_deepfm_pos, y_deepfm_neg, exposure, alpha_u=None, beta_i=None):
    if alpha_u is not None:
        exposure_new = exposure * alpha_u * beta_i
        loss_ab = ((alpha_u - 1) ** 2).mean() + ((beta_i - 1) ** 2).mean()
    else:
        exposure_new = exposure
        loss_ab = 0

    y_exposure = 1 / (1 + exposure_new) * y_deepfm_pos
    loss_y = ((y_exposure - y) ** 2).sum()

    bpr_click = 0
    loss = loss_y + args.lambda_ab * loss_ab
    return loss

def loss_pairwise(y, y_deepfm_pos, y_deepfm_neg, exposure, alpha_u=None, beta_i=None):
    if alpha_u is not None:
        exposure_new = exposure * alpha_u * beta_i
        loss_ab = ((alpha_u - 1) ** 2).sum() + ((beta_i - 1) ** 2).sum()
    else:
        exposure_new = exposure
        loss_ab = 0

    # y_exposure = 1 / (1 + exposure_new) * y_deepfm_pos
    # loss_y = ((y_exposure - y) ** 2).sum()

    bpr_click = - sigmoid(y_deepfm_pos - y_deepfm_neg).log().sum()
    loss = bpr_click + args.lambda_ab * loss_ab

    return loss

def loss_pairwise_pointwise(y, y_deepfm_pos, y_deepfm_neg, exposure, alpha_u=None, beta_i=None):
    if alpha_u is not None:
        exposure_new = exposure * alpha_u * beta_i
        loss_ab = ((alpha_u - 1) ** 2).sum() + ((beta_i - 1) ** 2).sum()
    else:
        exposure_new = exposure
        loss_ab = 0

    y_exposure = 1 / (1 + exposure_new) * y_deepfm_pos

    loss_y = ((y_exposure - y) ** 2).sum()
    bpr_click = - sigmoid(y_deepfm_pos - y_deepfm_neg).log().sum()
    loss = loss_y + 0.2 * bpr_click + args.lambda_ab * loss_ab

    return loss


if __name__ == '__main__':
    args = get_args()
    main(args)
