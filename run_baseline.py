import argparse
import collections
import datetime
import functools
import json
import os
import pickle
import random
import sys
import time
import logzero
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from core.user_model_dice import UserModel_DICE

sys.path.extend(["./src", "./src/DeepCTR-Torch"])
from core.evaluation.metrics import get_ranking_results
from core.inputs import SparseFeatP, input_from_feature_columns
from core.static_dataset import StaticDataset
from core.user_model_pairwise import UserModel_Pairwise
from environments.KuaiRec.env.KuaiEnv import compute_exposure_effect_kuaiRec
from core.util import  negative_sampling
from deepctr_torch.inputs import DenseFeat, build_input_features, combined_dnn_input

from util.utils import create_dir


def get_args_all():

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action="store_true")

    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument("--bpr_weight", type=float, default=0.5)
    parser.add_argument('--neg_K', default=5, type=int)

    # recommendation related:
    # parser.add_argument('--not_softmax', action="store_false")
    parser.add_argument('--is_softmax', dest='is_softmax', action='store_true')
    parser.add_argument('--no_softmax', dest='is_softmax', action='store_false')
    parser.set_defaults(is_softmax=True)

    parser.add_argument('--rankingK', default=(20, 10, 5), type=int, nargs="+")
    parser.add_argument('--max_turn', default=30, type=int)

    parser.add_argument('--is_all_item_ranking', dest='is_all_item_ranking', action='store_true')
    parser.add_argument('--no_all_item_ranking', dest='is_all_item_ranking', action='store_false')
    parser.set_defaults(all_item_ranking=False)

    parser.add_argument('--l2_reg_dnn', default=0.1, type=float)
    parser.add_argument('--lambda_ab', default=10, type=float)

    parser.add_argument('--epsilon', default=0, type=float)
    parser.add_argument('--is_ucb', dest='is_ucb', action='store_true')
    parser.add_argument('--no_ucb', dest='is_ucb', action='store_false')
    parser.set_defaults(is_ucb=False)

    parser.add_argument("--feature_dim", type=int, default=8)
    parser.add_argument("--entity_dim", type=int, default=8)
    parser.add_argument("--user_model_name", type=str, default="DeepFM")
    parser.add_argument('--dnn', default=(64, 64), type=int, nargs="+")
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--cuda', default=0, type=int)
    # # env:
    parser.add_argument('--leave_threshold', default=1, type=float)
    parser.add_argument('--num_leave_compute', default=5, type=int)
    # exposure parameters:
    parser.add_argument('--tau', default=0, type=float)

    parser.add_argument('--is_ab', dest='is_ab', action='store_true')
    parser.add_argument('--no_ab', dest='is_ab', action='store_false')
    parser.set_defaults(is_ab=False)
    parser.add_argument("--message", type=str, default="point")

    # TODO: add IPS, PD arguments
    parser.add_argument('--use_IPS', dest='use_IPS', action='store_true')
    parser.add_argument('--no_IPS',dest='use_IPS',action='store_false')
    parser.set_defaults(use_IPS=False)
    parser.add_argument('--use_PD', dest='use_PD', action='store_true')
    parser.add_argument('--no_PD', dest='use_PD', action='store_false')
    parser.set_defaults(use_PD=False)
    parser.add_argument('--PD_gamma', default=0.01, type=float)
    parser.add_argument('--use_DICE', dest='use_DICE', action='store_true')
    parser.add_argument('--no_DICE', dest='use_DICE', action='store_false')
    parser.set_defaults(use_DICE=True)

    args = parser.parse_known_args()[0]
    return args


def get_xy_columns(args, df_data, df_user, df_item, user_features, item_features, entity_dim, feature_dim, is_val=False):
    if args.env == "KuaiRand-v0" or args.env == "KuaiEnv-v0":
        feat = [x for x in df_item.columns if x[:4] == "feat"]
        x_columns = [SparseFeatP("user_id", df_data['user_id'].max() + 1, embedding_dim=entity_dim)] + \
                    [SparseFeatP(col, df_user[col].max() + 1, embedding_dim=feature_dim, padding_idx=0) for col in
                     user_features[1:]] + \
                    [SparseFeatP("item_id", df_data['item_id'].max() + 1, embedding_dim=entity_dim)] + \
                    [SparseFeatP(x,
                                 df_item[feat].max().max() + 1,
                                 embedding_dim=feature_dim,
                                 embedding_name="feat",  # Share the same feature!
                                 padding_idx=0  # using padding_idx in embedding!
                                 ) for x in feat] + \
                    [DenseFeat("duration_normed", 1)]

    else:
        x_columns = [SparseFeatP("user_id", df_data['user_id'].max() + 1, embedding_dim=entity_dim)] + \
                    [SparseFeatP(col, df_user[col].max() + 1, embedding_dim=feature_dim) for col in user_features[1:]] + \
                    [SparseFeatP("item_id", df_data['item_id'].max() + 1, embedding_dim=entity_dim)] + \
                    [SparseFeatP(col, df_item[col].max() + 1, embedding_dim=feature_dim) for col in item_features[1:]]

    # TODO: add columns for DICE(user_id_int, item_id_int)
    if args.use_DICE and not is_val:
        x_columns += [SparseFeatP("user_id_int", df_data['user_id'].max() + 1, embedding_dim=entity_dim),
                    SparseFeatP("item_id_int", df_data['item_id'].max() + 1, embedding_dim=entity_dim)]


    ab_columns = [SparseFeatP("alpha_u", df_data['user_id'].max() + 1, embedding_dim=1)] + \
                 [SparseFeatP("beta_i", df_data['item_id'].max() + 1, embedding_dim=1)]

    y_columns = [DenseFeat("y", 1)]
    return x_columns, y_columns, ab_columns


def load_dataset_train(args, user_features, item_features, reward_features, tau, entity_dim, feature_dim,
                       MODEL_SAVE_PATH, DATAPATH):
    if args.env == "CoatEnv-v0":
        from environments.coat.env.Coat import CoatEnv
        df_train, df_user, df_item, list_feat = CoatEnv.get_df_coat("train.ascii")
    elif args.env == "KuaiRand-v0":
        from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
        df_train, df_user, df_item, list_feat = KuaiRandEnv.get_df_kuairand("train_processed.csv")
    elif args.env == "KuaiEnv-v0":
        from environments.KuaiRec.env.KuaiEnv import KuaiEnv
        df_train, df_user, df_item, list_feat = KuaiEnv.get_df_kuairec("big_matrix_processed.csv")
    elif args.env == "YahooEnv-v0":
        from environments.YahooR3.env.Yahoo import YahooEnv
        df_train, df_user, df_item, list_feat = YahooEnv.get_df_yahoo("ydata-ymusic-rating-study-v1_0-train.txt")

    assert user_features[0] == "user_id"
    assert item_features[0] == "item_id"
    df_user = df_user[user_features[1:]]
    df_item = df_item[item_features[1:]]

    x_columns, y_columns, ab_columns = get_xy_columns(args, df_train, df_user, df_item, user_features, item_features, entity_dim, feature_dim)

    # if args.env == "CoatEnv-v0":
    #     from environments.coat.env.Coat import negative_sampling
    # elif args.env == "KuaiRand-v0":
    #     from environments.KuaiRand_Pure.env.KuaiRand import negative_sampling
    # elif args.env == "KuaiEnv-v0":
    #     from environments.KuaiRec.env.KuaiEnv import negative_sampling
    # elif args.env == "YahooEnv-v0":
    #     from environments.YahooR3.env.Yahoo import negative_sampling



    neg_in_train = True if args.env == "KuaiRand-v0" and reward_features[0] != "watch_ratio_normed" else False

    df_pos, df_neg = negative_sampling(df_train, df_item, df_user, reward_features[0],
                                              is_rand=True, neg_in_train=neg_in_train, neg_K=args.neg_K)

    # TODO: copy user_id,item_id for DICE

    df_x = df_pos[user_features + item_features]
    if args.use_DICE:
        df_x['user_id_int'] = df_x['user_id']
        df_x['item_id_int'] = df_x['item_id']
    if reward_features[0] == "hybrid":  # for kuairand
        a = df_pos["long_view"] + df_pos["is_like"] + df_pos["is_click"]
        df_y = a > 0
        df_y = pd.DataFrame(df_y, dtype=int, columns=["hybrid"])
        df_pos["hybrid"] = df_y["hybrid"]
    else:
        df_y = df_pos[reward_features]

    df_x_neg = df_neg[user_features + item_features]
    if args.use_DICE:
        df_x_neg['user_id_int'] = df_x_neg['user_id']
        df_x_neg['item_id_int'] = df_x_neg['item_id']
    df_x_neg = df_x_neg.rename(columns={k: k + "_neg" for k in df_x_neg.columns.to_numpy()})

    df_x_all = pd.concat([df_x, df_x_neg], axis=1)

    if tau == 0:
        exposure_pos = np.zeros([len(df_x_all), 1])
    else:
        timestamp = df_pos['timestamp']
        exposure_pos = compute_exposure_effect_kuaiRec(df_x, timestamp, list_feat, tau, MODEL_SAVE_PATH, DATAPATH)

    # TODO: add PD/IPS/DICE scores & "get" function

    score = exposure_pos

    if args.use_PD:
        if 'timestamp' in df_train.columns:
            timestamp = df_train['timestamp']
            all_time = df_pos['timestamp']
        else:
            timestamp = None
            all_time = None
        score = get_PD_score(df_train, df_x_all, timestamp, all_time, args.PD_gamma)

    if args.use_IPS:
        score = get_IPS_score(df_train, df_x_all)

    if args.use_DICE:
        score = get_DICE_score(df_train,df_x_all)

    dataset = StaticDataset(x_columns, y_columns, num_workers=4)
    dataset.compile_dataset(df_x_all, df_y, score)

    return dataset, df_user, df_item, x_columns, y_columns, ab_columns

def get_DICE_score(df_train,df_x_all) -> np.ndarray:
    counter = collections.Counter(df_train['item_id'])
    popularity_pos = df_x_all['item_id'].map(lambda x: counter[x])
    popularity_neg = df_x_all['item_id_neg'].map(lambda x: counter[x])
    # compute whether pos item is more popular than neg item
    mask = popularity_pos.to_numpy() > popularity_neg.to_numpy()
    DICE_score = np.where(mask,1,-1)
    return DICE_score

def get_PD_score(df_train,df_x_all,timestamp,all_time,gamma,num_bin=5) -> np.ndarray:
    if timestamp is None:
        num_bin = 1
        all_counter = collections.Counter(df_train['item_id'])
        all_total = sum(all_counter.values())
        popularity = np.array(df_x_all['item_id'].map(lambda x:all_counter[x]/all_total).to_frame())
        popularity_neg = np.array(df_x_all['item_id_neg'].map(lambda x:all_counter[x]/all_total).to_frame())
    else:
        time_max = timestamp.max()
        time_min = timestamp.min()
        print('Time range: [{}] ~ [{}]'.format(time.ctime(time_min),time.ctime(time_max)))
        interval = (time_max - time_min) / num_bin
        dict_start = {i: (interval * i + time_min, interval * (i + 1) + time_min) for i in range(num_bin)}
        dict_index = {}
        dict_counter = {}
        dict_total = {}

        # count items for every stage
        for i in range(num_bin):
            if i < num_bin - 1:
                index = (dict_start[i][0] <= timestamp) & (timestamp < dict_start[i][1])
            else:
                index = (dict_start[i][0] <= timestamp) & (timestamp <= dict_start[i][1])

            dict_index[i] = index
            dict_counter[i] = collections.Counter(df_train.loc[index,'item_id'])
            dict_total[i] = sum(dict_counter[i].values())
        # get popularity for every interaction
        popularity = np.zeros([len(df_x_all),1])
        assert len(df_x_all) == len(all_time)
        for i in range(num_bin):
            if i < num_bin - 1:
                index = (dict_start[i][0] <= all_time) & (all_time < dict_start[i][1])
            else:
                index = (dict_start[i][0] <= all_time) & (all_time <= dict_start[i][1])

            popularity[index] = df_x_all.loc[index,'item_id'].map(lambda x: dict_counter[i][x] / dict_total[i]).to_frame()
        
        # randomly pick popularity for neg samples
        index_neg = np.random.choice(num_bin,len(df_x_all), p = np.array(list(dict_total.values()))/sum(list(dict_total.values())))

        popularity_neg = np.array(list(map(lambda i,x: dict_counter[i][x] / dict_total[i], index_neg, df_x_all['item_id_neg'])))
        popularity_neg = np.expand_dims(popularity_neg,-1)
    
    popularity_all = np.concatenate([popularity,popularity_neg],axis=1)
    PD_score = popularity_all ** gamma
    return PD_score

def get_IPS_score(df_train,df_x_all) -> np.ndarray:
    IPS_item = collections.Counter(df_train['item_id'])
    IPS_data = df_x_all['item_id'].map(lambda x: IPS_item[x])
    IPS_data[IPS_data==0] = 1
    IPS_data = 1.0 / IPS_data
    return IPS_data.to_numpy()


def construct_complete_val_x(dataset_val, df_user, df_item, user_features, item_features):

    user_ids = np.unique(dataset_val.x_numpy[:,dataset_val.user_col].astype(int))
    item_ids = np.unique(dataset_val.x_numpy[:,dataset_val.item_col].astype(int))

    df_user_complete = pd.DataFrame(
        df_user.loc[user_ids].reset_index()[user_features].to_numpy().repeat(len(item_ids), axis=0),
        columns=df_user.reset_index()[user_features].columns)
    df_item_complete = pd.DataFrame(np.tile(df_item.loc[item_ids].reset_index()[item_features], (len(user_ids), 1)),
                                    columns=df_item.loc[item_ids].reset_index()[item_features].columns)

    df_x_complete = pd.concat([df_user_complete, df_item_complete], axis=1)
    return df_x_complete


def compute_normed_reward_for_all(user_model, dataset_val, df_user, df_item, user_features, item_features):
    df_x_complete = construct_complete_val_x(dataset_val, df_user, df_item, user_features, item_features)
    n_user, n_item = df_x_complete[["user_id", "item_id"]].nunique()
    user_ids = np.sort(df_x_complete["user_id"].unique())
    predict_mat = np.zeros((n_user, n_item))

    for i, user in tqdm(enumerate(user_ids), total=n_user, desc="predict all users' rewards on all items"):
        ui = torch.tensor(df_x_complete[df_x_complete["user_id"] == user].to_numpy(), dtype=torch.float,
                          device=user_model.device, requires_grad=False)
        reward_u = user_model.forward(ui).detach().squeeze().cpu().numpy()
        predict_mat[i] = reward_u

    minn = predict_mat.min()
    maxx = predict_mat.max()

    normed_mat = (predict_mat - minn) / (maxx - minn)

    return normed_mat


def load_dataset_val(args, user_features, item_features, reward_features, entity_dim, feature_dim):
    if args.env == "CoatEnv-v0":
        from environments.coat.env.Coat import CoatEnv
        df_val, df_user_val, df_item_val, list_feat = CoatEnv.get_df_coat("test.ascii")
    elif args.env == "KuaiRand-v0":
        from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
        df_val, df_user_val, df_item_val, list_feat = KuaiRandEnv.get_df_kuairand("test_processed.csv")
    elif args.env == "KuaiEnv-v0":
        from environments.KuaiRec.env.KuaiEnv import KuaiEnv
        df_val, df_user_val, df_item_val, list_feat = KuaiEnv.get_df_kuairec("small_matrix_processed.csv")
    elif args.env == "YahooEnv-v0":
        from environments.YahooR3.env.Yahoo import YahooEnv
        df_val, df_user_val, df_item_val, list_feat = YahooEnv.get_df_yahoo("ydata-ymusic-rating-study-v1_0-test.txt")

    assert user_features[0] == "user_id"
    assert item_features[0] == "item_id"
    df_user_val = df_user_val[user_features[1:]]
    df_item_val = df_item_val[item_features[1:]]

    df_x = df_val[user_features + item_features]
    if reward_features[0] == "hybrid":  # for kuairand
        a = df_val["long_view"] + df_val["is_like"] + df_val["is_click"]
        df_y = a > 0
        df_y = pd.DataFrame(df_y, dtype=int, columns=["hybrid"])
    else:
        df_y = df_val[reward_features]

    x_columns, y_columns, ab_columns = get_xy_columns(args, df_val, df_user_val, df_item_val, user_features, item_features,
                                                      entity_dim, feature_dim, is_val=True)

    dataset_val = StaticDataset(x_columns, y_columns, num_workers=4)
    dataset_val.compile_dataset(df_x, df_y)

    dataset_val.set_df_item_val(df_item_val)
    dataset_val.set_df_user_val(df_user_val)

    assert dataset_val.x_columns[0].name == "user_id"
    dataset_val.set_user_col(0)
    assert dataset_val.x_columns[len(user_features)].name == "item_id"
    dataset_val.set_item_col(len(user_features))

    if not any(df_y.to_numpy() % 1): # 整数
        # make sure the label is binary

        df_binary = pd.concat([df_val[["user_id", "item_id"]], df_y], axis=1)
        df_ones = df_binary.loc[df_binary[reward_features[0]] > 0]
        ground_truth = df_ones[["user_id", "item_id"] + reward_features].groupby("user_id").agg(list)
        ground_truth.rename(columns={"item_id": "item_id", reward_features[0]: "y"}, inplace=True)

        # for ranking purpose.
        threshold = args.rating_threshold
        index = ground_truth["y"].map(lambda x: [True if i >= threshold else False for i in x])
        df_temp = pd.DataFrame(index)
        df_temp.rename(columns={"y": "ind"}, inplace=True)
        df_temp["y"] = ground_truth["y"]
        df_temp["true_id"] = ground_truth["item_id"]
        df_true_id = df_temp.apply(lambda x: np.array(x["true_id"])[x["ind"]].tolist(), axis=1)
        df_true_y = df_temp.apply(lambda x: np.array(x["y"])[x["ind"]].tolist(), axis=1)

        if args.is_binarize:
            df_true_y = df_true_y.map(lambda x: [1] * len(x))

        ground_truth_revise = pd.concat([df_true_id, df_true_y], axis=1)
        ground_truth_revise.rename(columns={0: "item_id", 1: "y"}, inplace=True)
        dataset_val.set_ground_truth(ground_truth_revise)

        if args.all_item_ranking:
            dataset_val.set_all_item_ranking_in_evaluation(args.all_item_ranking)

            df_x_complete = construct_complete_val_x(dataset_val, df_user_val, df_item_val, user_features,
                                                     item_features)
            df_y_complete = pd.DataFrame(np.zeros(len(df_x_complete)), columns=df_y.columns)

            dataset_complete = StaticDataset(x_columns, y_columns, num_workers=4)
            dataset_complete.compile_dataset(df_x_complete, df_y_complete)
            dataset_val.set_dataset_complete(dataset_complete)

    return dataset_val, df_user_val, df_item_val


def prepare_dir_log(args):
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
    logzero.logger.info(json.dumps(vars(args), indent=2))
    return MODEL_SAVE_PATH, logger_path


def prepare_dataset(args, user_features, item_features, reward_features, MODEL_SAVE_PATH, DATAPATH):
    dataset_train, df_user, df_item, x_columns, y_columns, ab_columns = \
        load_dataset_train(args, user_features, item_features, reward_features,
                           args.tau, args.entity_dim, args.feature_dim, MODEL_SAVE_PATH, DATAPATH)
    if not args.is_ab:
        ab_columns = None

    dataset_val, df_user_val, df_item_val = load_dataset_val(args, user_features, item_features, reward_features,
                                                             args.entity_dim, args.feature_dim)
    return dataset_train, dataset_val, df_user, df_item, df_user_val, df_item_val, x_columns, y_columns, ab_columns


def setup_world_model(args, x_columns, y_columns, ab_columns, task, task_logit_dim, is_ranking=False):
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    random.seed(args.seed)

    # TODO: add DICE model
    if args.use_DICE:
        user_model = UserModel_DICE(x_columns, y_columns, task, task_logit_dim,
                                    dnn_hidden_units=args.dnn, seed=args.seed, l2_reg_dnn=args.l2_reg_dnn,
                                    device=device, ab_columns=ab_columns, init_std=0.001)
    else:
        user_model = UserModel_Pairwise(x_columns, y_columns, task, task_logit_dim,
                                    dnn_hidden_units=args.dnn, seed=args.seed, l2_reg_dnn=args.l2_reg_dnn,
                                    device=device, ab_columns=ab_columns, init_std=0.001)

    # TODO add PD/IPS/DICE loss
    if args.use_PD:
        if args.loss not in ['pair', 'pointpair', 'pairpoint', 'pp']:
            raise RuntimeError(f"{args.loss} is not supported using PD!")
        if args.loss == 'pair':
            loss_fun = loss_pairwise_PD
        else:
            loss_fun = loss_pairpoint_PD
    elif args.use_IPS:
        if args.loss == "pair":
            loss_fun = loss_pairwise_IPS
        elif args.loss == "point":
            loss_fun = loss_pointwise_IPS
        elif args.loss == "pointpair" or args.loss == "pairpoint" or args.loss == "pp":
            loss_fun = loss_pairpoint_IPS
        else:
            raise RuntimeError(f"{args.loss} is not supported using IPS!")
    elif args.use_DICE:
        if args.loss not in ['pair', 'pointpair', 'pairpoint', 'pp']:
            raise RuntimeError(f"{args.loss} is not supported using DICE!")
        if args.loss == 'pair':
            loss_fun = loss_pairwise_DICE
        else:
            loss_fun = loss_pairpoint_DICE
    else:
        if args.loss == "pair":
            loss_fun = loss_pairwise
        if args.loss == "point":
            loss_fun = loss_pointwise
        if args.loss == "pointneg":
            loss_fun = loss_pointwise_negative
        if args.loss == "pointpair" or args.loss == "pairpoint" or args.loss == "pp":
            loss_fun = loss_pairwise_pointwise

    user_model.compile(optimizer=args.optimizer,
                       # loss_dict=task_loss_dict,
                       loss_func=functools.partial(loss_fun, args=args),
                       metric_fun={
                           "MAE": lambda y, y_predict: nn.functional.l1_loss(torch.from_numpy(y).type(torch.float),
                                                                             torch.from_numpy(y_predict)).numpy(),
                           "MSE": lambda y, y_predict: nn.functional.mse_loss(torch.from_numpy(y).type(torch.float),
                                                                              torch.from_numpy(y_predict)).numpy(),
                           "RMSE": lambda y, y_predict: nn.functional.mse_loss(torch.from_numpy(y).type(torch.float),
                                                                               torch.from_numpy(
                                                                                   y_predict)).numpy() ** 0.5
                       },
                       metric_fun_ranking=
                       functools.partial(get_ranking_results, K=args.rankingK,
                                         metrics=["Recall", "Precision", "NDCG", "HT", "MAP", "MRR"]
                                         ) if is_ranking else None,
                       metrics=None)

    # No evaluation step at offline stage
    # model.compile_RL_test(
    #     functools.partial(test_kuaishou, env=env, dataset_val=dataset_val, is_softmax=args.is_softmax,
    #                       epsilon=args.epsilon, is_ucb=args.is_ucb))

    return user_model


# %% 6. Save model
def save_world_model(args, user_model, dataset_train, dataset_val, x_columns, df_user, df_item, df_user_val, df_item_val,
                     user_features, item_features, model_parameters, MODEL_SAVE_PATH, logger_path):
    MODEL_MAT_PATH = os.path.join(MODEL_SAVE_PATH, "mats", f"[{args.message}]_mat.pickle")
    MODEL_PARAMS_PATH = os.path.join(MODEL_SAVE_PATH, "params", f"[{args.message}]_params.pickle")
    MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "models", f"[{args.message}]_model.pt")
    MODEL_EMBEDDING_PATH = os.path.join(MODEL_SAVE_PATH, "embeddings", f"[{args.message}]_emb.pt")
    USER_EMBEDDING_PATH = os.path.join(MODEL_SAVE_PATH, "embeddings", f"[{args.message}]_emb_user.pt")
    ITEM_EMBEDDING_PATH = os.path.join(MODEL_SAVE_PATH, "embeddings", f"[{args.message}]_emb_item.pt")
    USER_VAL_EMBEDDING_PATH = os.path.join(MODEL_SAVE_PATH, "embeddings", f"[{args.message}]_emb_user_val.pt")
    ITEM_VAL_EMBEDDING_PATH = os.path.join(MODEL_SAVE_PATH, "embeddings", f"[{args.message}]_emb_item_val.pt")

    # (1) Compute and save Mat
    normed_mat = compute_normed_reward_for_all(user_model, dataset_val, df_user, df_item, user_features, item_features)
    with open(MODEL_MAT_PATH, "wb") as f:
        pickle.dump(normed_mat, f)

    # (2) Save params
    with open(MODEL_PARAMS_PATH, "wb") as output_file:
        pickle.dump(model_parameters, output_file)

    # (3) Save Model
    # TODO: save DICE model
    #  To cpu
    model = user_model.cpu()
    model.linear_model.device = "cpu"
    if args.use_DICE:
        model.linear_main.device= 'cpu'
        model.linear_ui.device = 'cpu'
    else:
        model.linear.device = "cpu"
    # for linear_model in user_model.linear_model_task:
    #     linear_model.device = "cpu"

    # model_save_path = os.path.join(MODEL_SAVE_PATH, "{}_{}.pt".format(args.user_model_name, args.message))
    torch.save(model.state_dict(), MODEL_PATH)

    # (4) Save Embedding
    if args.use_DICE:
        # do not save.....
        exit(0)
    torch.save(model.embedding_dict.state_dict(), MODEL_EMBEDDING_PATH)

    def save_embedding(df_save, columns, SAVEPATH):
        df_save = df_save.reset_index(drop=False)
        df_save = df_save[[column.name for column in columns]]

        feature_index = build_input_features(columns)
        tensor_save = torch.FloatTensor(df_save.to_numpy())
        sparse_embedding_list, dense_value_list = input_from_feature_columns(tensor_save, columns,
                                                                             model.embedding_dict,
                                                                             feature_index=feature_index,
                                                                             support_dense=True, device='cpu')
        representation_save = combined_dnn_input(sparse_embedding_list, dense_value_list)
        torch.save(representation_save, SAVEPATH)
        return representation_save

    user_columns = x_columns[:len(user_features)]
    item_columns = x_columns[len(user_features):]

    representation_save1 = save_embedding(df_item, item_columns, ITEM_EMBEDDING_PATH)
    representation_save2 = save_embedding(df_user, user_columns, USER_EMBEDDING_PATH)
    representation_save3 = save_embedding(df_item_val, item_columns, ITEM_VAL_EMBEDDING_PATH)
    representation_save4 = save_embedding(df_user_val, user_columns, USER_VAL_EMBEDDING_PATH)

    logzero.logger.info(f"user_model and its parameters have been saved in {MODEL_SAVE_PATH}")

    # %% 7. Upload logs

    REMOTE_ROOT = "/root/Counterfactual_IRS"
    LOCAL_PATH = logger_path
    REMOTE_PATH = os.path.join(REMOTE_ROOT, os.path.dirname(LOCAL_PATH))

    # my_upload(LOCAL_PATH, REMOTE_PATH, REMOTE_ROOT)


sigmoid = nn.Sigmoid()


def process_logit(y_deepfm_pos, score, alpha_u=None, beta_i=None, args=None):
    if alpha_u is not None:
        score_new = score * alpha_u * beta_i
        loss_ab = ((alpha_u - 1) ** 2).mean() + ((beta_i - 1) ** 2).mean()
    else:
        score_new = score
        loss_ab = 0
    loss_ab = args.lambda_ab * loss_ab
    y_weighted = 1 / (1 + score_new) * y_deepfm_pos
    return y_weighted, loss_ab


def loss_pointwise_negative(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None):
    y_weighted, loss_ab = process_logit(y_deepfm_pos, score, alpha_u=alpha_u, beta_i=beta_i, args=args)

    loss_y = ((y_weighted - y) ** 2).sum()
    loss_y_neg = ((y_deepfm_neg - 0) ** 2).sum()

    loss = loss_y + loss_y_neg + loss_ab
    return loss


def loss_pointwise(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None):
    y_weighted, loss_ab = process_logit(y_deepfm_pos, score, alpha_u=alpha_u, beta_i=beta_i, args=args)
    loss_y = ((y_weighted - y) ** 2).sum()

    loss = loss_y + loss_ab
    return loss


def loss_pairwise(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None):
    y_weighted, loss_ab = process_logit(y_deepfm_pos, score, alpha_u=alpha_u, beta_i=beta_i, args=args)
    # loss_y = ((y_exposure - y) ** 2).sum()
    bpr_click = - sigmoid(y_weighted - y_deepfm_neg).log().sum()
    loss = bpr_click + loss_ab

    return loss


def loss_pairwise_pointwise(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None):
    y_weighted, loss_ab = process_logit(y_deepfm_pos, score, alpha_u=alpha_u, beta_i=beta_i, args=args)

    loss_y = ((y_weighted - y) ** 2).sum()
    bpr_click = - sigmoid(y_weighted - y_deepfm_neg).log().sum()
    loss = loss_y + args.bpr_weight * bpr_click + loss_ab
    return loss

# TODO: add PD/IPS loss function


def loss_pairwise_PD(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None):
    score_pos = score[:, 0]
    score_neg = score[:, 1]
    bpr_click = - sigmoid(y_deepfm_pos * score_pos -
                          y_deepfm_neg * score_neg).log().sum()
    loss = bpr_click

    return loss


def loss_pairpoint_PD(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None):
    score_pos = score[:, 0]
    score_neg = score[:, 1]
    bpr_click = - sigmoid(y_deepfm_pos * score_pos -
                          y_deepfm_neg * score_neg).log().sum()

    loss_y = ((y_deepfm_pos - y) ** 2).sum()

    loss = loss_y + args.bpr_weight * bpr_click

    return loss


def loss_pointwise_IPS(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None):
    loss_y = (((y_deepfm_pos - y)**2)*score).mean()
    loss = loss_y
    return loss


def loss_pairwise_IPS(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None):
    loss_bpr = - (sigmoid(y_deepfm_pos - y_deepfm_neg).log() * score).mean()
    loss = loss_bpr
    return loss


def loss_pairpoint_IPS(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None):
    loss_y = (((y_deepfm_pos - y)**2)*score).mean()
    loss_bpr = - (sigmoid(y_deepfm_pos - y_deepfm_neg).log() * score).mean()
    loss = loss_y + loss_bpr
    return loss

def loss_pairwise_DICE(y, y_deepfm_pos, y_deepfm_neg,
                       y_deepfm_pos_int, y_deepfm_neg_int,
                       y_deepfm_pos_con, y_deepfm_neg_con, score, args=None):
    bpr_click = - sigmoid(y_deepfm_pos - y_deepfm_neg).log().mean()

    # score: whether pos is popular than neg
    mask_con = torch.where(score, torch.ones_like(score), -torch.ones_like(score))
    mask_int = ~score
    bpr_con = - (sigmoid(y_deepfm_pos_con - y_deepfm_neg_con).log() * mask_con).mean()
    bpr_int = - (sigmoid(y_deepfm_pos_int - y_deepfm_neg_int).log() * mask_int).mean()
    loss = bpr_click + bpr_con + bpr_int
    return loss

def loss_pairpoint_DICE(y, y_deepfm_pos, y_deepfm_neg,
                       y_deepfm_pos_int, y_deepfm_neg_int,
                       y_deepfm_pos_con, y_deepfm_neg_con, score, args=None):
    loss_y = ((y_deepfm_pos - y) ** 2).mean()
    bpr_click = - sigmoid(y_deepfm_pos - y_deepfm_neg).log().mean()

    # score: consider popularity, pos > neg => 1, pos < neg => -1
    mask_con = score
    mask_int = score < 0
    bpr_con = - (sigmoid(y_deepfm_pos_con - y_deepfm_neg_con).log() * mask_con).mean()
    bpr_int = - (sigmoid(y_deepfm_pos_int - y_deepfm_neg_int).log() * mask_int).mean()
    loss = loss_y + bpr_click + bpr_con + bpr_int
    return loss