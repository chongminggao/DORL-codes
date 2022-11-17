# -*- coding: utf-8 -*-
# @Time    : 2022/9/30 20:23
# @Author  : Chongming GAO
# @FileName: run_worldModel.py
import argparse
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
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from environments.KuaiRec.env.KuaiEnv import compute_exposure_effect_kuaiRec


sys.path.extend(["./src", "./src/DeepCTR-Torch"])
from core.user_model_ensemble import EnsembleModel
from core.evaluation.metrics import get_ranking_results
from core.inputs import SparseFeatP, input_from_feature_columns
from core.static_dataset import StaticDataset
from core.util import negative_sampling
from deepctr_torch.inputs import DenseFeat, build_input_features, combined_dnn_input

from util.utils import create_dir


def get_args_all():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action="store_true")

    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument("--bpr_weight", type=float, default=0.5)
    parser.add_argument('--neg_K', default=5, type=int)
    parser.add_argument('--n_models', default=5, type=int)


    # recommendation related:
    # parser.add_argument('--not_softmax', action="store_false")
    parser.add_argument('--is_softmax', dest='is_softmax', action='store_true')
    parser.add_argument('--no_softmax', dest='is_softmax', action='store_false')
    parser.set_defaults(is_softmax=False)

    parser.add_argument("--loss", type=str, default='pointneg')
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

    parser.add_argument("--dnn_activation", type=str, default="relu")
    parser.add_argument("--feature_dim", type=int, default=8)
    parser.add_argument("--entity_dim", type=int, default=8)
    parser.add_argument("--user_model_name", type=str, default="DeepFM")
    parser.add_argument('--dnn', default=(128, 128), type=int, nargs="+")
    parser.add_argument('--dnn_var', default=(), type=int, nargs="+")
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--cuda', default=0, type=int)

    # exposure parameters:
    parser.add_argument('--tau', default=0, type=float)

    parser.add_argument('--is_ab', dest='is_ab', action='store_true')
    parser.add_argument('--no_ab', dest='is_ab', action='store_false')
    parser.set_defaults(is_ab=False)
    parser.add_argument("--message", type=str, default="UM")

    args = parser.parse_known_args()[0]
    return args


def get_xy_columns(args, df_data, df_user, df_item, user_features, item_features, entity_dim, feature_dim):
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

    x_columns, y_columns, ab_columns = get_xy_columns(args, df_train, df_user, df_item, user_features, item_features,
                                                      entity_dim, feature_dim)

    neg_in_train = True if args.env == "KuaiRand-v0" and reward_features[0] != "watch_ratio_normed" else False
    neg_in_train = False  # todo: test for kuairand

    df_pos, df_neg = negative_sampling(df_train, df_item, df_user, reward_features[0],
                                       is_rand=True, neg_in_train=neg_in_train, neg_K=args.neg_K)

    df_x = df_pos[user_features + item_features]
    if reward_features[0] == "hybrid":  # for kuairand
        a = df_pos["long_view"] + df_pos["is_like"] + df_pos["is_click"]
        df_y = a > 0
        df_y = pd.DataFrame(df_y, dtype=int, columns=["hybrid"])
        df_pos["hybrid"] = df_y["hybrid"]
    else:
        df_y = df_pos[reward_features]

    df_x_neg = df_neg[user_features + item_features]
    df_x_neg = df_x_neg.rename(columns={k: k + "_neg" for k in df_x_neg.columns.to_numpy()})

    df_x_all = pd.concat([df_x, df_x_neg], axis=1)

    if tau == 0:
        exposure_pos = np.zeros([len(df_x_all), 1])
    else:
        timestamp = df_pos['timestamp']
        exposure_pos = compute_exposure_effect_kuaiRec(df_x, timestamp, list_feat, tau, MODEL_SAVE_PATH, DATAPATH)

    dataset = StaticDataset(x_columns, y_columns, num_workers=4)
    dataset.compile_dataset(df_x_all, df_y, exposure_pos)

    return dataset, df_user, df_item, x_columns, y_columns, ab_columns


def construct_complete_val_x(dataset_val, df_user, df_item, user_features, item_features):
    user_ids = np.unique(dataset_val.x_numpy[:, dataset_val.user_col].astype(int))
    item_ids = np.unique(dataset_val.x_numpy[:, dataset_val.item_col].astype(int))

    df_user_complete = pd.DataFrame(
        df_user.loc[user_ids].reset_index()[user_features].to_numpy().repeat(len(item_ids), axis=0),
        columns=df_user.reset_index()[user_features].columns)
    df_item_complete = pd.DataFrame(np.tile(df_item.loc[item_ids].reset_index()[item_features], (len(user_ids), 1)),
                                    columns=df_item.loc[item_ids].reset_index()[item_features].columns)

    df_x_complete = pd.concat([df_user_complete, df_item_complete], axis=1)
    return df_x_complete


def get_one_predicted_res(model, df_x_complete, test_loader, steps_per_epoch):
    mean_all = []
    var_all = []
    with torch.no_grad():
        for _, (x, y) in tqdm(enumerate(test_loader), total=steps_per_epoch, desc="Predicting data..."):
            x = x.to(model.user_models[0].device).float()
            mean, var = model.user_models[0].forward(x)
            y_pred = mean.cpu().data.numpy()  # .squeeze()
            var_all_cat = var.cpu().data.numpy()
            mean_all.append(y_pred)
            var_all.append(var_all_cat)
    mean_all_cat = np.concatenate(mean_all).astype("float64").reshape([-1])
    var_all_cat = np.concatenate(var_all).astype("float64").reshape([-1])

    # user_ids = np.sort(df_x_complete["user_id"].unique())

    num_user = len(df_x_complete["user_id"].unique())
    num_item = len(df_x_complete["item_id"].unique())

    if num_user != df_x_complete["user_id"].max() + 1:
        assert num_item != df_x_complete["item_id"].max() + 1
        lbe_user = LabelEncoder()
        lbe_item = LabelEncoder()

        lbe_user.fit(df_x_complete["user_id"])
        lbe_item.fit(df_x_complete["item_id"])

        mean_mat = csr_matrix(
            (mean_all_cat, (lbe_user.transform(df_x_complete["user_id"]), lbe_item.transform(df_x_complete["item_id"]))),
            shape=(num_user, num_item)).toarray()

        var_mat = csr_matrix(
            (var_all_cat, (lbe_user.transform(df_x_complete["user_id"]), lbe_item.transform(df_x_complete["item_id"]))),
            shape=(num_user, num_item)).toarray()
    else:
        assert num_item == df_x_complete["item_id"].max() + 1
        mean_mat = csr_matrix(
            (mean_all_cat, (df_x_complete["user_id"], df_x_complete["item_id"])),
            shape=(num_user, num_item)).toarray()

        var_mat = csr_matrix(
            (var_all_cat, (df_x_complete["user_id"], df_x_complete["item_id"])),
            shape=(num_user, num_item)).toarray()

    # minn = mean_mat.min()
    # maxx = mean_mat.max()
    # normed_mat = (mean_mat - minn) / (maxx - minn)
    # return normed_mat

    return mean_mat, var_mat


def compute_mean_var(ensemble_models, dataset_val, df_user, df_item, user_features, item_features, x_columns,
                     y_columns):
    df_x_complete = construct_complete_val_x(dataset_val, df_user, df_item, user_features, item_features)
    n_user, n_item = df_x_complete[["user_id", "item_id"]].nunique()

    print("predict all users' rewards on all items")

    dataset_um = StaticDataset(x_columns, y_columns, num_workers=4)
    dataset_um.compile_dataset(df_x_complete, pd.DataFrame(np.zeros([len(df_x_complete), 1]), columns=["y"]))

    sample_num = len(dataset_um)
    batch_size = 10000
    steps_per_epoch = (sample_num - 1) // batch_size + 1

    test_loader = DataLoader(dataset=dataset_um.get_dataset_eval(), shuffle=False, batch_size=batch_size,
                             num_workers=dataset_um.num_workers)

    mean_mat_list, var_mat_list = [], []
    for model in ensemble_models.user_models:
        mean_mat, var_mat = get_one_predicted_res(model, df_x_complete, test_loader, steps_per_epoch)
        mean_mat_list.append(mean_mat)
        var_mat_list.append(var_mat)

    return mean_mat_list, var_mat_list



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

    x_columns, y_columns, ab_columns = get_xy_columns(args, df_val, df_user_val, df_item_val, user_features,
                                                      item_features,
                                                      entity_dim, feature_dim)

    dataset_val = StaticDataset(x_columns, y_columns, num_workers=4)
    dataset_val.compile_dataset(df_x, df_y)

    dataset_val.set_df_item_val(df_item_val)
    dataset_val.set_df_user_val(df_user_val)

    assert dataset_val.x_columns[0].name == "user_id"
    dataset_val.set_user_col(0)
    assert dataset_val.x_columns[len(user_features)].name == "item_id"
    dataset_val.set_item_col(len(user_features))

    if not any(df_y.to_numpy() % 1):  # 整数
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

    ensemble_models = EnsembleModel(args.n_models, x_columns, y_columns, task, task_logit_dim,
                                    dnn_hidden_units=args.dnn, dnn_hidden_units_var=args.dnn_var,
                                    seed=args.seed, l2_reg_dnn=args.l2_reg_dnn,
                                    device=device, ab_columns=ab_columns,
                                    dnn_activation=args.dnn_activation, init_std=0.001)

    if args.loss == "pair":
        loss_fun = loss_pairwise
    if args.loss == "point":
        loss_fun = loss_pointwise
    if args.loss == "pointneg":
        loss_fun = loss_pointwise_negative
    if args.loss == "pointpair" or args.loss == "pairpoint" or args.loss == "pp":
        loss_fun = loss_pairwise_pointwise

    ensemble_models.compile(optimizer=args.optimizer,
                            # loss_dict=task_loss_dict,
                            loss_func=functools.partial(loss_fun, args=args),
                            metric_fun={
                                "MAE": lambda y, y_predict: nn.functional.l1_loss(torch.from_numpy(y).type(torch.float),
                                                                                  torch.from_numpy(y_predict)).numpy(),
                                "MSE": lambda y, y_predict: nn.functional.mse_loss(
                                    torch.from_numpy(y).type(torch.float),
                                    torch.from_numpy(y_predict)).numpy(),
                                "RMSE": lambda y, y_predict: nn.functional.mse_loss(
                                    torch.from_numpy(y).type(torch.float),
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

    return ensemble_models



def get_detailed_path(Path_old, num):
    path_list = Path_old.split(".")
    assert len(path_list) >= 2
    filename = path_list[-2]

    path_list_new = path_list[:-2] + [filename + f"_M{num}"] + path_list[-1:]
    Path_new = ".".join(path_list_new)
    return Path_new


# %% 6. Save model
def save_world_model(args, ensemble_models, dataset_val, x_columns, y_columns, df_user, df_item, df_user_val,
                     df_item_val,
                     user_features, item_features, model_parameters, MODEL_SAVE_PATH, logger_path):
    MODEL_MAT_PATH = os.path.join(MODEL_SAVE_PATH, "mats", f"[{args.message}]_mat.pickle") # todo: deprecated
    MODEL_PARAMS_PATH = os.path.join(MODEL_SAVE_PATH, "params", f"[{args.message}]_params.pickle")
    MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "models", f"[{args.message}]_model.pt")
    MODEL_EMBEDDING_PATH = os.path.join(MODEL_SAVE_PATH, "embeddings", f"[{args.message}]_emb.pt")
    USER_EMBEDDING_PATH = os.path.join(MODEL_SAVE_PATH, "embeddings", f"[{args.message}]_emb_user.pt")
    ITEM_EMBEDDING_PATH = os.path.join(MODEL_SAVE_PATH, "embeddings", f"[{args.message}]_emb_item.pt")
    USER_VAL_EMBEDDING_PATH = os.path.join(MODEL_SAVE_PATH, "embeddings", f"[{args.message}]_emb_user_val.pt")
    ITEM_VAL_EMBEDDING_PATH = os.path.join(MODEL_SAVE_PATH, "embeddings", f"[{args.message}]_emb_item_val.pt")

    # (1) Compute and save Mat
    # # todo：暂时不需要
    # mean_mat_list, var_mat_list = compute_mean_var(ensemble_models, dataset_val, df_user, df_item, user_features, item_features,
    #                               x_columns, y_columns)
    # with open(MODEL_MAT_PATH, "wb") as f:
    #     pickle.dump(normed_mat, f)

    # (2) Save params
    with open(MODEL_PARAMS_PATH, "wb") as output_file:
        pickle.dump(model_parameters, output_file)

    # (3) Save Model
    #  To cpu

    # model = user_model.cpu()
    # model.linear_model.device = "cpu"
    # model.linear.device = "cpu"
    #
    # torch.save(model.state_dict(), MODEL_PATH)


    for i, model in enumerate(ensemble_models.user_models):
        MODEL_PATH_new = get_detailed_path(MODEL_PATH, i)

        model = model.cpu()
        model.linear_model.device = "cpu"
        model.linear.device = "cpu"
        torch.save(model.state_dict(), MODEL_PATH_new)

    # (4) Save Embedding
    # torch.save(model.embedding_dict.state_dict(), MODEL_EMBEDDING_PATH)

    def save_embedding(model, df_save, columns, SAVEPATH):
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

    for i, model in enumerate(ensemble_models.user_models):
        ITEM_EMBEDDING_PATH_new = get_detailed_path(ITEM_EMBEDDING_PATH, i)
        USER_EMBEDDING_PATH_new = get_detailed_path(USER_EMBEDDING_PATH, i)
        ITEM_VAL_EMBEDDING_PATH_new = get_detailed_path(ITEM_VAL_EMBEDDING_PATH, i)
        USER_VAL_EMBEDDING_PATH_new = get_detailed_path(USER_VAL_EMBEDDING_PATH, i)

        representation_save1 = save_embedding(model, df_item, item_columns, ITEM_EMBEDDING_PATH_new)
        representation_save2 = save_embedding(model, df_user, user_columns, USER_EMBEDDING_PATH_new)
        representation_save3 = save_embedding(model, df_item_val, item_columns, ITEM_VAL_EMBEDDING_PATH_new)
        representation_save4 = save_embedding(model, df_user_val, user_columns, USER_VAL_EMBEDDING_PATH_new)

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


def loss_pointwise_negative(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None, log_var_neg=None):
    y_weighted, loss_ab = process_logit(y_deepfm_pos, score, alpha_u=alpha_u, beta_i=beta_i, args=args)


    if log_var is not None:
        inv_var = torch.exp(-log_var)
        inv_var_neg = torch.exp(-log_var_neg)
        loss_var_pos = log_var.sum()
        loss_var_neg = log_var_neg.sum()
    else:
        inv_var = 1
        inv_var_neg = 1
        loss_var_pos = 0
        loss_var_neg = 0


    loss_y = (((y_weighted - y) ** 2) * inv_var).sum()
    loss_y_neg = (((y_deepfm_neg - 0) ** 2) * inv_var_neg).sum()

    loss = loss_y + loss_y_neg + loss_ab + loss_var_pos + loss_var_neg
    return loss


def loss_pointwise(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None, log_var_neg=None):
    y_weighted, loss_ab = process_logit(y_deepfm_pos, score, alpha_u=alpha_u, beta_i=beta_i, args=args)

    if log_var is not None:
        inv_var = torch.exp(-log_var)
        loss_var_pos = log_var.sum()
    else:
        inv_var = 1
        loss_var_pos = 0

    loss_y = (((y_weighted - y) ** 2) * inv_var).sum()

    loss = loss_y + loss_ab + loss_var_pos

    return loss


def loss_pairwise(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None, log_var_neg=None):
    y_weighted, loss_ab = process_logit(y_deepfm_pos, score, alpha_u=alpha_u, beta_i=beta_i, args=args)
    # loss_y = ((y_exposure - y) ** 2).sum()

    bpr_click = - sigmoid(y_weighted - y_deepfm_neg).log().sum()
    loss = bpr_click + loss_ab

    return loss


def loss_pairwise_pointwise(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None, log_var_neg=None):
    y_weighted, loss_ab = process_logit(y_deepfm_pos, score, alpha_u=alpha_u, beta_i=beta_i, args=args)
    if log_var is not None:
        inv_var = torch.exp(-log_var)
        loss_var_pos = log_var.sum()
    else:
        inv_var = 1
        loss_var_pos = 0
    loss_y = (((y_weighted - y) ** 2) * inv_var).sum()
    bpr_click = - sigmoid(y_weighted - y_deepfm_neg).log().sum()
    loss = loss_y + args.bpr_weight * bpr_click + loss_ab + loss_var_pos
    return loss
