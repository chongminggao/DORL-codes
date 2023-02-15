# -*- coding: utf-8 -*-

import argparse
import functools
import os
import random
import sys
import traceback

import logzero
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from torch import nn

from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
from results_for_paper.get_discrepency import get_percentage
from run_worldModel_ensemble import get_datapath, prepare_dir_log, prepare_dataset, get_task, get_args_all, \
    get_args_dataset_specific

sys.path.extend(["./src", "./src/DeepCTR-Torch"])
from core.evaluation.evaluator import test_static_model_in_RL_env
from core.configs import get_common_args, get_features, get_true_env, \
    get_training_item_domination
from core.user_model_ensemble import EnsembleModel
from core.evaluation.metrics import get_ranking_results

from util.utils import LoggerCallback_Update

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


########################
## For paper: run on lab5: python get_recommended_categories.py --cuda 7 -seed 1 --epoch 4 --env KuaiRand-v0 --top_visual 0.001  --message "s1_tf0.001" &
## For paper: run on lab5: python get_recommended_categories.py --cuda 7 -seed 1 --epoch 10 --env KuaiRand-v0 --top_visual_user 10  --message "s1_peruser10" &

def get_args_epsilonGreedy():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_model_name", type=str, default="EpsilonGreedy")
    parser.add_argument('--epsilon', default=0, type=float)
    parser.add_argument('--n_models', default=1, type=int)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument("--message", type=str, default="epsilon-greedy")
    parser.add_argument('--top_visual', default=0.2, type=float)
    parser.add_argument('--top_visual_user', default=10, type=int)
    args = parser.parse_known_args()[0]
    return args


def draw_cat(res_train, res_test, res_predicted, num, epoch, message):
    df = pd.DataFrame({k + 1: [x, y, z] for k, (x, y, z) in enumerate(zip(res_test, res_train, res_predicted))})
    df["domain"] = ["Test set", "Training set", "Recommended"]
    ######
    # df = pd.read_csv("results_for_paper/figures/recommended_category_kuairand/feat_KuaiRand_s1_tf0.001_e3.csv")

    colors = sns.color_palette("muted", n_colors=num)
    colors = sns.color_palette("Set2", n_colors=num)
    colors1 = sns.color_palette("pastel", n_colors=num)
    colors2 = sns.color_palette(n_colors=num)
    colors = sns.color_palette("Paired", n_colors=num * 2 + 12)

    fig = plt.figure(figsize=(5, 1))
    hatchs = ["////", "\\\\\\\\"]

    for i in range(num, 0, -1):
        # print(i)
        # sns.barplot(x=i, y="domain", data=df, color=colors[2 * i + 10], hatch=hatchs[i%len(hatchs)], edgecolor=colors[2 * i - 1], lw=0.5)
        # plt.rcParams['hatch.color'] = colors[2 * i - 1 + 10]

        sns.barplot(x=i, y="domain", data=df, color=colors1[i - 1], hatch=hatchs[i % len(hatchs)],
                    edgecolor=colors2[i - 1], lw=0.4)

    # plt.xlabel(featname)
    plt.xlabel(None)
    plt.ylabel(None)
    # plt.axis("off")
    plt.xticks([])
    dirpath = os.path.join("results_for_paper", "figures", "recommended_category_kuairand")
    pdfpath = os.path.join(dirpath, f'feat_KuaiRand_{message}_e{epoch}.pdf')
    csv_path = os.path.join(dirpath, f'feat_KuaiRand_{message}_e{epoch}.csv')

    plt.savefig(pdfpath, bbox_inches='tight', pad_inches=0)
    df.to_csv(csv_path, index=False)
    # plt.savefig(os.path.join(CODEPATH, f'feat_{envname}_{featname}.png'), bbox_inches='tight',
    #             pad_inches=0)
    # plt.show()
    plt.close()


def test_whether_dominate(xy_predict, df_true_list, res_train, res_test, df_feat, sorted_index, epoch,
                          top_visual, message):
    # top_visual = 0.5
    pos_items_predicted = xy_predict.sort_values(["y_pred"], ascending=False)[:int(top_visual * len(xy_predict))][
        "item_id"]
    feat_predicted = df_feat.loc[pos_items_predicted].to_numpy()
    cats_predicted = feat_predicted.reshape(-1)
    pos_cat_predicted = cats_predicted[cats_predicted > 0]

    cat_set = sorted_index
    res_predicted = get_percentage(pos_cat_predicted, cat_set, is_test=False, sorted_idx=cat_set)
    draw_cat(res_train, res_test, res_predicted, len(cat_set), epoch, message)
    # draw_cat(res_train, res_test, res_predicted, len(cat_set), epoch)
    return dict()


def test_whether_dominate_per_user(xy_predict, df_true_list, res_train, res_test, df_feat, sorted_index, epoch,
                                   message, top_visual_user=10):
    # top_visual = 0.5

    pos_items_predicted = []
    groups = xy_predict.groupby("user_id")
    for user, group in groups:
        top_items = group.sort_values(["y_pred"], ascending=False).iloc[:top_visual_user]["item_id"]
        pos_items_predicted.extend(top_items)

    # pos_items_predicted = xy_predict.sort_values(["y_pred"], ascending=False)[:int(top_visual * len(xy_predict))]["item_id"]

    feat_predicted = df_feat.loc[pos_items_predicted].to_numpy()
    cats_predicted = feat_predicted.reshape(-1)
    pos_cat_predicted = cats_predicted[cats_predicted > 0]

    cat_set = sorted_index
    res_predicted = get_percentage(pos_cat_predicted, cat_set, is_test=False, sorted_idx=cat_set)
    draw_cat(res_train, res_test, res_predicted, len(cat_set), epoch, message)
    # draw_cat(res_train, res_test, res_predicted, len(cat_set), epoch)
    return dict()


def setup_world_model(args, x_columns, y_columns, ab_columns, task, task_logit_dim, is_ranking, MODEL_SAVE_PATH):
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    random.seed(args.seed)

    ensemble_models = EnsembleModel(args.n_models, args.message, MODEL_SAVE_PATH, x_columns, y_columns, task,
                                    task_logit_dim,
                                    dnn_hidden_units=args.dnn, dnn_hidden_units_var=args.dnn_var,
                                    seed=args.seed, l2_reg_dnn=args.l2_reg_dnn,
                                    device=device, ab_columns=ab_columns,
                                    dnn_activation=args.dnn_activation, init_std=0.001)

    if args.loss == "pair":
        loss_fun = loss_pairwise_Standard
    if args.loss == "point":
        loss_fun = loss_pointwise_Standard
    if args.loss == "pointneg":
        loss_fun = loss_pointwise_negative_Standard
    if args.loss == "pointpair" or args.loss == "pairpoint" or args.loss == "pp":
        loss_fun = loss_pairwise_pointwise_Standard

    sorted_index = pd.read_csv("results_for_paper/majority_indices.csv").to_numpy().squeeze()
    res_train = pd.read_csv("results_for_paper/majority_train_kuairand.csv").to_numpy().squeeze()
    res_test = pd.read_csv("results_for_paper/majority_test_kuairand.csv").to_numpy().squeeze()

    list_feat, df_feat = KuaiRandEnv.load_category()
    myfunc = functools.partial(test_whether_dominate, res_train=res_train, res_test=res_test, df_feat=df_feat,
                               sorted_index=sorted_index, top_visual=args.top_visual,
                               message=args.message)
    myfunc = functools.partial(test_whether_dominate_per_user, res_train=res_train, res_test=res_test, df_feat=df_feat,
                               sorted_index=sorted_index, top_visual_user=args.top_visual_user,
                               message=args.message)

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
                            metric_fun_ranking=myfunc,
                            # functools.partial(get_ranking_results, K=args.rankingK,
                            #                   metrics=["Recall", "Precision", "NDCG", "HT", "MAP", "MRR"]
                            #                   ) if is_ranking else None,
                            metrics=None)

    return ensemble_models


sigmoid = nn.Sigmoid()


def loss_pointwise_negative_Standard(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None,
                                     log_var=None, log_var_neg=None):
    loss_y = (((y_deepfm_pos - y) ** 2)).sum()
    loss_y_neg = (((y_deepfm_neg - 0) ** 2)).sum()

    loss = loss_y + loss_y_neg
    return loss


def loss_pointwise_Standard(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None,
                            log_var_neg=None):
    loss_y = (((y_deepfm_pos - y) ** 2)).sum()

    loss = loss_y

    return loss


def loss_pairwise_Standard(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None,
                           log_var_neg=None):
    bpr_click = - (sigmoid(y_deepfm_pos - y_deepfm_neg).log()).sum()
    loss = bpr_click

    return loss


def loss_pairwise_pointwise_Standard(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None,
                                     log_var=None, log_var_neg=None):
    loss_y = (((y_deepfm_pos - y) ** 2)).sum()
    bpr_click = - (sigmoid(y_deepfm_pos - y_deepfm_neg).log()).sum()
    loss = loss_y + args.bpr_weight * bpr_click
    return loss


def main(args, is_save=False):
    # %% 1. Prepare dir
    DATAPATH = get_datapath(args.env)
    args = get_common_args(args)
    MODEL_SAVE_PATH, logger_path = prepare_dir_log(args)

    # %% 2. Prepare dataset
    user_features, item_features, reward_features = get_features(args.env, args.is_userinfo)

    dataset_train, dataset_val, df_user, df_item, df_user_val, df_item_val, x_columns, y_columns, ab_columns = \
        prepare_dataset(args, user_features, item_features, reward_features, MODEL_SAVE_PATH, DATAPATH)

    # %% 3. Setup model
    task, task_logit_dim, is_ranking = get_task(args.env, args.yfeat)
    ensemble_models = setup_world_model(args, x_columns, y_columns, ab_columns,
                                        task, task_logit_dim, is_ranking, MODEL_SAVE_PATH)

    env, env_task_class, kwargs_um = get_true_env(args, read_user_num=None)

    item_feat_domination = get_training_item_domination(args.env)
    ensemble_models.compile_RL_test(
        functools.partial(test_static_model_in_RL_env, env=env, dataset_val=dataset_val, is_softmax=args.is_softmax,
                          epsilon=args.epsilon, is_ucb=args.is_ucb, need_transform=args.need_transform,
                          num_trajectory=args.num_trajectory, item_feat_domination=item_feat_domination,
                          force_length=args.force_length, top_rate=args.top_rate, draw_bar=False)) # draw_bar = Flase here

    # %% 5. Learn and evaluate model

    history_list = ensemble_models.fit_data(dataset_train, dataset_val,
                                            batch_size=args.batch_size, epochs=args.epoch, shuffle=True,
                                            callbacks=[LoggerCallback_Update(logger_path)])

    # %% 6. Save model
    if is_save:
        ensemble_models.get_save_entropy_mat(args.env, args.entropy_window)
        ensemble_models.save_all_models(dataset_val, x_columns, y_columns, df_user, df_item, df_user_val, df_item_val,
                                        user_features, item_features, args.deterministic)


if __name__ == '__main__':
    args_all = get_args_all()
    args = get_args_dataset_specific(args_all.env)
    args_epsilon = get_args_epsilonGreedy()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_epsilon.__dict__)

    args_all.all_item_ranking = True  # Todo: for visual

    try:
        main(args_all, is_save=False)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
