# -*- coding: utf-8 -*-
import argparse
import functools

import os

import sys

sys.path.extend(["./src", "./src/DeepCTR-Torch"])
# from environments.coat.env.Coat import CoatEnv, construct_complete_val_x, compute_normed_reward_for_all
from core.evaluation.evaluator import test_static_model_in_RL_env
from environments.coat.env.Coat import CoatEnv
from run_baseline import prepare_dir_log, prepare_dataset, setup_world_model, save_world_model, get_args_all

# from core.evaluation.metrics import get_ranking_results
# from core.inputs import SparseFeatP, input_from_feature_columns
# from core.user_model_pairwise import UserModel_Pairwise
# from core.util import compute_exposure_effect_kuaiRec
# from deepctr_torch.inputs import DenseFeat, build_input_features, combined_dnn_input
# import pandas as pd
# import numpy as np
# from core.user_model import StaticDataset
# import logzero

from logzero import logger

# from environments.coat.env.Coat import CoatEnv, negative_sampling
# from util.upload import my_upload

from util.utils import LoggerCallback_Update

CODEPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(CODEPATH, "environments", "coat")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default='CoatEnv-v0')
    parser.add_argument("--yfeat", type=str, default='rating')
    parser.add_argument("--loss", type=str, default='pp')
    parser.add_argument('--rating_threshold', default=4, type=float)

    parser.add_argument('--is_binarize', dest='is_binarize', action='store_true')
    parser.add_argument('--no_binarize', dest='is_binarize', action='store_false')
    parser.set_defaults(is_binarize=True)

    parser.add_argument('--is_userinfo', dest='is_userinfo', action='store_true')
    parser.add_argument('--no_userinfo', dest='is_userinfo', action='store_false')
    parser.set_defaults(is_userinfo=True)

    args = parser.parse_known_args()[0]
    return args


def main(args):
    # %% 1. Prepare dir
    MODEL_SAVE_PATH, logger_path = prepare_dir_log(args)

    # %% 2. Prepare dataset
    user_features = ["user_id", 'gender_u', 'age', 'location', 'fashioninterest']
    item_features = ['item_id', 'gender_i', "jackettype", 'color', 'onfrontpage']
    reward_features = [args.yfeat]
    dataset_train, dataset_val, df_user, df_item, df_user_val, df_item_val, x_columns, y_columns, ab_columns = \
        prepare_dataset(args, user_features, item_features, reward_features, MODEL_SAVE_PATH, DATAPATH)

    # %% 3. Setup model
    task = "regression"
    task_logit_dim = 1
    is_ranking = True
    user_model = setup_world_model(args, x_columns, y_columns, ab_columns, task, task_logit_dim, is_ranking)

    # %% 4. Setup RL environment
    mat, df_item, mat_distance = CoatEnv.load_mat()
    kwargs_um = {"mat": mat,
                 "df_item": df_item,
                 "mat_distance": mat_distance,
                 "num_leave_compute": args.num_leave_compute,
                 "leave_threshold": args.leave_threshold,
                 "max_turn": args.max_turn}

    env = CoatEnv(**kwargs_um)
    user_model.compile_RL_test(
        functools.partial(test_static_model_in_RL_env, env=env, dataset_val=dataset_val, is_softmax=args.is_softmax,
                          epsilon=args.epsilon, is_ucb=args.is_ucb, need_transform=False))

    # %% 5. Learn and evaluate model
    history = user_model.fit_data(dataset_train, dataset_val,
                                  batch_size=args.batch_size, epochs=args.epoch, shuffle=True,
                                  callbacks=[[LoggerCallback_Update(logger_path)]])
    logger.info(history.history)

    # %% 6. Save model
    model_parameters = {"feature_columns": x_columns, "y_columns": y_columns, "task": task,
                        "task_logit_dim": task_logit_dim, "dnn_hidden_units": args.dnn, "seed": args.seed,
                        "device": "cpu",
                        "ab_columns": ab_columns}

    save_world_model(args, user_model, dataset_train, dataset_val, x_columns, df_user, df_item, df_user_val, df_item_val,
                     user_features, item_features, model_parameters, MODEL_SAVE_PATH, logger_path)


if __name__ == '__main__':
    args_all = get_args_all()
    args = get_args()
    args_all.__dict__.update(args.__dict__)
    main(args_all)
