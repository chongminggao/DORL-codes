# -*- coding: utf-8 -*-
import argparse
import functools

import os

import sys

from core.evaluation.evaluator import test_static_model_in_RL_env
from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
from run_worldModel import save_world_model, setup_world_model, prepare_dataset, prepare_dir_log, get_args_all

sys.path.extend(["./src", "./src/DeepCTR-Torch"])


from logzero import logger


from util.utils import LoggerCallback_Update

CODEPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(CODEPATH, "environments", "KuaiRand_Pure", "data")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='KuaiRand-v0')
    parser.add_argument("--yfeat", type=str, default='is_click')
    parser.add_argument("--loss", type=str, default='point')
    parser.add_argument('--rating_threshold', default=1, type=float)

    parser.add_argument('--is_binarize', dest='is_binarize', action='store_true')
    parser.add_argument('--no_binarize', dest='is_binarize', action='store_false')
    parser.set_defaults(is_binarize=True)

    parser.add_argument('--is_userinfo', dest='is_userinfo', action='store_true')
    parser.add_argument('--no_userinfo', dest='is_userinfo', action='store_false')
    parser.set_defaults(is_userinfo=False)

    args = parser.parse_known_args()[0]
    return args


def main(args):
    # %% 1. Prepare dir
    MODEL_SAVE_PATH, logger_path = prepare_dir_log(args)

    # %% 4. Setup RL environment
    filename = ""
    if args.yfeat == "is_click":
        filename = "kuairand_is_click.csv"
    elif args.yfeat == "is_like":
        filename = "kuairand_is_like.csv"
    elif args.yfeat == "long_view":
        filename = "kuairand_long_view.csv"
    elif args.yfeat == "watch_ratio_normed":
        filename = "kuairand_watchratio.csv"

    filepath_GT = os.path.join(DATAPATH, "..", "MF_results_GT", filename)
    mat, df_item, mat_distance = KuaiRandEnv.load_mat(filepath_GT)
    kwargs_um = {"mat": mat,
                 "df_item": df_item,
                 "mat_distance": mat_distance,
                 "num_leave_compute": args.num_leave_compute,
                 "leave_threshold": args.leave_threshold,
                 "max_turn": args.max_turn}

    env = KuaiRandEnv(**kwargs_um)

    # %% 2. Prepare dataset
    user_features = ["user_id", 'user_active_degree', 'is_live_streamer', 'is_video_author', 'follow_user_num_range',
                     'fans_user_num_range', 'friend_user_num_range', 'register_days_range'] \
                    + [f'onehot_feat{x}' for x in range(18)]
    if not args.is_userinfo:
        user_features = ["user_id"]  # TODO!!!!
    item_features = ["item_id"] + ["feat" + str(i) for i in range(3)] + ["duration_normed"]
    reward_features = [args.yfeat]
    dataset_train, dataset_val, df_user, df_item, df_user_val, df_item_val, x_columns, y_columns, ab_columns = \
        prepare_dataset(args, user_features, item_features, reward_features, MODEL_SAVE_PATH, DATAPATH)

    # %% 3. Setup model
    task = "regression" if args.yfeat == "watch_ratio_normed" else "binary"
    task_logit_dim = 1
    is_ranking = True
    user_model = setup_world_model(args, x_columns, y_columns, ab_columns, task, task_logit_dim, is_ranking)

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

    save_world_model(args, user_model, dataset_train, dataset_val, x_columns, df_user, df_item, df_user_val,
                     df_item_val,
                     user_features, item_features, model_parameters, MODEL_SAVE_PATH, logger_path)


if __name__ == '__main__':
    args_all = get_args_all()
    args = get_args()
    args_all.__dict__.update(args.__dict__)
    main(args_all)
