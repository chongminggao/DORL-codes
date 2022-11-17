# -*- coding: utf-8 -*-
# @Time    : 2022/7/19 15:47
# @Author  : Chongming GAO
# @FileName: mytestRL.py

import argparse
import functools

import os

import sys
import traceback

from run_worldModel_ensemble import prepare_dir_log, prepare_dataset, setup_world_model, save_world_model, get_args_all

sys.path.extend(["./src","./src/DeepCTR-Torch"])

from logzero import logger

from environments.KuaiRec.env.KuaiEnv import KuaiEnv
from core.evaluation.evaluator import test_static_model_in_RL_env
from util.utils import create_dir, LoggerCallback_Update

CODEPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(CODEPATH, "environments", "KuaiRec", "data")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='KuaiEnv-v0')
    parser.add_argument("--yfeat", type=str, default='watch_ratio_normed')
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument('--neg_K', default=3, type=int)

    parser.add_argument("--feature_dim", type=int, default=8)
    parser.add_argument("--entity_dim", type=int, default=8)
    parser.add_argument('--batch_size', default=4096, type=int)

    parser.add_argument('--is_binarize', dest='is_binarize', action='store_true')
    parser.add_argument('--no_binarize', dest='is_binarize', action='store_false')
    parser.set_defaults(is_binarize=False)

    parser.add_argument('--is_userinfo', dest='is_userinfo', action='store_true')
    parser.add_argument('--no_userinfo', dest='is_userinfo', action='store_false')
    parser.set_defaults(is_userinfo=False)

    parser.add_argument("--dnn_activation", type=str, default="swish")
    parser.add_argument('--leave_threshold', default=0, type=int) # todo
    parser.add_argument('--num_leave_compute', default=1, type=int) # todo

    args = parser.parse_known_args()[0]
    return args

def main(args):
    # %% 1. Prepare dir
    MODEL_SAVE_PATH, logger_path = prepare_dir_log(args)

    # %% 2. Prepare dataset
    user_features = ["user_id"]
    item_features = ["item_id"] + ["feat" + str(i) for i in range(4)] + ["duration_normed"]
    reward_features = [args.yfeat]
    dataset_train, dataset_val, df_user, df_item, df_user_val, df_item_val, x_columns, y_columns, ab_columns = \
        prepare_dataset(args, user_features, item_features, reward_features, MODEL_SAVE_PATH, DATAPATH)

    # %% 3. Setup model
    task = "regression"
    task_logit_dim = 1
    is_ranking = False
    ensemble_models = setup_world_model(args, x_columns, y_columns, ab_columns, task, task_logit_dim, is_ranking)

    # %% 4. Prepare Envs
    mat, lbe_user, lbe_item, list_feat, df_video_env, df_dist_small = KuaiEnv.load_mat()
    kwargs = {"mat": mat,
              "lbe_user": lbe_user,
              "lbe_item": lbe_item,
              "num_leave_compute": args.num_leave_compute,
              "leave_threshold": args.leave_threshold,
              "max_turn": args.max_turn,
              "list_feat": list_feat,
              "df_video_env": df_video_env,
              "df_dist_small": df_dist_small}
    env = KuaiEnv(**kwargs)
    ensemble_models.compile_RL_test(
        functools.partial(test_static_model_in_RL_env, env=env, dataset_val=dataset_val, is_softmax=args.is_softmax,
                          epsilon=args.epsilon, is_ucb=args.is_ucb, need_transform=True))

    # %% 5. Learn model
    history = ensemble_models.fit_data(dataset_train, dataset_val,
                                  batch_size=args.batch_size, epochs=args.epoch, shuffle=True,
                                  callbacks=[[LoggerCallback_Update(logger_path)]])
    # logger.info(history.history)

    # %% 6. Save model
    model_parameters = {"feature_columns": x_columns, "y_columns": y_columns, "task": task,
                        "task_logit_dim": task_logit_dim, "dnn_hidden_units": args.dnn, "seed": args.seed,
                        "device": "cpu",
                        "ab_columns": ab_columns}

    save_world_model(args, ensemble_models, dataset_val, x_columns, y_columns, df_user, df_item, df_user_val, df_item_val,
                     user_features, item_features, model_parameters, MODEL_SAVE_PATH, logger_path)


if __name__ == '__main__':
    args_all = get_args_all()
    args = get_args()
    args_all.__dict__.update(args.__dict__)
    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logger.error(var)
