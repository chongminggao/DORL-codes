import argparse
import datetime
import functools
import json
import os
import pprint
import random
import time
import traceback
from collections import defaultdict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import sys

from tqdm import tqdm

from policy_utils import prepare_dir_log, prepare_buffer_via_offline_data

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from core.evaluation.evaluator import Callback_Coverage_Count
from core.layers import Actor_Linear
from core.policy.sqn import SQN
from core.trainer.offline import offline_trainer
from run_Policy_Main import prepare_user_model_and_env
from core.configs import get_training_data, get_true_env, get_common_args
from core.collector2 import Collector
from core.inputs import get_dataset_columns

from core.state_tracker2 import StateTracker_Caser, StateTracker_GRU, StateTracker_SASRec

from tianshou.data import VectorReplayBuffer, Batch
from tianshou.env import DummyVectorEnv

from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net, MLP
from tianshou.utils.net.discrete import Actor, Critic

# from util.upload import my_upload
from util.utils import create_dir, LoggerCallback_Policy, save_model_fn
import logzero
from logzero import logger

try:
    import envpool
except ImportError:
    envpool = None


def get_args_all():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    # parser.add_argument("--user_model_name", type=str, default="DeepFM")
    parser.add_argument("--model_name", type=str, default="SQN")
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--cuda', default=1, type=int)

    parser.add_argument('--cpu', dest='cpu', action='store_true')
    parser.set_defaults(cpu=False)

    parser.add_argument('--is_save', dest='is_save', action='store_true')
    parser.add_argument('--no_save', dest='is_save', action='store_false')
    parser.set_defaults(is_save=False)

    parser.add_argument('--is_exploration_noise', dest='exploration_noise', action='store_true')
    parser.add_argument('--no_exploration_noise', dest='exploration_noise', action='store_false')
    parser.set_defaults(exploration_noise=True)

    # State_tracker
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--window_sqn", type=int, default=10)
    parser.add_argument("--which_tracker", type=str, default="caser")  # in {"caser", "sasrec", "gru"}
    # State_tracker Caser
    parser.add_argument('--filter_sizes', type=int, nargs='*', default=[2, 3, 4])
    parser.add_argument("--num_filters", type=int, default=16)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    # State_tracker SASRec
    # parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--num_heads", type=int, default=1)
    # State_tracker GRU

    # SQN head:
    parser.add_argument('--which_head', type=str, default='qhead')  # in {"shead", "qhead", "bcq"}

    # tianshou
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--epoch', type=int, default=200)
    # parser.add_argument('--step-per-epoch', type=int, default=15000)
    # parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])

    parser.add_argument('--test-num', type=int, default=100)

    parser.add_argument('--render', type=float, default=0)
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--step-per-epoch', type=int, default=1000)

    parser.add_argument('--logdir', type=str, default='log')

    # BCQ
    parser.add_argument("--eps-test", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--unlikely-action-threshold", type=float, default=0.6)
    parser.add_argument("--imitation-logits-penalty", type=float, default=0.01)
    parser.add_argument("--update-per-epoch", type=int, default=5000)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-interval", type=int, default=4)

    # parser.add_argument("--read_message", type=str, default="UM")
    parser.add_argument("--message", type=str, default="BCQ_with_emb")

    args = parser.parse_known_args()[0]
    return args

    # %% 4. Setup model


def setup_policy_model(args, env, buffer, test_envs):
    # ensemble_models, _, _ = prepare_user_model_and_env(args)

    # saved_embedding = ensemble_models.load_val_user_item_embedding(freeze_emb=args.freeze_emb)
    user_columns, action_columns, feedback_columns, have_user_embedding, have_action_embedding, have_feedback_embedding = \
        get_dataset_columns(args.embedding_dim, args.embedding_dim, env.mat.shape[0], env.mat.shape[1],
                            envname=args.env)

    args.action_shape = action_columns[0].vocabulary_size
    args.state_dim = action_columns[0].embedding_dim

    args.max_action = env.action_space.high[0]

    if args.which_tracker.lower() == "caser":
        state_tracker = StateTracker_Caser(user_columns, action_columns, feedback_columns, args.state_dim,
                                           device=args.device,
                                           window_size=args.window_sqn,
                                           filter_sizes=args.filter_sizes, num_filters=args.num_filters,
                                           dropout_rate=args.dropout_rate).to(args.device)
    elif args.which_tracker.lower() == "gru":
        state_tracker = StateTracker_GRU(user_columns, action_columns, feedback_columns, args.state_dim,
                                         device=args.device,
                                         window_size=args.window_sqn).to(args.device)
    elif args.which_tracker.lower() == "sasrec":
        state_tracker = StateTracker_SASRec(user_columns, action_columns, feedback_columns, args.state_dim,
                                            device=args.device, window_size=args.window_sqn,
                                            dropout_rate=args.dropout_rate, num_heads=args.num_heads).to(args.device)

    # state_tracker = state_tracker.to(args.device)

    model_final_layer = Actor_Linear(state_tracker.final_dim, args.action_shape, device=args.device).to(args.device)
    imitation_final_layer = Actor_Linear(state_tracker.final_dim, args.action_shape, device=args.device).to(args.device)

    actor_critic = ActorCritic(model_final_layer, imitation_final_layer)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    policy = SQN(
        model_final_layer,
        imitation_final_layer,
        optim,
        args.gamma,
        args.n_step,
        args.target_update_freq,
        args.eps_test,
        args.unlikely_action_threshold,
        args.imitation_logits_penalty,
        state_tracker=state_tracker,
        buffer=buffer,
        which_head=args.which_head
    )

    # collector
    # buffer has been gathered
    # train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(
        policy, test_envs,
        VectorReplayBuffer(args.buffer_size, len(test_envs)),
        preprocess_fn=state_tracker.build_state,
        exploration_noise=args.exploration_noise,
    )

    return policy, test_collector, state_tracker, optim


def learn_policy(args, policy, buffer, test_collector, state_tracker, optim, MODEL_SAVE_PATH, logger_path):
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.env.replace("-", "_")}_bcq'
    log_path = os.path.join(args.logdir, args.env, 'bcq', log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger1 = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    policy.callbacks = [Callback_Coverage_Count(test_collector), LoggerCallback_Policy(logger_path)]
    model_save_path = os.path.join(MODEL_SAVE_PATH, "{}_{}.pt".format(args.model_name, args.message))

    result = offline_trainer(
        policy,
        buffer,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.test_num,
        args.batch_size,
        save_best_fn=save_best_fn,
        # stop_fn=stop_fn,
        logger=logger1,
        save_model_fn=functools.partial(save_model_fn,
                                        model_save_path=model_save_path,
                                        state_tracker=state_tracker,
                                        optim=optim,
                                        is_save=args.is_save)
    )

    print(__file__)
    pprint.pprint(result)
    logger.info(result)


def main(args):
    # %% 1. Prepare the saved path.
    MODEL_SAVE_PATH, logger_path = prepare_dir_log(args)

    # %% 2. Prepare user model and environment
    env, buffer, test_envs = prepare_buffer_via_offline_data(args)

    # %% 3. Setup policy
    policy, test_collector, state_tracker, optim = setup_policy_model(args, env, buffer, test_envs)

    # %% 4. Learn policy
    learn_policy(args, policy, buffer, test_collector, state_tracker, optim, MODEL_SAVE_PATH, logger_path)


if __name__ == '__main__':
    args_all = get_args_all()
    args = get_common_args(args_all)
    args_all.__dict__.update(args.__dict__)

    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
