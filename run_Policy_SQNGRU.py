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

from core.evaluation.evaluator import Callback_Coverage_Count
from core.layers import Actor_Linear
from core.policy.sqn import SQN

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from core.policy.discrete_bcq import DiscreteBCQPolicy_withEmbedding
from core.trainer.offline import offline_trainer
from run_Policy_Main import prepare_user_model_and_env
from core.configs import get_training_data, get_true_env, get_common_args
from core.collector2 import Collector
from core.inputs import get_dataset_columns

from core.state_tracker2 import StateTracker_Caser, StateTracker_GRU

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
    parser.add_argument("--user_model_name", type=str, default="DeepFM")
    parser.add_argument("--model_name", type=str, default="BCQ")
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

    # for state_tracker
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=10)

    # head:
    parser.add_argument('--which_head', type=str, default='shead') # in {"shead", "qhead", "bcq"}

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

    parser.add_argument("--read_message", type=str, default="UM")
    parser.add_argument("--message", type=str, default="BCQ_with_emb")

    args = parser.parse_known_args()[0]
    return args


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
    # df_user_num["item_id"] += 1

    df_user_num_sorted = df_user_num.sort_values("item_id", ascending=False)

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

    df_user_items = df_train[["user_id", "item_id", args.yfeat]].groupby("user_id").agg(list)
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
    df_train = df_train.head(10000)
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

    # %% 4. Setup model


def setup_policy_model(args, env, buffer, test_envs):
    ensemble_models, _, _ = prepare_user_model_and_env(args)

    # saved_embedding = ensemble_models.load_val_user_item_embedding(freeze_emb=args.freeze_emb)
    user_columns, action_columns, feedback_columns, have_user_embedding, have_action_embedding, have_feedback_embedding = \
        get_dataset_columns(args.embedding_dim, args.embedding_dim, env.mat.shape[0], env.mat.shape[1], envname=args.env)

    args.action_shape = action_columns[0].vocabulary_size
    args.state_dim = action_columns[0].embedding_dim

    args.max_action = env.action_space.high[0]

    state_tracker = StateTracker_GRU(user_columns, action_columns, feedback_columns, args.state_dim, device=args.device,
                                     window_size=args.window_size).to(args.device)

    model_final_layer = Actor_Linear(state_tracker.final_dim, args.action_shape, device=args.device)
    imitation_final_layer = Actor_Linear(state_tracker.final_dim, args.action_shape, device=args.device)

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

    result = offline_trainer(
        policy,
        buffer,
        test_collector,
        args.epoch,
        args.update_per_epoch,
        args.test_num,
        args.batch_size,
        # stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
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
