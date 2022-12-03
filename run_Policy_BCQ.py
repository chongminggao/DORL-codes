import argparse
import datetime
import functools
import json
import os
import pickle
import pprint
import random
import time
from collections import defaultdict

import gym
import numpy as np
import pandas as pd
# import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

import sys

from tqdm import tqdm

from core.trainer.offline import offline_trainer
from run_Policy2 import prepare_user_model_and_env
from run_worldModel_ensemble import load_dataset_val
from tianshou.policy import BCQPolicy
from tianshou.utils.net.continuous import Perturbation, VAE

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from core.user_model_ensemble import EnsembleModel
from core.configs import get_features, get_training_data, get_true_env, get_val_data
from core.collector2 import Collector
from core.inputs import get_dataset_columns
from core.policy.a2c2 import A2CPolicy_withEmbedding
from core.state_tracker2 import StateTrackerAvg2
from core.trainer.onpolicy import onpolicy_trainer
from core.worldModel.simulated_env import SimulatedEnv

from tianshou.data import VectorReplayBuffer, Batch
from tianshou.env import DummyVectorEnv

from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net, MLP
from tianshou.utils.net.discrete import Actor, Critic

# from util.upload import my_upload
from util.utils import create_dir, LoggerCallback_RL, LoggerCallback_Policy, save_model_fn
import logzero
from logzero import logger

try:
    import envpool
except ImportError:
    envpool = None


def get_args_all():
    parser = argparse.ArgumentParser()

    parser.add_argument("--user_model_name", type=str, default="DeepFM")
    parser.add_argument("--model_name", type=str, default="A2C_with_emb")
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--cuda', default=1, type=int)

    parser.add_argument('--is_ab', dest='is_ab', action='store_true')
    parser.add_argument('--no_ab', dest='is_ab', action='store_false')
    parser.set_defaults(is_ab=True)

    parser.add_argument('--is_userinfo', dest='is_userinfo', action='store_true')
    parser.add_argument('--no_userinfo', dest='is_userinfo', action='store_false')
    parser.set_defaults(is_userinfo=False)

    parser.add_argument('--cpu', dest='cpu', action='store_true')
    parser.set_defaults(cpu=False)

    parser.add_argument('--is_save', dest='is_save', action='store_true')
    parser.add_argument('--no_save', dest='is_save', action='store_false')
    parser.set_defaults(is_save=False)

    parser.add_argument('--is_use_userEmbedding', dest='use_userEmbedding', action='store_true')
    parser.add_argument('--no_use_userEmbedding', dest='use_userEmbedding', action='store_false')
    parser.set_defaults(use_userEmbedding=False)

    parser.add_argument('--is_exploration_noise', dest='exploration_noise', action='store_true')
    parser.add_argument('--no_exploration_noise', dest='exploration_noise', action='store_false')
    parser.set_defaults(exploration_noise=True)
    parser.add_argument('--eps', default=0.1, type=float)

    parser.add_argument('--is_all_item_ranking', dest='is_all_item_ranking', action='store_true')
    parser.add_argument('--no_all_item_ranking', dest='is_all_item_ranking', action='store_false')
    parser.set_defaults(all_item_ranking=False)

    parser.add_argument('--is_freeze_emb', dest='freeze_emb', action='store_true')
    parser.add_argument('--no_freeze_emb', dest='freeze_emb', action='store_false')
    parser.set_defaults(freeze_emb=False)

    # Env

    parser.add_argument('--leave_threshold', default=10, type=float)
    parser.add_argument('--num_leave_compute', default=3, type=int)
    parser.add_argument('--max_turn', default=30, type=int)

    # state_tracker
    parser.add_argument('--dim_state', default=20, type=int)
    parser.add_argument('--dim_model', default=64, type=int)
    parser.add_argument('--nhead', default=4, type=int)
    # parser.add_argument('--max_len', default=100, type=int)
    parser.add_argument('--window', default=2, type=int)

    # tianshou
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=200)
    # parser.add_argument('--step-per-epoch', type=int, default=15000)
    # parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])


    parser.add_argument('--test-num', type=int, default=100)

    parser.add_argument('--render', type=float, default=0)
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--step-per-epoch', type=int, default=50000)
    parser.add_argument('--step-per-collect', type=int, default=16)
    parser.add_argument('--update-per-step', type=float, default=1 / 16)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    # parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--logdir', type=str, default='log')
    # parser.add_argument(
    #     '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    # )


    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--step-per-epoch', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=1024)

    parser.add_argument("--vae-hidden-sizes", type=int, nargs='*', default=[64, 64])
    # default to 2 * action_dim
    parser.add_argument('--latent_dim', type=int, default=None)
    parser.add_argument("--gamma", default=0.99)
    parser.add_argument("--tau", default=0.005)
    # Weighting for Clipped Double Q-learning in BCQ
    parser.add_argument("--lmbda", default=0.75)
    # Max perturbation hyper-parameter for BCQ
    parser.add_argument("--phi", default=0.05)

    # parser.add_argument(
    #     '--watch',
    #     default=False,
    #     action='store_true',
    #     help='watch the play of pre-trained policy only',
    # )
    # parser.add_argument("--show-progress", action="store_true")


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

def get_buffer_size(args, df_train, env):
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
    df_val, df_user_val, df_item_val, list_feat = get_val_data(args.env)
    df_train = df_train.head(10000)
    if "time_ms" in df_train.columns:
        df_train.rename(columns={"time_ms": "timestamp"}, inplace=True)
        df_train = df_train.sort_values(["user_id", "timestamp"])
    if not "timestamp" in df_train.columns:
        df_train = df_train.sort_values(["user_id"])

    df_train[["user_id", "item_id"]].to_numpy()

    env, env_task_class, kwargs_um = get_true_env(args)
    buffer = get_buffer_size(args, df_train, env)
    env.max_turn = args.max_turn

    test_envs = DummyVectorEnv(
        [lambda: env_task_class(**kwargs_um) for _ in range(args.test_num)])

    args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    # seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    return env, buffer, test_envs

    # %% 4. Setup model
def setup_policy_model(args, env, test_envs):
    ensemble_models, _, _ = prepare_user_model_and_env(args)

    saved_embedding = ensemble_models.load_val_user_item_embedding(freeze_emb=args.freeze_emb)
    user_columns, action_columns, feedback_columns, \
    have_user_embedding, have_action_embedding, have_feedback_embedding = \
        get_dataset_columns(saved_embedding["feat_user"].weight.shape[1], saved_embedding["feat_item"].weight.shape[1], envname=args.env, env=env)


    args.action_shape = action_columns[0].vocabulary_size
    if args.use_userEmbedding:
        args.state_shape = action_columns[0].embedding_dim + saved_embedding.feat_user.weight.shape[1]
    else:
        args.state_shape = action_columns[0].embedding_dim

    args.max_action = env.action_space.high[0]

    state_tracker = StateTrackerAvg2(user_columns, action_columns, feedback_columns, args.state_shape,
                                    saved_embedding, device=args.device, window=args.window,
                                    use_userEmbedding=args.use_userEmbedding, MAX_TURN=args.max_turn + 1).to(args.device)

    net_a = MLP(
        input_dim=args.state_shape + args.action_shape,
        output_dim=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
    )
    actor = Perturbation(
        net_a, max_action=args.max_action, device=args.device, phi=args.phi
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # vae
    # output_dim = 0, so the last Module in the encoder is ReLU
    vae_encoder = MLP(
        input_dim=args.state_shape + args.action_shape,
        hidden_sizes=args.vae_hidden_sizes,
        device=args.device,
    )
    if not args.latent_dim:
        args.latent_dim = args.action_shape * 2
    vae_decoder = MLP(
        input_dim=args.state_shape + args.latent_shape,
        output_dim=args.action_shape,
        hidden_sizes=args.vae_hidden_sizes,
        device=args.device,
    )
    vae = VAE(
        vae_encoder,
        vae_decoder,
        hidden_dim=args.vae_hidden_sizes[-1],
        latent_dim=args.latent_dim,
        max_action=args.max_action,
        device=args.device,
    ).to(args.device)
    vae_optim = torch.optim.Adam(vae.parameters())
    optim = [actor_optim, critic1_optim, critic2_optim, vae_optim]

    policy = BCQPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        vae,
        vae_optim,
        device=args.device,
        gamma=args.gamma,
        tau=args.tau_BCQ,
        lmbda=args.lmbda,
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



    return policy, test_collector, optim

def learn_policy(args, policy, buffer, test_collector, state_tracker, optim, MODEL_SAVE_PATH, logger_path):
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_bcq'
    log_path = os.path.join(args.logdir, args.task, 'bcq', log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger1 = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    policy.callbacks = [LoggerCallback_Policy(logger_path)]
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

