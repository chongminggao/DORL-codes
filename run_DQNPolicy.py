# -*- coding: utf-8 -*-
# @Time    : 2022/7/31 11:05
# @Author  : Chongming GAO
# @FileName: run_DQNPolicy.py

import argparse
import datetime
import json
import os
import pickle
import pprint
import time

import gym
import logzero
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from core.inputs import get_dataset_columns
from core.policy.dqn import DQNPolicy
from core.state_tracker import StateTrackerTransformer, StateTrackerAvg
from core.user_model import compute_input_dim
from core.user_model_pairwise import UserModel_Pairwise
from core.worldModel.simulated_env import SimulatedEnv
from environments.KuaiRec.env.KuaiEnv import KuaiEnv
from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from util.utils import create_dir


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="KuaiEnv-v0")
    parser.add_argument("--user_model_name", type=str, default="DeepFM")
    parser.add_argument("--model_name", type=str, default="CIRS")
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--cuda', default=1, type=int)

    parser.add_argument('--is_ab', dest='is_ab', action='store_true')
    parser.add_argument('--no_ab', dest='is_ab', action='store_false')
    parser.set_defaults(is_ab=True)

    parser.add_argument('--cpu', dest='cpu', action='store_true')
    parser.set_defaults(cpu=False)

    parser.add_argument('--is_save', dest='is_save', action='store_true')
    parser.add_argument('--no_save', dest='is_save', action='store_false')
    parser.set_defaults(is_save=False)

    # Env
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument('--tau', default=100, type=float)
    parser.add_argument('--gamma_exposure', default=10, type=float)

    parser.add_argument('--leave_threshold', default=0, type=int)
    parser.add_argument('--num_leave_compute', default=1, type=int)
    parser.add_argument('--max_turn', default=30, type=int)

    # state_tracker
    parser.add_argument('--dim_state', default=20, type=int)
    parser.add_argument('--dim_model', default=32, type=int)
    parser.add_argument('--nhead', default=4, type=int)
    # parser.add_argument('--max_len', default=100, type=int)

    parser.add_argument('--task', type=str, default='CartPole-v0')
    parser.add_argument('--reward-threshold', type=float, default=None)
    # parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--step-per-epoch', type=int, default=10000)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument(
        '--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128]
    )
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--prioritized-replay', action="store_true", default=False)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument("--read_message", type=str, default="UserModel1")
    parser.add_argument("--message", type=str, default="OMIRS")

    args = parser.parse_known_args()[0]
    return args


def main(args):
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
    logzero.logger.info(json.dumps(vars(args), indent=2))

    if args.cpu:
        device = "cpu"
    else:
        device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    # %% 2. prepare user model

    UM_SAVE_PATH = os.path.join(".", "saved_models", args.env, args.user_model_name)
    MODEL_MAT_PATH = os.path.join(UM_SAVE_PATH, "mats", f"mat_{args.read_message}.pickle")
    MODEL_PARAMS_PATH = os.path.join(UM_SAVE_PATH, "params", f"params_{args.read_message}.pickle")
    MODEL_PATH = os.path.join(UM_SAVE_PATH, "models", f"model_{args.read_message}.pt")
    MODEL_EMBEDDING_PATH = os.path.join(UM_SAVE_PATH, "embeddings", f"emb_{args.read_message}.pt")
    USER_EMBEDDING_PATH = os.path.join(UM_SAVE_PATH, "embeddings", f"emb_user_{args.read_message}.pt")
    ITEM_EMBEDDING_PATH = os.path.join(UM_SAVE_PATH, "embeddings", f"emb_item_{args.read_message}.pt")

    # USERMODEL_Path = os.path.join(".", "saved_models", args.env, args.user_model_name)
    # model_parameter_path = os.path.join(USERMODEL_Path,
    #                                     "{}_params_{}.pickle".format(args.user_model_name, args.read_message))
    # model_save_path = os.path.join(USERMODEL_Path, "{}_{}.pt".format(args.user_model_name, args.read_message))

    with open(MODEL_PARAMS_PATH, "rb") as file:
        model_params = pickle.load(file)

    model_params["device"] = "cpu"
    user_model = UserModel_Pairwise(**model_params)
    user_model.load_state_dict(torch.load(MODEL_PATH))

    # debug: for saving gpu space
    # user_model = user_model.to(device)
    # user_model.device = device
    # user_model.linear_model.device = device
    # user_model.linear.device = device

    if hasattr(user_model, 'ab_embedding_dict') and args.is_ab:
        alpha_u = user_model.ab_embedding_dict["alpha_u"].weight.detach().cpu().numpy()
        beta_i = user_model.ab_embedding_dict["beta_i"].weight.detach().cpu().numpy()
    else:
        print("Note there are no available alpha and beta！！")
        alpha_u = np.ones([7176, 1])
        beta_i = np.ones([10729, 1])

    # env = gym.make('VirtualTB-v0')

    # %% 3. prepare envs
    mat, lbe_user, lbe_video, list_feat, df_video_env, df_dist_small = KuaiEnv.load_mat()
    # register(
    #     id=args.env,  # 'KuaiEnv-v0',
    #     entry_point='environments.KuaishouRec.env.kuaiEnv:KuaiEnv',
    #     kwargs={"mat": mat,
    #             "lbe_user": lbe_user,
    #             "lbe_video": lbe_video,
    #             "num_leave_compute": args.num_leave_compute,
    #             "leave_threshold": args.leave_threshold,
    #             "max_turn": args.max_turn,
    #             "list_feat": list_feat,
    #             "df_video_env": df_video_env,
    #             "df_dist_small": df_dist_small}
    # )
    # env = gym.make(args.env)
    kwargs_um = {"mat": mat,
                 "lbe_user": lbe_user,
                 "lbe_video": lbe_video,
                 "num_leave_compute": args.num_leave_compute,
                 "leave_threshold": args.leave_threshold,
                 "max_turn": args.max_turn,
                 "list_feat": list_feat,
                 "df_video_env": df_video_env,
                 "df_dist_small": df_dist_small}
    env = KuaiEnv(**kwargs_um)

    # normed_mat = KuaiEnv.compute_normed_reward(user_model, lbe_user, lbe_video, df_video_env,)
    # mat_save_path = os.path.join(USERMODEL_Path, "normed_mat-{}.pickle".format(args.read_message))
    with open(MODEL_MAT_PATH, "rb") as file:
        normed_mat = pickle.load(file)
    # register(
    #     id='SimulatedEnv-v0',
    #     entry_point='core.env.simulatedEnv.simulated_env:SimulatedEnv',
    #     kwargs={"user_model": user_model,
    #             "task_name": args.env,
    #             "version": args.version,
    #             "tau": args.tau,
    #             "alpha_u": alpha_u,
    #             "beta_i": beta_i,
    #             "normed_mat": normed_mat,
    #             "gamma_exposure": args.gamma_exposure}
    # )
    # simulatedEnv = gym.make("SimulatedEnv-v0")

    kwargs = {"user_model": user_model,
              "task_env_param": kwargs_um,
              "task_name": args.env,
              "version": args.version,
              "tau": args.tau,
              "alpha_u": alpha_u,
              "beta_i": beta_i,
              "normed_mat": normed_mat,
              "gamma_exposure": args.gamma_exposure}
    simulatedEnv = SimulatedEnv(**kwargs)

    state_shape = simulatedEnv.observation_space.shape or simulatedEnv.observation_space.n
    action_shape = simulatedEnv.action_space.shape or simulatedEnv.action_space.n
    max_action = simulatedEnv.action_space.high[0]

    # envs = [KuaiEnv(**kwargs_um) for _ in range(args.training_num)]
    #
    # kwargs_list = [kwargs.copy() for _ in range(args.training_num)]
    # for ee, kw in zip(envs,kwargs_list):
    #     kw["task_env"] = ee
    # train_envs = DummyVectorEnv(
    #     [lambda: SimulatedEnv(**kw) for kw in kwargs_list])
    train_envs = DummyVectorEnv(
        [lambda: SimulatedEnv(**kwargs) for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv(
        [lambda: KuaiEnv(**kwargs_um) for _ in range(args.test_num)])

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # %% 4. Setup model

    user_embedding = torch.load(USER_EMBEDDING_PATH)
    item_embedding = torch.load(ITEM_EMBEDDING_PATH)
    user_embedding = user_embedding[env.lbe_user.classes_]
    item_embedding = item_embedding[env.lbe_video.classes_]
    saved_embedding = torch.nn.ModuleDict({"feat_user": torch.nn.Embedding.from_pretrained(user_embedding, freeze=True),
                                           "feat_item": torch.nn.Embedding.from_pretrained(item_embedding, freeze=True)})

    user_columns, action_columns, feedback_columns, \
    has_user_embedding, has_action_embedding, has_feedback_embedding = \
        get_dataset_columns(user_embedding.shape[1], item_embedding.shape[1], envname=args.env, env=env)

    # assert args.dim_model == compute_input_dim(action_columns)

    state_tracker = StateTrackerAvg(user_columns, action_columns, feedback_columns,
                                    dim_model=args.dim_model, saved_embedding=saved_embedding)


    # state_tracker = StateTrackerTransformer(user_columns, action_columns, feedback_columns,
    #                                         dim_model=args.dim_model, dim_state=args.dim_state,
    #                                         dim_max_batch=max(args.training_num, args.test_num),
    #                                         dataset=args.env,
    #                                         have_user_embedding=has_user_embedding,
    #                                         have_action_embedding=has_action_embedding,
    #                                         have_feedback_embedding=has_feedback_embedding,
    #                                         use_pretrained_embedding=True, saved_embedding=saved_embedding,
    #                                         nhead=args.nhead, d_hid=128, nlayers=2, dropout=0.1,
    #                                         device=device, seed=args.seed, MAX_TURN=args.max_turn + 1).to(device)

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # Q_param = V_param = {"hidden_sizes": [128]}
    # model
    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        # dueling=(Q_param, V_param),
    ).to(args.device)


    optim_RL = torch.optim.Adam(net.parameters(), lr=args.lr)
    # optim_RL = torch.optim.Adam(
    #     list(actor.parameters()) +
    #     list(critic.parameters()), lr=args.lr)
    # optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)
    # optim = [optim_RL, optim_state]


    policy = DQNPolicy(
        net,
        optim_RL,
        args.gamma,
        args.n_step,
        target_update_freq=args.target_update_freq,
    )
    # buffer
    if args.prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            alpha=args.alpha,
            beta=args.beta,
        )
    else:
        buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))
    # collector
    train_collector = Collector(policy, train_envs, buf, exploration_noise=True,
                                preprocess_fn=state_tracker.build_state)
    test_collector = Collector(policy, test_envs, exploration_noise=True,
                               preprocess_fn=state_tracker.build_state)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    log_path = os.path.join(args.logdir, args.task, 'dqn')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold

    def train_fn(epoch, env_step):
        # eps annnealing, just a demo
        if env_step <= 10000:
            policy.set_eps(args.eps_train)
        elif env_step <= 50000:
            eps = args.eps_train - (env_step - 10000) / \
                  40000 * (0.9 * args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * args.eps_train)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        update_per_step=args.update_per_step,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    )
    assert stop_fn(result['best_reward'])



def test_dqn(args=get_args()):
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    if args.reward_threshold is None:
        default_reward_threshold = {"CartPole-v0": 195}
        args.reward_threshold = default_reward_threshold.get(
            args.task, env.spec.reward_threshold
        )
    # train_envs = gym.make(args.task)
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)]
    )
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)]
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # Q_param = V_param = {"hidden_sizes": [128]}
    # model
    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        # dueling=(Q_param, V_param),
    ).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = DQNPolicy(
        net,
        optim,
        args.gamma,
        args.n_step,
        target_update_freq=args.target_update_freq,
    )
    # buffer
    if args.prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            alpha=args.alpha,
            beta=args.beta,
        )
    else:
        buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))
    # collector
    train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    log_path = os.path.join(args.logdir, args.task, 'dqn')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold

    def train_fn(epoch, env_step):
        # eps annnealing, just a demo
        if env_step <= 10000:
            policy.set_eps(args.eps_train)
        elif env_step <= 50000:
            eps = args.eps_train - (env_step - 10000) / \
                  40000 * (0.9 * args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * args.eps_train)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        update_per_step=args.update_per_step,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    )
    assert stop_fn(result['best_reward'])

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = gym.make(args.task)
        policy.eval()
        policy.set_eps(args.eps_test)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


def test_pdqn(args=get_args()):
    args.prioritized_replay = True
    args.gamma = .95
    args.seed = 1
    test_dqn(args)


if __name__ == '__main__':
    args = get_args()
    main(args)
    # test_dqn(get_args())
