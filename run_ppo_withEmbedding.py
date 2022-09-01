# -*- coding: utf-8 -*-
# @Time    : 2022/7/28 15:35
# @Author  : Chongming GAO
# @FileName: testRL.py


import datetime
import functools
import json
import os
import pickle
import pprint
import time

import gym
import torch
import argparse
import numpy as np

from tensorflow.python.keras.callbacks import History

from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from core.inputs import get_dataset_columns
from core.collector import Collector
# from tianshou.data import Collector
from core.state_tracker import StateTrackerTransformer
from core.user_model import compute_input_dim
from core.policy.ppo import PPOPolicy
from core.user_model_pairwise import UserModel_Pairwise
from core.worldModel.simulated_env import SimulatedEnv

from environments.KuaiRec.env.KuaiEnv import KuaiEnv
from tianshou.utils import BasicLogger
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
# from tianshou.trainer import onpolicy_trainer
from core.trainer.onpolicy import onpolicy_trainer
from tianshou.data import VectorReplayBuffer
from tianshou.utils.net.discrete import Actor, Critic

import logzero
from logzero import logger

# from util.upload import my_upload
from util.utils import create_dir, LoggerCallback_RL, save_model_fn, LoggerCallback_Policy



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="KuaiEnv-v0")
    parser.add_argument("--user_model_name", type=str, default="DeepFM")
    parser.add_argument("--model_name", type=str, default="CIRS_with_emb")
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
    parser.add_argument('--dim_model', default=64, type=int)
    parser.add_argument('--nhead', default=4, type=int)
    # parser.add_argument('--max_len', default=100, type=int)

    # tianshou
    parser.add_argument('--buffer-size', type=int, default=11000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--step-per-epoch', type=int, default=15000)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])

    parser.add_argument('--episode-per-collect', type=int, default=100)
    parser.add_argument('--training-num', type=int, default=100)

    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--render', type=float, default=0)

    # ppo
    parser.add_argument('--vf-coef', type=float, default=0.25)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=1)
    parser.add_argument('--norm-adv', type=int, default=1)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument("--save-interval", type=int, default=1000)

    parser.add_argument("--read_message", type=str, default="UserModel1")
    parser.add_argument("--message", type=str, default="CIRS_emb")

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
    logger.info(json.dumps(vars(args), indent=2))

    if args.cpu:
        device = "cpu"
    else:
        device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    # %% 2. prepare user model

    UM_SAVE_PATH = os.path.join(".", "saved_models", args.env, args.user_model_name)
    MODEL_MAT_PATH = os.path.join(UM_SAVE_PATH, "mats", f"[{args.read_message}]_mat.pickle")
    MODEL_PARAMS_PATH = os.path.join(UM_SAVE_PATH, "params", f"[{args.read_message}]_params.pickle")
    MODEL_PATH = os.path.join(UM_SAVE_PATH, "models", f"[{args.read_message}]_model.pt")
    MODEL_EMBEDDING_PATH = os.path.join(UM_SAVE_PATH, "embeddings", f"[{args.read_message}]_emb.pt")
    USER_EMBEDDING_PATH = os.path.join(UM_SAVE_PATH, "embeddings", f"[{args.read_message}]_emb_user.pt")
    ITEM_EMBEDDING_PATH = os.path.join(UM_SAVE_PATH, "embeddings", f"[{args.read_message}]_emb_item.pt")

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
                                           "feat_item": torch.nn.Embedding.from_pretrained(item_embedding,
                                                                                           freeze=True)})
    user_columns, action_columns, feedback_columns, \
    have_user_embedding, have_action_embedding, have_feedback_embedding = \
        get_dataset_columns(user_embedding.shape[1], item_embedding.shape[1], envname=args.env, env=env)

    # assert args.dim_model == compute_input_dim(action_columns)
    # args.dim_model = compute_input_dim(action_columns)
    state_tracker = StateTrackerTransformer(user_columns, action_columns, feedback_columns,
                                            dim_model=args.dim_model, dim_state=args.dim_state,
                                            dim_max_batch=max(args.training_num, args.test_num),
                                            dataset=args.env,
                                            have_user_embedding=have_user_embedding,
                                            have_action_embedding=have_action_embedding,
                                            have_feedback_embedding=have_feedback_embedding,
                                            use_pretrained_embedding=True, saved_embedding=saved_embedding,
                                            nhead=args.nhead, d_hid=128, nlayers=2, dropout=0.1,
                                            device=device, seed=args.seed, MAX_TURN=args.max_turn + 1).to(device)

    net = Net(args.dim_state, hidden_sizes=args.hidden_sizes, device=device)
    actor = Actor(net, env.mat.shape[1], device=device).to(device)
    critic = Critic(net, device=device).to(device)
    # critic = Critic(Net(state_shape, hidden_sizes=args.hidden_sizes, device=device), device=device).to(device)

    # orthogonal initialization
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    optim_RL = torch.optim.Adam(
        list(actor.parameters()) +
        list(critic.parameters()), lr=args.lr)
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)
    optim = [optim_RL, optim_state]

    # replace DiagGuassian with Independent(Normal) which is equivalent
    # pass *logits to be consistent with policy.forward

    dist = torch.distributions.Categorical

    policy = PPOPolicy(
        actor, critic, optim, dist,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
        # dual_clip=args.dual_clip,
        # dual clip cause monotonically increasing log_std :)
        value_clip=args.value_clip,
        gae_lambda=args.gae_lambda,
        action_space=simulatedEnv.action_space,
        action_bound_method="" if args.env == "KuaiEnv-v0" else "clip",
        action_scaling=False if args.env == "KuaiEnv-v0" else True
    )

    # %% 5. Prepare the collectors and logs
    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        preprocess_fn=state_tracker.build_state
    )
    test_collector = Collector(
        policy, test_envs,
        preprocess_fn=state_tracker.build_state
    )

    # log
    log_path = os.path.join(MODEL_SAVE_PATH)
    writer = SummaryWriter(log_path)
    logger1 = BasicLogger(writer, save_interval=args.save_interval)

    # def save_fn(policy):
    #     torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))
    #
    # def stop_fn(mean_rewards):
    #     return mean_rewards >= simulatedEnv.spec.reward_threshold
    #
    # def save_checkpoint_fn(epoch, env_step, gradient_step):
    #     # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    #     torch.save({
    #         'model': policy.state_dict(),
    #         'optim_RL': optim[0].state_dict(),
    #         'optim_state': optim[1].state_dict(),
    #     }, os.path.join(log_path, 'checkpoint.pth'))

    # if args.resume:
    #     # load from existing checkpoint
    #     print(f"Loading agent under {log_path}")
    #     ckpt_path = os.path.join(log_path, 'checkpoint.pth')
    #     if os.path.exists(ckpt_path):
    #         checkpoint = torch.load(ckpt_path, map_location=args.device)
    #         policy.load_state_dict(checkpoint['model'])
    #         optim.load_state_dict(checkpoint['optim'])
    #         print("Successfully restore policy and optim.")
    #     else:
    #         print("Fail to restore policy and optim.")

    # policy.callbacks = [History()] + [LoggerCallback_RL(logger_path)]

    policy.callbacks = [LoggerCallback_Policy(logger_path)]

    # %% 6. Learn the model

    model_save_path = os.path.join(MODEL_SAVE_PATH, "{}_{}.pt".format(args.model_name, args.message))

    result = onpolicy_trainer(policy, train_collector, test_collector,
                              args.epoch, args.step_per_epoch,
                              args.repeat_per_collect, args.test_num, args.batch_size,
                              episode_per_collect=args.episode_per_collect,
                              # stop_fn=stop_fn,
                              # save_fn=save_fn,
                              logger=logger1,
                              resume_from_log=args.resume,
                              # save_checkpoint_fn=save_checkpoint_fn,
                              save_model_fn=functools.partial(save_model_fn,
                                                              model_save_path=model_save_path,
                                                              state_tracker=state_tracker,
                                                              optim=optim,
                                                              is_save=args.is_save)
                              )

    print(__file__)
    pprint.pprint(result)

    # %% 7. save info

    # model_save_path = os.path.join(MODEL_SAVE_PATH, "{}_{}.pt".format(args.model_name, args.message))
    # torch.save(model.state_dict(), model_save_path)
    torch.save({
        'policy': policy.cpu().state_dict(),
        'optim_RL': optim[0].state_dict(),
        'optim_state': optim[1].state_dict(),
        'state_tracker': state_tracker.cpu().state_dict(),
    }, model_save_path)

    REMOTE_ROOT = "/root/Counterfactual_IRS"
    LOCAL_PATH = logger_path
    REMOTE_PATH = os.path.join(REMOTE_ROOT, os.path.dirname(LOCAL_PATH))

    # my_upload(LOCAL_PATH, REMOTE_PATH, REMOTE_ROOT)


if __name__ == '__main__':
    args = get_args()
    main(args)
