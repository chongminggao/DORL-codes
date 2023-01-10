import argparse
import functools
import os
import pickle
import pprint
import random
import traceback

import numpy as np
import pandas as pd
# import pytest
import torch

import sys
sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from policy_utils import prepare_dir_log

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from core.collector_set import CollectorSet
from core.evaluation.evaluator import Callback_Coverage_Count
from core.user_model_ensemble import EnsembleModel
from core.configs import get_true_env, get_common_args, get_val_data, get_training_item_domination
from core.collector2 import Collector
from core.inputs import get_dataset_columns
from core.policy.a2c2 import A2CPolicy_withEmbedding
from core.state_tracker2 import StateTrackerAvg2
from core.trainer.onpolicy import onpolicy_trainer
from core.worldModel.simulated_env import SimulatedEnv

from tianshou.data import VectorReplayBuffer
from tianshou.env import DummyVectorEnv

from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic

# from util.upload import my_upload
from util.utils import LoggerCallback_Policy, save_model_fn
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
    parser.add_argument("--model_name", type=str, default="A2C_with_emb")
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--cuda', default=0, type=int)

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
    parser.set_defaults(exploration_noise=False)
    parser.add_argument('--eps', default=0.1, type=float)

    parser.add_argument('--is_all_item_ranking', dest='is_all_item_ranking', action='store_true')
    parser.add_argument('--no_all_item_ranking', dest='is_all_item_ranking', action='store_false')
    parser.set_defaults(all_item_ranking=False)

    parser.add_argument('--is_freeze_emb', dest='freeze_emb', action='store_true')
    parser.add_argument('--no_freeze_emb', dest='freeze_emb', action='store_false')
    parser.set_defaults(freeze_emb=False)


    # Env
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument('--tau', default=100, type=float)
    parser.add_argument('--gamma_exposure', default=10, type=float)

    parser.add_argument('--lambda_variance', default=1, type=float)
    parser.add_argument('--lambda_entropy', default=1, type=float)

    parser.add_argument('--is_exposure_intervention', dest='use_exposure_intervention', action='store_true')
    parser.add_argument('--no_exposure_intervention', dest='use_exposure_intervention', action='store_false')
    parser.set_defaults(use_exposure_intervention=False)

    # tianshou
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])

    parser.add_argument('--episode-per-collect', type=int, default=100)
    parser.add_argument('--training-num', type=int, default=100)
    parser.add_argument('--test-num', type=int, default=100)

    parser.add_argument('--render', type=float, default=0)
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--step-per-epoch', type=int, default=100000)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')

    # a2c special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--max-grad-norm', type=float, default=None)
    parser.add_argument('--gae-lambda', type=float, default=1.)
    parser.add_argument('--rew-norm', action="store_true", default=False)

    parser.add_argument("--read_message", type=str, default="UM")
    parser.add_argument("--message", type=str, default="A2C_with_emb")

    args = parser.parse_known_args()[0]
    return args


def prepare_user_model_and_env(args):
    args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    random.seed(args.seed)

    UM_SAVE_PATH = os.path.join(".", "saved_models", args.env, args.user_model_name)
    # MODEL_MAT_PATH = os.path.join(UM_SAVE_PATH, "mats", f"[{args.read_message}]_mat.pickle")
    MODEL_PARAMS_PATH = os.path.join(UM_SAVE_PATH, "params", f"[{args.read_message}]_params.pickle")

    with open(MODEL_PARAMS_PATH, "rb") as file:
        model_params = pickle.load(file)

    n_models = model_params["n_models"]
    model_params.pop('n_models')

    ensemble_models = EnsembleModel(n_models, args.read_message, UM_SAVE_PATH, **model_params)
    ensemble_models.load_all_models()

    user_model = ensemble_models.user_models[0]
    if hasattr(user_model, 'ab_embedding_dict') and args.is_ab:
        alpha_u = user_model.ab_embedding_dict["alpha_u"].weight.detach().cpu().numpy()
        beta_i = user_model.ab_embedding_dict["beta_i"].weight.detach().cpu().numpy()
    else:
        print("Note there are no available alpha and beta！！")
        alpha_u = None
        beta_i = None

    return ensemble_models, alpha_u, beta_i

    # %% 3. prepare envs


def prepare_envs(args, ensemble_models, alpha_u, beta_i):
    env, env_task_class, kwargs_um = get_true_env(args)

    # user_features, item_features, reward_features = get_features(args.env, args.is_userinfo)
    # embedding_dim = ensemble_models.user_models[0].feature_columns[0].embedding_dim

    # dataset_val, df_user_val, df_item_val = load_dataset_val(args, user_features, item_features, reward_features, embedding_dim, embedding_dim)

    # entropy_user, map_entropy = ensemble_models.get_save_entropy_mat(args.env, args.entropy_window)

    entropy_dict = dict()
    if 0 in args.entropy_window:
        entropy_path = os.path.join(ensemble_models.Entropy_PATH, "user_entropy.csv")
        entropy = pd.read_csv(entropy_path)
        entropy.set_index("user_id", inplace=True)
        entropy_mat_0 = entropy.to_numpy().reshape([-1])
        entropy_dict.update({"on_user": entropy_mat_0})

    if len(set(args.entropy_window) - set([0])):
        savepath = os.path.join(ensemble_models.Entropy_PATH, "map_entropy.pickle")
        map_entropy = pickle.load(open(savepath, 'rb'))
        entropy_dict.update({"map": map_entropy})

    with open(ensemble_models.PREDICTION_MAT_PATH, "rb") as file:
        predicted_mat = pickle.load(file)

    with open(ensemble_models.VAR_MAT_PATH, "rb") as file:
        maxvar_mat = pickle.load(file)

    kwargs = {
        "ensemble_models": ensemble_models,
        # "dataset_val": dataset_val,
        # "need_transform": args.need_transform,
        "env_task_class": env_task_class,
        # "user_model": user_model,
        "use_exposure_intervention": args.use_exposure_intervention,
        "task_env_param": kwargs_um,
        "task_name": args.env,
        "version": args.version,
        "tau": args.tau,
        "alpha_u": alpha_u,
        "beta_i": beta_i,
        "lambda_entropy": args.lambda_entropy,
        "lambda_variance": args.lambda_variance,
        "predicted_mat": predicted_mat,
        "maxvar_mat": maxvar_mat,
        "entropy_dict": entropy_dict,
        "entropy_window": args.entropy_window,
        "gamma_exposure": args.gamma_exposure,
        "step_n_actions": max(args.entropy_window)
    }

    # simulatedEnv = SimulatedEnv(**kwargs)

    train_envs = DummyVectorEnv(
        [lambda: SimulatedEnv(**kwargs) for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv(
        [lambda: env_task_class(**kwargs_um) for _ in range(args.test_num)])
    test_envs_NX_0 = DummyVectorEnv(
        [lambda: env_task_class(**kwargs_um) for _ in range(args.test_num)])
    test_envs_NX_x = DummyVectorEnv(
        [lambda: env_task_class(**kwargs_um) for _ in range(args.test_num)])

    test_envs_dict = {"FB": test_envs, "NX_0": test_envs_NX_0, f"NX_{args.force_length}": test_envs_NX_x}

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    # test_envs.seed(args.seed)

    return env, train_envs, test_envs_dict

    # %% 4. Setup model


def setup_policy_model(args, ensemble_models, env, train_envs, test_envs_dict):
    saved_embedding = ensemble_models.load_val_user_item_embedding(freeze_emb=args.freeze_emb)

    user_columns, action_columns, feedback_columns, have_user_embedding, have_action_embedding, have_feedback_embedding = \
        get_dataset_columns(saved_embedding["feat_user"].weight.shape[1], saved_embedding["feat_item"].weight.shape[1],
                            env.mat.shape[0], env.mat.shape[1], envname=args.env)

    args.action_shape = env.mat.shape[1]
    if args.use_userEmbedding:
        args.state_dim = action_columns[0].embedding_dim + saved_embedding.feat_user.weight.shape[1]
    else:
        args.state_dim = action_columns[0].embedding_dim

    state_tracker = StateTrackerAvg2(user_columns, action_columns, feedback_columns, args.state_dim,
                                     saved_embedding, device=args.device, window_size=args.window_size,
                                     use_userEmbedding=args.use_userEmbedding).to(args.device)

    if args.cpu:
        args.device = "cpu"
    else:
        args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    # model
    net = Net(args.state_dim, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = Actor(net, args.action_shape, device=args.device).to(args.device)
    critic = Critic(net, device=args.device).to(args.device)
    optim_RL = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=args.lr)
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)
    optim = [optim_RL, optim_state]

    dist = torch.distributions.Categorical
    policy = A2CPolicy_withEmbedding(
        actor,
        critic,
        optim,
        dist,
        state_tracker=state_tracker,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        reward_normalization=args.rew_norm,
        action_space=env.action_space,
        action_bound_method="",  # not clip
        action_scaling=False
    )
    policy.set_eps(args.eps)

    # %% 5. Prepare the collectors and logs
    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        preprocess_fn=state_tracker.build_state,
        exploration_noise=args.exploration_noise,
    )
    # test_collector = Collector(
    #     policy, test_envs_dict,
    #     VectorReplayBuffer(args.buffer_size, len(test_envs)),
    #     preprocess_fn=state_tracker.build_state,
    #     exploration_noise=args.exploration_noise,
    # )
    policy.set_collector(train_collector)

    test_collector_set = CollectorSet(policy, test_envs_dict, args.buffer_size, args.test_num,
                                      preprocess_fn=state_tracker.build_state,
                                      exploration_noise=args.exploration_noise,
                                      force_length=args.force_length)

    return policy, train_collector, test_collector_set, state_tracker, optim


def learn_policy(args, env, policy, train_collector, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH, logger_path):
    # log
    # log_path = os.path.join(args.logdir, args.env, 'a2c')
    # writer = SummaryWriter(log_path)
    # logger1 = TensorboardLogger(writer)

    # def save_best_fn(policy):
    #     torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    # env = test_collector_set.env
    df_val, df_user_val, df_item_val, list_feat = get_val_data(args.env)
    item_feat_domination = get_training_item_domination(args.env)
    policy.callbacks = [
        Callback_Coverage_Count(test_collector_set, df_item_val, args.need_transform, item_feat_domination,
                                lbe_item=env.lbe_item if args.need_transform else None, top_rate=args.top_rate),
        LoggerCallback_Policy(logger_path, args.force_length)]
    model_save_path = os.path.join(MODEL_SAVE_PATH, "{}_{}.pt".format(args.model_name, args.message))

    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector_set,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        episode_per_collect=args.episode_per_collect,
        # stop_fn=stop_fn,
        # save_best_fn=save_best_fn,
        # logger=logger1,
        save_model_fn=functools.partial(save_model_fn,
                                        model_save_path=model_save_path,
                                        state_tracker=state_tracker,
                                        optim=optim,
                                        is_save=args.is_save)
    )
    # assert stop_fn(result['best_reward'])

    print(__file__)
    pprint.pprint(result)
    logger.info(result)


def main(args):
    # %% 1. Prepare the saved path.
    MODEL_SAVE_PATH, logger_path = prepare_dir_log(args)

    # %% 2. Prepare user model and environment
    ensemble_models, alpha_u, beta_i = prepare_user_model_and_env(args)
    env, train_envs, test_envs_dict = prepare_envs(args, ensemble_models, alpha_u, beta_i)

    # %% 3. Setup policy
    policy, train_collector, test_collector_set, state_tracker, optim = setup_policy_model(args, ensemble_models, env,
                                                                                           train_envs, test_envs_dict)

    # %% 4. Learn policy
    learn_policy(args, env, policy, train_collector, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH, logger_path)


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
