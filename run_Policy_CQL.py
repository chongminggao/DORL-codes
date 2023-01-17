import argparse
import functools
import os
import pprint
import sys
import traceback

import torch

from policy_utils import prepare_dir_log, prepare_buffer_via_offline_data, setup_offline_state_tracker

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from core.collector_set import CollectorSet
from core.evaluation.evaluator import Callback_Coverage_Count
from core.policy.discrete_cql import DiscreteCQLPolicy_withEmbedding
from core.trainer.offline import offline_trainer
from run_Policy_Main import prepare_user_model_and_env, get_args_all
from core.configs import get_val_data, get_common_args, \
    get_training_item_domination

from tianshou.utils.net.common import Net

from util.utils import LoggerCallback_Policy, save_model_fn
import logzero
from logzero import logger

try:
    import envpool
except ImportError:
    envpool = None


def get_args_CQL():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="CQL")
    parser.add_argument('--num-quantiles', type=int, default=20)
    parser.add_argument("--min-q-weight", type=float, default=10.)
    parser.add_argument("--eps-test", type=float, default=0.001)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument("--message", type=str, default="CQL")

    args = parser.parse_known_args()[0]
    return args

    # %% 4. Setup model


def setup_policy_model(args, state_tracker, buffer, test_envs_dict):
    # ensemble_models, _, _ = prepare_user_model_and_env(args)

    net = Net(
        args.state_dim,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        softmax=False,
        num_atoms=args.num_quantiles
    )
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    policy = DiscreteCQLPolicy_withEmbedding(
        net,
        optim,
        args.gamma,
        args.num_quantiles,
        args.n_step,
        args.target_update_freq,
        min_q_weight=args.min_q_weight,
        state_tracker=state_tracker,
        buffer=buffer,
    ).to(args.device)

    # collector
    # buffer has been gathered
    test_collector_set = CollectorSet(policy, test_envs_dict, args.buffer_size, args.test_num,
                                      preprocess_fn=state_tracker.build_state,
                                      exploration_noise=args.exploration_noise,
                                      force_length=args.force_length)

    return policy, test_collector_set, optim


def learn_policy(args, env, policy, buffer, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH, logger_path):
    # log
    # t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    # log_file = f'seed_{args.seed}_{t0}-{args.env.replace("-", "_")}_cql'
    # log_path = os.path.join(args.logdir, args.env, 'cql', log_file)
    # writer = SummaryWriter(log_path)
    # writer.add_text("args", str(args))
    # logger1 = TensorboardLogger(writer)
    # def save_best_fn(policy):
    #     torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    df_val, df_user_val, df_item_val, list_feat = get_val_data(args.env)
    item_feat_domination = get_training_item_domination(args.env)
    policy.callbacks = [
        Callback_Coverage_Count(test_collector_set, df_item_val, args.need_transform, item_feat_domination,
                                lbe_item=env.lbe_item if args.need_transform else None, top_rate=args.top_rate),
        LoggerCallback_Policy(logger_path, args.force_length)]
    model_save_path = os.path.join(MODEL_SAVE_PATH, "{}_{}.pt".format(args.model_name, args.message))

    result = offline_trainer(
        policy,
        buffer,
        test_collector_set,
        args.epoch,
        args.step_per_epoch,
        args.test_num,
        args.batch_size,
        # save_best_fn=save_best_fn,
        # stop_fn=stop_fn,
        # logger=logger1,
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
    env, buffer, test_envs_dict = prepare_buffer_via_offline_data(args)

    # %% 3. Setup policy
    state_tracker = setup_offline_state_tracker(args, env, buffer, test_envs_dict)
    policy, test_collector_set, optim = setup_policy_model(args, state_tracker, buffer, test_envs_dict)

    # %% 4. Learn policy
    learn_policy(args, env, policy, buffer, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH, logger_path)


if __name__ == '__main__':
    args_all = get_args_all()
    args = get_common_args(args_all)
    args_CQL = get_args_CQL()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_CQL.__dict__)

    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
