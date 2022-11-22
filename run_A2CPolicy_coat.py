import argparse
import os
import traceback
import sys

from run_Policy import get_args_all, prepare_dir_log, prepare_user_model_and_env, setup_policy_model, prepare_envs, \
    learn_policy

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from core.configs import get_common_args
from logzero import logger

try:
    import envpool
except ImportError:
    envpool = None


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="CoatEnv-v0")
    # parser.add_argument('--entropy_window', type=int, nargs='*', default=[0])

    parser.add_argument('--leave_threshold', default=10, type=float)
    parser.add_argument('--num_leave_compute', default=3, type=int)
    parser.add_argument('--max_turn', default=30, type=int)

    # state_tracker
    parser.add_argument('--dim_state', default=20, type=int)
    parser.add_argument('--dim_model', default=64, type=int)
    parser.add_argument('--nhead', default=4, type=int)
    # parser.add_argument('--max_len', default=100, type=int)
    parser.add_argument('--window', default=3, type=int)

    parser.add_argument("--read_message", type=str, default="UM")
    parser.add_argument("--message", type=str, default="A2C_with_emb")

    args = parser.parse_known_args()[0]
    return args


def main(args):
    # %% 1. Prepare the saved path.
    args = get_common_args(args)
    MODEL_SAVE_PATH, logger_path = prepare_dir_log(args)

    # %% 2. Prepare user model and environment
    ensemble_models, alpha_u, beta_i = prepare_user_model_and_env(args)
    env, train_envs, test_envs = prepare_envs(args, ensemble_models, alpha_u, beta_i)

    # %% 3. Setup policy
    policy, train_collector, test_collector, state_tracker, optim = setup_policy_model(args, ensemble_models, env, train_envs, test_envs)

    # %% 4. Learn policy
    learn_policy(args, policy, train_collector, test_collector, state_tracker, optim, MODEL_SAVE_PATH, logger_path)

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

