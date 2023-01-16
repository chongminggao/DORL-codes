import argparse
import sys
import traceback

from run_Policy_Main import get_args_all, main

# import pytest
sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from core.configs import get_common_args

import logzero

def get_args_ips_policy():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_model_name", type=str, default="DeepFM-IPS")
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument("--model_name", type=str, default="IPS")
    parser.add_argument("--read_message", type=str, default="DeepFM-IPS")
    parser.add_argument("--message", type=str, default="IPS")
    args = parser.parse_known_args()[0]
    return args

if __name__ == '__main__':
    args_all = get_args_all()
    args = get_common_args(args_all)
    args_ips = get_args_ips_policy()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_ips.__dict__)

    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
