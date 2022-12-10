# -*- coding: utf-8 -*-
# @Time    : 2021/8/2 7:56 下午
# @Author  : Chongming GAO
# @FileName: utils.py


import logzero
import torch
from logzero import logger
import os
from tensorflow.python.keras.callbacks import Callback

from util.upload import my_upload


def create_dir(create_dirs):
    """
    创建所需要的目录
    """
    for dir in create_dirs:
        if not os.path.exists(dir):
            logger.info('Create dir: %s' % dir)
            try:
                os.mkdir(dir)
            except FileExistsError:
                print("The dir [{}] already existed".format(dir))


# my nas docker path
REMOTE_ROOT = "/root/Rethink_RL4RS"
class LoggerCallback_Update(Callback):
    def __init__(self, logger_path):
        super().__init__()
        self.LOCAL_PATH = logger_path
        self.REMOTE_ROOT = REMOTE_ROOT
        self.REMOTE_PATH = os.path.join(REMOTE_ROOT, os.path.dirname(logger_path))

    def on_epoch_end(self, epoch, logs=None):
        # 1. write logger
        logger.info("Epoch: [{}], Info: [{}]".format(epoch, logs))

        # 2. upload logger
        # self.upload_logger()

    def upload_logger(self):
        try:
            my_upload(self.LOCAL_PATH, self.REMOTE_PATH, self.REMOTE_ROOT)
        except Exception:
            print("Failed: Uploading file [{}] to remote path [{}]".format(self.LOCAL_PATH, self.REMOTE_PATH))

class Callback_saveEmbedding(Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path

    def on_train_end(self, model):
        torch.save(model.state_dict(), self.save_path)
        logger.info("Model embedding has been saved at {}".format(self.save_path))


class LoggerCallback_RL(LoggerCallback_Update):
    def __init__(self, logger_path):
        super().__init__(logger_path)

    def on_epoch_end(self, epoch, logs=None):
        num_test = logs["n/ep"]
        len_tra = logs["n/st"] / num_test
        R_tra = logs["rew"]
        ctr = R_tra / len_tra

        result = dict()
        result['num_test'] = num_test
        result['len_tra'] = len_tra
        result['R_tra'] = R_tra
        result['ctr'] = f"{ctr:.3f}"

        # 1. write logger
        logger.info("Epoch: [{}], Info: [{}]".format(epoch, result))

        # 2. upload logger
        # self.upload_logger()


class LoggerCallback_Policy():
    def __init__(self, logger_path):
        self.LOCAL_PATH = logger_path
        self.REMOTE_ROOT = REMOTE_ROOT
        self.REMOTE_PATH = os.path.join(REMOTE_ROOT, os.path.dirname(logger_path))

    def on_epoch_begin(self, epoch, **kwargs):
        pass

    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_epoch_end(self, epoch, results=None, **kwargs):
        num_test = results["n/ep"]
        len_tra = results["n/st"] / num_test
        R_tra = results["rew"]
        ctr = R_tra / len_tra

        result = dict()
        result['num_test'] = num_test
        result['CV'] = f"{results['CV']:.3f}"
        result['CV_turn'] = f"{results['CV_turn']:.3f}"
        result['ctr'] = f"{ctr:.3f}"
        result['len_tra'] = len_tra
        result['R_tra'] = R_tra

        # 1. write logger
        logger.info("Epoch: [{}], Info: [{}]".format(epoch, result))

        # 2. upload logger
        # self.upload_logger()


def save_model_fn(epoch, policy, model_save_path, optim, state_tracker, is_save=False):
    if not is_save:
        return
    model_save_path = model_save_path[:-3] + "-e{}".format(epoch) + model_save_path[-3:]
    # torch.save(model.state_dict(), model_save_path)
    torch.save({
        'policy': policy.state_dict(),
        'optim_RL': optim[0].state_dict(),
        'optim_state': optim[1].state_dict(),
        'state_tracker': state_tracker.state_dict(),
    }, model_save_path)