# -*- coding: utf-8 -*-
# @Time    : 2022/11/14 14:50
# @Author  : Chongming GAO
# @FileName: user_model_ensemble.py
from multiprocessing import Pool, Process

from core.user_model_pairwise_variance import UserModel_Pairwise_Variance
from logzero import logger


# def collect_res(res_list, res):


class EnsembleModel():
    def __init__(self, num_models, *args, **kwargs):

        self.user_models = [UserModel_Pairwise_Variance(*args, **kwargs) for i in range(num_models)]

    def compile(self, *args, **kwargs):
        for model in self.user_models:
            model.compile(*args, **kwargs)

    def compile_RL_test(self, *args, **kwargs):
        for model in self.user_models:
            model.compile_RL_test(*args, **kwargs)

    def fit_data(self, *args, **kwargs):

        # pool = Pool()
        # for model in self.user_models:
        #     res = pool.apply_async(func=fit_data_handler, args=(model,) + args, kwds=kwargs, callback=lambda x: print(x))  # 实例化进程对象
        #     print(res.get())
        #
        # pool.close()
        # pool.join()

        history_list = []
        for model in self.user_models:
            history = model.fit_data(*args, **kwargs)
            # print(history)
            # logger.info(history.history)
            logger.info("\n")
            history_list.append(history.history)

        logger.info("============ Summarized results =============")
        for hist in history_list:
            logger.info(hist)
            # logger.info("\n")

        return history_list


