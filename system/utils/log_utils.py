import numpy as np
import os
import time
from typing import Iterable
import logging

from torch.utils.tensorboard import SummaryWriter


class Logger(SummaryWriter):

    def __init__(self, path: str):
        super(Logger, self).__init__(path)
        self.txt_logger_path = os.path.join(path, 'txt_logger_output.txt')

    def logging(self, s: str) -> None:
        print(s)
        with open(self.txt_logger_path, mode='a') as f:
            f.write('[' + time.asctime(time.localtime(time.time())) + ']    ' + s + '\n')

    def add_scalars_dict(self, prefix: str, dic: dict, rnd: int) -> None:
        str_repr = []
        for k in dic.keys():
            self.add_scalar(f'{prefix}/{k}', dic[k], rnd)
            str_repr.append('{}: {:.2f}'.format(k, dic[k]))
        txt_info = f'[{prefix.upper()}]\t'
        self.logging(txt_info + '\t'.join(str_repr))


class VariableMonitor:

    def __init__(self):
        self.length = {}
        self.dic = {}

    def append(self, item: dict, weight: float = 1) -> None:
        for k in item.keys():
            if k not in self.dic.keys():
                self.dic[k] = []
                self.length[k] = 0
            self.dic[k].append(weight * item[k])
            self.length[k] += weight

    def variable_mean(self) -> dict:
        return {k: sum(self.dic[k]) / self.length[k] for k in self.dic.keys()}


def set_logger(file_path = 'log.txt', handle = 1):
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create Handler
    # type 1: file handler
    # type 2: stream handler
    if handle == 1:
        log_handler = logging.FileHandler(file_path, mode='w', encoding='UTF-8')
    elif handle == 2:
        log_handler = logging.StreamHandler()
    else:
        log_handler = logging.FileHandler(file_path, mode='w', encoding='UTF-8')

    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
    # formatter = logging.Formatter('%(levelname)s - %(message)s')
    log_handler.setFormatter(formatter)

    # Add to logger
    logger.addHandler(log_handler)

    return logger
