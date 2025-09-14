import pandas as pd
import numpy as np
import os
import h5py
import pdb
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedGroupKFold
import torch
from datetime import datetime 
import logging  # 新增日志模块
import random
from timm.layers import DropPath, trunc_normal_
def setup_logger(log_dir, cancer_type, prediction_type, run_id=None):
    """设置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建唯一日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_id:
        log_filename = f"{cancer_type}_{prediction_type}_{run_id}_{timestamp}.log"
    else:
        log_filename = f"{cancer_type}_{prediction_type}_{timestamp}.log"
    
    log_path = os.path.join(log_dir, log_filename)
    
    # 创建日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger, log_path


def custom_collate_fn(batch):
    """过滤无效样本的collate函数"""
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def patient_kfold(dataset, n_splits=5, random_state=0, valid_size=0.1):
    """Perform cross-validation with patient split.
    """
    indices = np.arange(len(dataset))

    patients_unique = np.unique(dataset.patient_id)

    skf = KFold(n_splits, shuffle=True, random_state=random_state)
    ind = skf.split(patients_unique)

    train_idx = []
    valid_idx = []
    test_idx = []

    for k, (ind_train, ind_test) in enumerate(ind):

        patients_train = patients_unique[ind_train]
        patients_test = patients_unique[ind_test]

        test_idx.append(indices[np.any(np.array(dataset.patient_id)[:, np.newaxis] ==
                                       np.array(patients_test)[np.newaxis], axis=1)])

        if valid_size > 0:
            patients_train, patients_valid = train_test_split(
                patients_train, test_size=valid_size, random_state=0)
            valid_idx.append(indices[np.any(np.array(dataset.patient_id)[:, np.newaxis] ==
                                            np.array(patients_valid)[np.newaxis], axis=1)])

        train_idx.append(indices[np.any(np.array(dataset.patient_id)[:, np.newaxis] ==
                                        np.array(patients_train)[np.newaxis], axis=1)])

    return train_idx, valid_idx, test_idx
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)