import os

import  torch
from types import SimpleNamespace
import numpy as np
import random
from torch_geometric.loader import DataLoader
from utils_public import TrainProcessor
import os
import yaml

random.seed(42); np.random.seed(42)

# 模型路径列表
MODEL_PATHS = [
    "./result-public/Y/1.pth",
    "./result-public/Y/2.pth",
    "./result-public/Y/3.pth",
    "./result-public/Y/4.pth"
]

if __name__ == '__main__':
    # args
    # 从 YAML 文件加载参数
    with open('config.yaml', 'r') as file:
        args = yaml.safe_load(file)
    args = SimpleNamespace(**args)
    print(args)
    len_ls = [33,]
    all_results = {}
    for seq_len in len_ls:
        train_ls, val_ls, test_ls = torch.load(f'/home/atmel/disk_sdb/A_MyCode/workspace/TransPTM/data/DeepPSP/MyDataset/result/Train/Y/4-FOLD/Fold-0-PT/{seq_len}.pt')
        test_data_loader = DataLoader(test_ls, batch_size=args.batch_size, drop_last=True)

        Result = []
        Result_softmax = []
        model_list = []
        for model_path in MODEL_PATHS:
            model_list.append(torch.load(model_path)['model_state_dict'].to(args.device))
        only_test = TrainProcessor(
            model=model_list,
            loaders=[None, None, None],
            args=args
        )
        metrics = only_test.test_for_predict(model_list, test_data_loader)
        print(f"Metrics - Acc: {metrics.acc:.4f} AUROC: {metrics.auroc:.4f} AUPRC: {metrics.auprc:.4f} SN: {metrics.sn:.4f} SP: {metrics.sp:.4f} MCC: {metrics.mcc:.4f} F1: {metrics.f1:.4f} Precision: {metrics.precision:.4f} Recall: {metrics.recall:.4f}")
