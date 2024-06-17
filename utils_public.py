import time
import torch
import torch.optim as optim
import copy
import numpy as np
from torch import nn
from types import SimpleNamespace
from sklearn.metrics import roc_auc_score, average_precision_score,recall_score, f1_score, matthews_corrcoef, confusion_matrix,precision_score
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

class TrainProcessor:
    def __init__(self, model, loaders, args):
        self.model = model
        self.train_loader, self.val_loader, self.test_loader = loaders
        self.args = args
        tr_type = getattr(self.args,"type",0)
        if tr_type != 2:
            self.optimizer, self.scheduler = self.build_optimizer()

    def build_optimizer(self):
        args = self.args
        # return an iterator
        filter_fn = filter(lambda p: p.requires_grad, self.model.parameters())  # params is a generator (kind of iterator)

        # optimizer
        weight_decay = args.weight_decay
        if args.opt == 'adam':
            optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
        elif args.opt == 'sgd':
            optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
        elif args.opt == 'rmsprop':
            optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
        elif args.opt == 'adagrad':
            optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)

        # scheduler
        if args.opt_scheduler == 'none':
            return None, optimizer
        elif args.opt_scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
        elif args.opt_scheduler == 'reduceOnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             patience=args.lr_decay_patience,
                                                             factor=args.lr_decay_factor)
        else:
            raise Exception('Unknown optimizer type')

        return optimizer, scheduler
    @torch.no_grad()
    def test_for_predict(self, model_list, dataloader):
        [model.eval() for model in model_list]

        pred_ls = []
        y_ls = []
        for batch in dataloader:
            batch = batch.to(self.args.device)
            stacked_simple_tensors = torch.stack([model_list[i](batch) for i in range(len(model_list))], dim=0)
            avg_simple_tensor = torch.mean(stacked_simple_tensors, dim=0, keepdim=False)
            pred_ls.append(avg_simple_tensor)
            y_ls.append(batch.y)

        pred = torch.cat(pred_ls, dim=0).reshape(-1)  # 预测出来的y=1的概率
        y = torch.cat(y_ls, dim=0).reshape(-1)  # 真实的标签
        
        # 保存PT文件（保留这一个，记得把PT文件的名字改了，避免重复）
        torch.save((y, pred), './ablation/ST/xiaorong3/tensors2.pt')
        metrics = {}
        # metrics['loss'] = model.loss(pred, y).item()
        metrics['acc'] = (pred.round() == y).sum() / len(pred)

        y = y.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()

        pred_labels = (pred >= 0.5).astype(float)
        
        metrics['auroc'] = roc_auc_score(y, pred)
        metrics['auprc'] = average_precision_score(y, pred)
        metrics['sn'] = recall_score(y, pred_labels)
        tn, fp, fn, tp = confusion_matrix(y, pred_labels).ravel()
        metrics['sp'] = tn / (tn + fp)
        metrics['mcc'] = matthews_corrcoef(y, pred_labels)
        metrics['f1'] = f1_score(y, pred_labels)
        metrics['precision'] = precision_score(y, pred_labels)
        metrics['recall'] = metrics['sn']

        # metrics['f1'] = get_f1(y, pred)

        return SimpleNamespace(**metrics)