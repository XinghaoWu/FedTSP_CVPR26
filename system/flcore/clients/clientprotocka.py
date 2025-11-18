import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientprotoeval import clientProtoEval, agg_func
from flcore.clients.clientstruct import cka_loss
from flcore.clients.clientbase import load_item, save_item
from collections import defaultdict
import torch.nn.functional as F
import math

def cka_loss_detach(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12):
    """
    线性 CKA 的 Gram 版，分母的范数 stop-grad。
    X, Y: (N, D)
    返回: 1 - CKA
    """
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)

    Gx = Xc @ Xc.T      # (N, N)
    Gy = Yc @ Yc.T      # (N, N)

    num = torch.trace(Gx @ Gy)  # = <Gx, Gy>_F

    den_x = torch.norm(Gx).detach()
    den_y = torch.norm(Gy).detach()
    den = (den_x * den_y).clamp_min(eps)

    cka = num / den
    return 1.0 - cka

def cka_loss_detach_cov(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12):
    """
    线性 CKA 的协方差式（推荐）：在 D×D 上计算，分母 stop-grad。
    X, Y: (N, D)
    返回: 1 - CKA
    """
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)

    XtY = Xc.T @ Yc     # (D, D)
    XtX = Xc.T @ Xc     # (D, D)
    YtY = Yc.T @ Yc     # (D, D)

    num = (XtY * XtY).sum()  # ||XtY||_F^2

    den_x = torch.sqrt((XtX * XtX).sum()).detach()
    den_y = torch.sqrt((YtY * YtY).sum()).detach()
    den = (den_x * den_y).clamp_min(eps)

    cka = num / den
    return 1.0 - cka


class clientProtoCKA(clientProtoEval):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.lamda = args.lamda
        self.tag = args.tag

        self.model = load_item(self.role, 'model', self.save_folder_name)

        self.global_prototype = None
        self.global_round = 0


    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        self.model = model
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        # optimizer_logit_scale = torch.optim.SGD([self.logit_scale], lr=self.learning_rate)
        model.to(self.device)
        model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        protos = defaultdict(list)
        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = model.base(x)
                output = model.head(rep)
                loss = self.loss(output, y)

                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[i, :] = global_protos[y_c].data
                    if self.tag == 'detach':
                        cka_loss_val = cka_loss_detach(rep, proto_new)
                    elif self.tag == 'detach_cov':
                        cka_loss_val = cka_loss_detach_cov(rep, proto_new)
                        print(f'detach_cov', end=' ')
                    elif self.tag == '':
                        cka_loss_val = cka_loss(rep, proto_new)
                        print(f'default cka', end=' ')
                    else: raise NotImplementedError
                    loss += self.lamda * cka_loss_val
                    

                # only accumulate features in the last epoch
                if step == max_local_epochs - 1:
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        protos[y_c].append(rep[i, :].detach().data)

                optimizer.zero_grad()
                # optimizer_logit_scale.zero_grad()
                loss.backward()
                optimizer.step()
                # optimizer_logit_scale.step()

        save_item(agg_func(protos), self.role, 'protos', self.save_folder_name)
        save_item(model, self.role, 'model', self.save_folder_name)
        model.to('cpu')

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def test_metrics(self, specific_testloader=None):
        testloader = self.load_test_data() if specific_testloader is None else specific_testloader
        model = self.model
        model.to(self.device)
        model.eval()

        test_acc = 0
        test_num = 0


        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

        model.to('cpu')
        return test_acc, test_num, 0


    def train_metrics(self):
        trainloader = self.load_train_data()
        model = self.model
        # global_protos = self.global_prototype
        model.to(self.device)
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = model.base(x)
                output = model.head(rep)
                loss = self.loss(output, y)

                # loss_dis_center = self.loss_dis_center_func(rep, global_protos, y)
                # loss += self.get_transfer_lambda(self.global_round) * loss_dis_center

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        model.to('cpu')
        return losses, train_num

