import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict
import torch.nn.functional as F
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader

def pairwise_l2_distance(X, eps: float = 1e-12):
    """
    Compute pairwise L2 distance matrix for X.
    X: (N, D)
    Return: Dmat: (N, N), Dmat[i, j] = ||x_i - x_j||_2
    """
    # squared norms (N, 1)
    X_norm_sq = (X ** 2).sum(dim=1, keepdim=True)          # (N, 1)
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i x_j^T
    dist_sq = X_norm_sq + X_norm_sq.T - 2.0 * (X @ X.T)    # (N, N)
    dist_sq = torch.clamp(dist_sq, min=0.0)
    Dmat = torch.sqrt(dist_sq + eps)
    return Dmat

def pairwise_l2_distance_sq(X):
    """
    Compute pairwise squared L2 distance matrix for X.
    X: (N, D)
    Return: Dmat_sq: (N, N), Dmat_sq[i, j] = ||x_i - x_j||_2^2
    """
    X_norm_sq = (X ** 2).sum(dim=1, keepdim=True)             # (N, 1)
    dist_sq = X_norm_sq + X_norm_sq.T - 2.0 * (X @ X.T)       # (N, N)
    dist_sq = torch.clamp(dist_sq, min=0.0)
    return dist_sq

# ===================== 结构对齐家族 =====================

def cka_loss(X, Y):
    """
    Calculate the CKA loss between two feature sets X and Y.
    """
    # Centering the matrices
    X_centered = X - X.mean(dim=0, keepdim=True)
    Y_centered = Y - Y.mean(dim=0, keepdim=True)

    # Compute Gram matrices (similarity matrices)
    gram_X = X_centered @ X_centered.T
    gram_Y = Y_centered @ Y_centered.T

    # Compute the CKA score
    cka_value = torch.trace(gram_X @ gram_Y) / (torch.norm(gram_X) * torch.norm(gram_Y))

    return 1 - cka_value  # Return 1 - CKA to make it a loss

def gram_mse_loss(X, Y, center: bool = True, normalize: bool = True):
    """
    Gram-MSE structural loss.
    S: inner-product Gram matrix (optionally centered)
    D: L2 (Frobenius MSE between Gram matrices)
    X, Y: (N, D)
    """
    if normalize:
        X = F.normalize(X, dim=1)
        Y = F.normalize(Y, dim=1)
    if center:
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

    gram_X = X @ X.T        # (N, N)
    gram_Y = Y @ Y.T        # (N, N)
    return F.mse_loss(gram_X, gram_Y)

def rdm_mse_loss(X, Y):
    """
    RDM-MSE structural loss.
    S: L2 distance matrix D(i,j) = ||x_i - x_j||_2
    D: L2 (Frobenius MSE between distance matrices)
    X, Y: (N, D)
    """
    D_X = pairwise_l2_distance(X)   # (N, N)
    D_Y = pairwise_l2_distance(Y)   # (N, N)
    return F.mse_loss(D_X, D_Y)

def rdm_mse_sq_loss(X, Y, normalize: bool = False):
    """
    RDM-MSE structural loss.
    S: L2 distance matrix D(i,j) = ||x_i - x_j||_2
    D: L2 (Frobenius MSE between distance matrices)
    X, Y: (N, D)
    """
    if normalize:
        X = F.normalize(X, dim=1)
        Y = F.normalize(Y, dim=1)
    D_X = pairwise_l2_distance_sq(X)
    D_Y = pairwise_l2_distance_sq(Y)
    return F.mse_loss(D_X, D_Y)

def rdm_cos_loss(X, Y, eps: float = 1e-12):
    """
    RDM-Cos structural loss.
    S: L2 distance matrix D(i,j) = ||x_i - x_j||_2
    D: cosine similarity between vectorized upper-triangular distance entries
    X, Y: (N, D)
    """
    N = X.size(0)
    if N <= 1:
        # too few points to define a meaningful structure
        return torch.tensor(0.0, device=X.device)

    D_X = pairwise_l2_distance(X)   # (N, N)
    D_Y = pairwise_l2_distance(Y)   # (N, N)

    # take upper triangle (i < j) to avoid duplicates and zeros on diagonal
    idx = torch.triu_indices(N, N, offset=1, device=X.device)
    vX = D_X[idx[0], idx[1]]        # (N*(N-1)/2,)
    vY = D_Y[idx[0], idx[1]]        # (N*(N-1)/2,)

    denom = (vX.norm() * vY.norm()).clamp_min(eps)
    cos_sim = (vX @ vY) / denom
    return 1.0 - cos_sim

def rdm_cos_sq_loss(X, Y, eps: float = 1e-12, normalize: bool = False):
    """
    RDM-Cos structural loss.
    S: L2 distance matrix D(i,j) = ||x_i - x_j||_2
    D: cosine similarity between vectorized upper-triangular distance entries
    X, Y: (N, D)
    """
    N = X.size(0)
    if N <= 1:
        # too few points to define a meaningful structure
        return torch.tensor(0.0, device=X.device)

    if normalize:
        X = F.normalize(X, dim=1)
        Y = F.normalize(Y, dim=1)

    D_X = pairwise_l2_distance_sq(X)   # (N, N)
    D_Y = pairwise_l2_distance_sq(Y)   # (N, N)

    # take upper triangle (i < j) to avoid duplicates and zeros on diagonal
    idx = torch.triu_indices(N, N, offset=1, device=X.device)
    vX = D_X[idx[0], idx[1]]        # (N*(N-1)/2,)
    vY = D_Y[idx[0], idx[1]]        # (N*(N-1)/2,)

    denom = (vX.norm() * vY.norm()).clamp_min(eps)
    cos_sim = (vX @ vY) / denom
    return 1.0 - cos_sim

# ===================== 坐标对齐：MSE / Cosine =====================
def coord_mse_loss(X, Y):
    """
    Coordinate-level MSE alignment.
    直接对每个样本 / prototype 做点对点 L2，对应传统的 MSE feature alignment。
    X, Y: (N, D)
    """
    return F.mse_loss(X, Y)


def coord_cosine_loss(X, Y):
    """
    Coordinate-level cosine alignment.
    对每一行计算 cosine similarity，然后取平均，loss = 1 - mean_cos。
    X, Y: (N, D)
    """
    # N==0 理论上不会出现；N==1 也没问题
    cos = F.cosine_similarity(X, Y, dim=1)   # (N,)
    return 1.0 - cos.mean()

def struct_loss(X, Y, mode: str = "cka"):
    if mode == "cka":
        return cka_loss(X, Y)
    elif mode == "gram_mse":
        return gram_mse_loss(X, Y)
    elif mode == "rdm_mse":
        return rdm_mse_loss(X, Y)
    elif mode == "rdm_cos":
        return rdm_cos_loss(X, Y)
    elif mode == "rdm_mse_sq":
        return rdm_mse_sq_loss(X, Y)
    elif mode == "rdm_cos_sq":
        return rdm_cos_sq_loss(X, Y)
    elif mode == "rdm_mse_sq_norm":
        return rdm_mse_sq_loss(X, Y, normalize=True)
    elif mode == "rdm_cos_sq_norm":
        return rdm_cos_sq_loss(X, Y, normalize=True)
    elif mode == "mse":
        return coord_mse_loss(X, Y)
    elif mode == "cosine":
        return coord_cosine_loss(X, Y)
    else:
        raise ValueError(f"Unknown struct_loss mode: {mode}")

class clientStruct(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda
        self.gamma = args.gamma

        self.model = load_item(self.role, 'model', self.save_folder_name)

        self.logit_scale = nn.Parameter(torch.tensor(4.6052))
        self.global_proto = None
        self.local_proto = None


    def train(self):
        trainloader = self.load_train_data()
        # model = load_item(self.role, 'model', self.save_folder_name)
        # self.model = model
        model = self.model
        # global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        global_protos = copy.deepcopy(self.global_proto)
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

                if self.args.version == 1:
                    if global_protos is not None:
                        proto_new = copy.deepcopy(rep.detach())
                        for i, yy in enumerate(y):
                            y_c = yy.item()
                            if type(global_protos[y_c]) != type([]):
                                proto_new[i, :] = global_protos[y_c].data
                        
                        cka_loss_val = struct_loss(rep, proto_new, mode=self.args.struct_loss_type)
                        # print(f'CKA Loss:{cka_loss_val.item()}')
                        loss += cka_loss_val * self.lamda  # Add CKA loss to the total loss

                        # loss += self.loss_mse(proto_new, rep) * self.lamda
                elif self.args.version in [2, 3, 4, 5]:
                    classes_in_batch = torch.unique(y)                 # e.g. [1, 3, 7]
                    n_cls = classes_in_batch.size(0)

                    proto_local = torch.zeros(n_cls, rep.size(1), device=self.device)
                    for idx_c, c in enumerate(classes_in_batch):
                        mask = (y == c)                                # bool (B,)
                        proto_local[idx_c] = rep[mask].mean(0)         # (D,)
                    
                    selected_global = []
                    valid_idx       = []                               # 记录真正有全局原型的行
                    for idx_c, c in enumerate(classes_in_batch):
                        if (global_protos is not None and
                            type(global_protos[c.item()]) != type([])):    # 该类在服务器有原型
                            selected_global.append(global_protos[c.item()].to(self.device))
                            valid_idx.append(idx_c)
                    # print(f'selected global:{len(selected_global)}') +2
                    if selected_global and len(selected_global) > 1:                                # 至少存在 1 个有效类
                        proto_global = torch.stack(selected_global, 0)     # (N_valid, D)
                        proto_local  = proto_local[valid_idx]              # (N_valid, D)   # todo: 需要检查local和global是否对应

                        # --------- ④ 计算 CKA loss 并加入总损失 ---------
                        cka_loss_val = struct_loss(proto_local, proto_global, mode=self.args.struct_loss_type)
                        # print(f'cka loss:{cka_loss_val.item()}')
                        loss += cka_loss_val * self.lamda

                    if self.args.version in [4, 5]:
                        if global_protos is not None:
                            proto_new = copy.deepcopy(rep.detach())
                            for i, yy in enumerate(y):
                                y_c = yy.item()
                                if type(global_protos[y_c]) != type([]):
                                    proto_new[i, :] = global_protos[y_c].data
                        
                            cka_loss_val = struct_loss(rep, proto_new, mode=self.args.struct_loss_type)
                            # print(f'CKA Loss:{cka_loss_val.item()}')
                            loss += cka_loss_val * self.gamma  # Add CKA loss to the total loss
                    
                # In Version 5, only calculate prototypes from the features in the last epoch
                accumulate_features = True
                if self.args.version in [5] and step != max_local_epochs - 1:
                        accumulate_features = False
                if accumulate_features:
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        protos[y_c].append(rep[i, :].detach().data)

                optimizer.zero_grad()
                # optimizer_logit_scale.zero_grad()
                loss.backward()
                optimizer.step()
                # optimizer_logit_scale.step()

        # save_item(agg_func(protos), self.role, 'protos', self.save_folder_name)
        # save_item(model, self.role, 'model', self.save_folder_name)
        self.local_proto = agg_func(protos)
        model.to('cpu')

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        # print(f'client: {self.id}. logit_scale: {self.logit_scale.item()}')

    def test_metrics(self, specific_testloader=None):
        testloader = self.load_test_data() if specific_testloader is None else specific_testloader
        model = self.model
        # model = load_item(self.role, 'model', self.save_folder_name)
        # global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        model.eval()
        model.to(self.device)
    
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
        # model = load_item(self.role, 'model', self.save_folder_name)
        model = self.model
        # global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        global_protos = copy.deepcopy(self.global_proto)
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

                # Use args.version=5 loss calculation
                classes_in_batch = torch.unique(y)                 # e.g. [1, 3, 7]
                n_cls = classes_in_batch.size(0)

                proto_local = torch.zeros(n_cls, rep.size(1), device=self.device)
                for idx_c, c in enumerate(classes_in_batch):
                    mask = (y == c)                                # bool (B,)
                    proto_local[idx_c] = rep[mask].mean(0)         # (D,)

                selected_global = []
                valid_idx       = []                               # 记录真正有全局原型的行
                for idx_c, c in enumerate(classes_in_batch):
                    if (global_protos is not None and
                        type(global_protos[c.item()]) != type([])):    # 该类在服务器有原型
                        selected_global.append(global_protos[c.item()].to(self.device))
                        valid_idx.append(idx_c)

                if selected_global and len(selected_global) > 1:                                # 至少存在 1 个有效类
                    proto_global = torch.stack(selected_global, 0)     # (N_valid, D)
                    proto_local  = proto_local[valid_idx]              # (N_valid, D)

                    # --------- ④ 计算 CKA loss 并加入总损失 ---------
                    cka_loss_val = struct_loss(proto_local, proto_global, mode=self.args.struct_loss_type)
                    loss += cka_loss_val * self.lamda

                # Add instance-level alignment loss (gamma term)
                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[i, :] = global_protos[y_c].data

                    cka_loss_val = struct_loss(rep, proto_new, mode=self.args.struct_loss_type)
                    loss += cka_loss_val * self.gamma  # Add CKA loss to the total loss

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        model.to('cpu')
        return losses, train_num

    def get_local_prototpye(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()
        protos = defaultdict(list)
        for i, (x, y) in enumerate(trainloader):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            rep = self.model.base(x)

            for i, yy in enumerate(y):
                y_c = yy.item()
                protos[y_c].append(rep[i, :].detach().data)

        self.model.to('cpu')

        proto_dict = agg_func(protos)
        return proto_dict
        # label_order = [i for i in range(self.args.num_classes)]
        # K = len(label_order)
        # D = next(iter(proto_dict.values())).numel()          # 特征维
        # mat = np.zeros((K, D), dtype=np.float32)            # 先全 0
        # for i, lbl in enumerate(label_order):
        #     if lbl in proto_dict:                           # 该客户端有该类
        #         mat[i] = proto_dict[lbl].detach().cpu().numpy()
        #         # 若你想用“缺失类掩码”而非 0 占位，可在这里记录一个 mask
        # # print(f'Client {self.id}, prototype {mat}')
        # return mat


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos