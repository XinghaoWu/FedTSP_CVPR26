import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict
import torch.nn.functional as F

# ===================== Import struct_loss functions from clientstruct =====================
def pairwise_l2_distance(X, eps: float = 1e-12):
    """
    Compute pairwise L2 distance matrix for X.
    X: (N, D)
    Return: Dmat: (N, N), Dmat[i, j] = ||x_i - x_j||_2
    """
    X_norm_sq = (X ** 2).sum(dim=1, keepdim=True)
    dist_sq = X_norm_sq + X_norm_sq.T - 2.0 * (X @ X.T)
    dist_sq = torch.clamp(dist_sq, min=0.0)
    Dmat = torch.sqrt(dist_sq + eps)
    return Dmat

def pairwise_l2_distance_sq(X):
    """
    Compute pairwise squared L2 distance matrix for X.
    X: (N, D)
    Return: Dmat_sq: (N, N), Dmat_sq[i, j] = ||x_i - x_j||_2^2
    """
    X_norm_sq = (X ** 2).sum(dim=1, keepdim=True)
    dist_sq = X_norm_sq + X_norm_sq.T - 2.0 * (X @ X.T)
    dist_sq = torch.clamp(dist_sq, min=0.0)
    return dist_sq

def cka_loss(X, Y):
    """
    Calculate the CKA loss between two feature sets X and Y.
    """
    X_centered = X - X.mean(dim=0, keepdim=True)
    Y_centered = Y - Y.mean(dim=0, keepdim=True)
    gram_X = X_centered @ X_centered.T
    gram_Y = Y_centered @ Y_centered.T
    cka_value = torch.trace(gram_X @ gram_Y) / (torch.norm(gram_X) * torch.norm(gram_Y))
    return 1 - cka_value

def gram_mse_loss(X, Y, center: bool = True, normalize: bool = True):
    """
    Gram-MSE structural loss.
    """
    if normalize:
        X = F.normalize(X, dim=1)
        Y = F.normalize(Y, dim=1)
    if center:
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
    gram_X = X @ X.T
    gram_Y = Y @ Y.T
    return F.mse_loss(gram_X, gram_Y)

def rdm_mse_loss(X, Y):
    """
    RDM-MSE structural loss.
    """
    D_X = pairwise_l2_distance(X)
    D_Y = pairwise_l2_distance(Y)
    return F.mse_loss(D_X, D_Y)

def rdm_mse_sq_loss(X, Y, normalize: bool = False):
    """
    RDM-MSE structural loss.
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
    """
    N = X.size(0)
    if N <= 1:
        return torch.tensor(0.0, device=X.device)
    D_X = pairwise_l2_distance(X)
    D_Y = pairwise_l2_distance(Y)
    idx = torch.triu_indices(N, N, offset=1, device=X.device)
    vX = D_X[idx[0], idx[1]]
    vY = D_Y[idx[0], idx[1]]
    denom = (vX.norm() * vY.norm()).clamp_min(eps)
    cos_sim = (vX @ vY) / denom
    return 1.0 - cos_sim

def rdm_cos_sq_loss(X, Y, eps: float = 1e-12, normalize: bool = False):
    """
    RDM-Cos structural loss.
    """
    N = X.size(0)
    if N <= 1:
        return torch.tensor(0.0, device=X.device)
    if normalize:
        X = F.normalize(X, dim=1)
        Y = F.normalize(Y, dim=1)
    D_X = pairwise_l2_distance_sq(X)
    D_Y = pairwise_l2_distance_sq(Y)
    idx = torch.triu_indices(N, N, offset=1, device=X.device)
    vX = D_X[idx[0], idx[1]]
    vY = D_Y[idx[0], idx[1]]
    denom = (vX.norm() * vY.norm()).clamp_min(eps)
    cos_sim = (vX @ vY) / denom
    return 1.0 - cos_sim

def coord_mse_loss(X, Y):
    """
    Coordinate-level MSE alignment.
    """
    return F.mse_loss(X, Y)

def coord_cosine_loss(X, Y):
    """
    Coordinate-level cosine alignment.
    """
    cos = F.cosine_similarity(X, Y, dim=1)
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
# ===================== End of struct_loss functions =====================


class clientTGPStruct(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda
        self.gamma = args.gamma

        self.model = load_item(self.role, 'model', self.save_folder_name)

        self.logit_scale = nn.Parameter(torch.tensor(4.6052))


    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        self.model = model
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        # optimizer_logit_scale = torch.optim.SGD([self.logit_scale], lr=self.learning_rate)
        # model.to(self.device)
        model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

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

                # Use version=5 struct_loss: class-level alignment
                if global_protos is not None:
                    classes_in_batch = torch.unique(y)
                    n_cls = classes_in_batch.size(0)

                    proto_local = torch.zeros(n_cls, rep.size(1), device=self.device)
                    for idx_c, c in enumerate(classes_in_batch):
                        mask = (y == c)
                        proto_local[idx_c] = rep[mask].mean(0)

                    selected_global = []
                    valid_idx = []
                    for idx_c, c in enumerate(classes_in_batch):
                        if (global_protos is not None and
                            type(global_protos[c.item()]) != type([])):
                            selected_global.append(global_protos[c.item()].to(self.device))
                            valid_idx.append(idx_c)

                    if selected_global and len(selected_global) > 1:
                        proto_global = torch.stack(selected_global, 0)
                        proto_local = proto_local[valid_idx]
                        cka_loss_val = struct_loss(proto_local, proto_global, mode=self.args.struct_loss_type)
                        loss += cka_loss_val * self.lamda

                    # Use version=5 struct_loss: instance-level alignment
                    proto_new = copy.deepcopy(rep.detach())
                    for idx, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[idx, :] = global_protos[y_c].data

                    cka_loss_val = struct_loss(rep, proto_new, mode=self.args.struct_loss_type)
                    loss += cka_loss_val * self.gamma

                optimizer.zero_grad()
                # optimizer_logit_scale.zero_grad()
                loss.backward()
                optimizer.step()
                # optimizer_logit_scale.step()

        self.collect_protos()
        save_item(model, self.role, 'model', self.save_folder_name)
        self.model.to('cpu')

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        # print(f'client: {self.id}. logit_scale: {self.logit_scale.item()}')


    def collect_protos(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        save_item(agg_func(protos), self.role, 'protos', self.save_folder_name)

    # def test_metrics(self):
    #     testloader = self.load_test_data()
    #     model = load_item(self.role, 'model', self.save_folder_name)
    #     global_protos = load_item('Server', 'global_protos', self.save_folder_name)
    #     model.eval()

    #     test_acc = 0
    #     test_num = 0

    #     if global_protos is not None:
    #         with torch.no_grad():
    #             for x, y in testloader:
    #                 if type(x) == type([]):
    #                     x[0] = x[0].to(self.device)
    #                 else:
    #                     x = x.to(self.device)
    #                 y = y.to(self.device)
    #                 rep = model.base(x)

    #                 output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
    #                 for i, r in enumerate(rep):
    #                     for j, pro in global_protos.items():
    #                         if type(pro) != type([]):
    #                             output[i, j] = self.loss_mse(r, pro)

    #                 test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()
    #                 test_num += y.shape[0]

    #         return test_acc, test_num, 0
    #     else:
    #         return 0, 1e-5, 0

    def test_metrics(self, specific_testloader=None):
        testloader = self.load_test_data() if specific_testloader is None else specific_testloader
        # model = self.model
        model = load_item(self.role, 'model', self.save_folder_name)
        # global_protos = load_item('Server', 'global_protos', self.save_folder_name)
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
    
            return test_acc, test_num, 0


    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        # model.to(self.device)
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

                # Use version=5 struct_loss: class-level alignment
                if global_protos is not None:
                    classes_in_batch = torch.unique(y)
                    n_cls = classes_in_batch.size(0)

                    proto_local = torch.zeros(n_cls, rep.size(1), device=self.device)
                    for idx_c, c in enumerate(classes_in_batch):
                        mask = (y == c)
                        proto_local[idx_c] = rep[mask].mean(0)

                    selected_global = []
                    valid_idx = []
                    for idx_c, c in enumerate(classes_in_batch):
                        if (global_protos is not None and
                            type(global_protos[c.item()]) != type([])):
                            selected_global.append(global_protos[c.item()].to(self.device))
                            valid_idx.append(idx_c)

                    if selected_global and len(selected_global) > 1:
                        proto_global = torch.stack(selected_global, 0)
                        proto_local = proto_local[valid_idx]
                        cka_loss_val = struct_loss(proto_local, proto_global, mode=self.args.struct_loss_type)
                        loss += cka_loss_val * self.lamda

                    # Use version=5 struct_loss: instance-level alignment
                    proto_new = copy.deepcopy(rep.detach())
                    for idx, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[idx, :] = global_protos[y_c].data

                    cka_loss_val = struct_loss(rep, proto_new, mode=self.args.struct_loss_type)
                    loss += cka_loss_val * self.gamma

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num


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