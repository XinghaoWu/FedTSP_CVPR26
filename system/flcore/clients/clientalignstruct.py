import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item, matrix_to_dict_proto
from collections import defaultdict
import torch.nn.functional as F
import math

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


class clientAlignStruct(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda

        self.model = load_item(self.role, 'model', self.save_folder_name)

        self.global_prototype = None
        self.global_round = 0
        self.loss_dis_center_func = Contrastive_Loss_Center(temperature=1, device=self.device)
        
        self.gamma = args.gamma


    def train(self):
        trainloader = self.load_train_data()
        model = self.model
        model.to(self.device)
        global_protos = self.global_prototype
        proto_dict = matrix_to_dict_proto(self.global_prototype)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
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
                if proto_dict is not None:
                    classes_in_batch = torch.unique(y)
                    n_cls = classes_in_batch.size(0)

                    proto_local = torch.zeros(n_cls, rep.size(1), device=self.device)
                    for idx_c, c in enumerate(classes_in_batch):
                        mask = (y == c)
                        proto_local[idx_c] = rep[mask].mean(0)

                    selected_global = []
                    valid_idx = []
                    for idx_c, c in enumerate(classes_in_batch):
                        if (proto_dict is not None and
                            type(proto_dict[c.item()]) != type([])):
                            selected_global.append(proto_dict[c.item()].to(self.device))
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
                        if type(proto_dict[y_c]) != type([]):
                            proto_new[idx, :] = proto_dict[y_c].data

                    cka_loss_val = struct_loss(rep, proto_new, mode=self.args.struct_loss_type)
                    loss += cka_loss_val * self.gamma

                optimizer.zero_grad()
                # optimizer_logit_scale.zero_grad()
                loss.backward()
                optimizer.step()
                # optimizer_logit_scale.step()
        model.to('cpu')
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    # AlignFed utilize lamda decay trick
    def get_transfer_lambda(self, global_epoch):
        decay_rate = math.log(self.args.final_lamda / self.args.lamda) / self.args.global_rounds
        lamda = self.args.lamda * math.exp(decay_rate * global_epoch)
        return lamda

    def set_parameters(self, global_classifier):
        self.model.head.load_state_dict(global_classifier.state_dict())

    # def test_metrics(self):
    #     testloader = self.load_test_data()
    #     model = load_item(self.role, 'model', self.save_folder_name)
    #     global_protos = load_item('Server', 'global_protos', self.save_folder_name)
    #     model.eval()
    #
    #     test_acc = 0
    #     test_num = 0
    #
    #     if global_protos is not None:
    #         with torch.no_grad():
    #             for x, y in testloader:
    #                 if type(x) == type([]):
    #                     x[0] = x[0].to(self.device)
    #                 else:
    #                     x = x.to(self.device)
    #                 y = y.to(self.device)
    #                 rep = model.base(x)
    #
    #                 output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
    #                 for i, r in enumerate(rep):
    #                     for j, pro in global_protos.items():
    #                         if type(pro) != type([]):
    #                             output[i, j] = self.loss_mse(r, pro)
    #
    #                 test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()
    #                 test_num += y.shape[0]
    #
    #         return test_acc, test_num, 0
    #     else:
    #         return 0, 1e-5, 0

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
        global_protos = self.global_prototype
        proto_dict = matrix_to_dict_proto(self.global_prototype)
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

                # Use version=5 struct_loss: class-level alignment
                if proto_dict is not None:
                    classes_in_batch = torch.unique(y)
                    n_cls = classes_in_batch.size(0)

                    proto_local = torch.zeros(n_cls, rep.size(1), device=self.device)
                    for idx_c, c in enumerate(classes_in_batch):
                        mask = (y == c)
                        proto_local[idx_c] = rep[mask].mean(0)

                    selected_global = []
                    valid_idx = []
                    for idx_c, c in enumerate(classes_in_batch):
                        if (proto_dict is not None and
                            type(proto_dict[c.item()]) != type([])):
                            selected_global.append(proto_dict[c.item()].to(self.device))
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
                        if type(proto_dict[y_c]) != type([]):
                            proto_new[idx, :] = proto_dict[y_c].data

                    cka_loss_val = struct_loss(rep, proto_new, mode=self.args.struct_loss_type)
                    loss += cka_loss_val * self.gamma

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        model.to('cpu')
        return losses, train_num


class Contrastive_Loss_Center(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=1, device='cuda:0'):
        super(Contrastive_Loss_Center, self).__init__()
        self.temperature = temperature
        self.device = device

    def L1_norm(self, feature1, feature2):
        feature1_temp = feature1.view(feature1.shape[0], 1, feature1.shape[1])
        feature2_temp = feature2.view(1, feature2.shape[0], feature2.shape[1])
        return -torch.sum(torch.abs(feature1_temp - feature2_temp), dim=(2))

    def L2_norm(self, feature1, feature2):
        feature1_temp = feature1.view(feature1.shape[0], 1, feature1.shape[1])
        feature2_temp = feature2.view(1, feature2.shape[0], feature2.shape[1])
        return -torch.sum((feature1_temp - feature2_temp) ** 2, dim=(2))

    def CosineSim(self, feature1, feature2):
        return torch.matmul(feature1, feature2.T)

    def distance(self, feature1, feature2, distance_type):
        if distance_type == 'L1':
            return self.L1_norm(feature1, feature2)
        elif distance_type == 'L2':
            return self.L2_norm(feature1, feature2)
        elif distance_type == 'cos':
            return self.CosineSim(feature1, feature2)
        else:
            raise ('Only support distance type: L1 | L2 | cos')

    def forward(self, features, feature_center, labels=None, negative_exp=1.0, distance_type = 'cos'):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, ...].
            feature_center: feature center client for per class [num_cls, ...]
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        device = self.device
        eps = 1e-6
        cls_num = feature_center.shape[0]
        # generate mask
        labels = labels.contiguous().view(-1, 1)

        # print('labels:', labels)
        cls_list = torch.arange(cls_num).to(device)
        # mask [bs, cls_num]
        mask = torch.eq(labels, cls_list.T).float().to(device)

        # compute logits
        # [bs, cls_num]

        features_norm = F.normalize(features, dim=1)
        #
        feature_center_norm = F.normalize(feature_center, dim=1)

        center_dot_feature = torch.div(
            self.distance(features_norm, feature_center_norm, distance_type),
            self.temperature)

        # compute log_prob: [bs, cls_num]
        exp_logits = torch.exp(center_dot_feature)
        log_prob = center_dot_feature - torch.log(torch.pow(exp_logits.sum(0, keepdim=True), negative_exp))

        mean_log_prob_pos = (mask * log_prob).sum(0) / (mask.sum(0) + eps)

        # loss
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss