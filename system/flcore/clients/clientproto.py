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


class clientProto(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda

        self.model = load_item(self.role, 'model', self.save_folder_name)

        self.logit_scale = nn.Parameter(torch.tensor(4.6052))


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
                    loss += self.loss_mse(proto_new, rep) * self.lamda
                    # image_features = rep / rep.norm(dim=-1, keepdim=True)
                    # text_features = proto_new / proto_new.norm(dim=-1, keepdim=True)
                    # logit_scale = self.logit_scale.exp()
                    # logits_per_image = logit_scale * image_features @ text_features.t()
                    # loss += F.cross_entropy(logits_per_image, y) * self.lamda

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

                optimizer.zero_grad()
                # optimizer_logit_scale.zero_grad()
                loss.backward()
                optimizer.step()
                # optimizer_logit_scale.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        save_item(agg_func(protos), self.role, 'protos', self.save_folder_name)
        save_item(model, self.role, 'model', self.save_folder_name)
        model.to('cpu')

        # print(f'client: {self.id}. logit_scale: {self.logit_scale.item()}')


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

                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[i, :] = global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

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