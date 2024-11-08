import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict
import clip
from transformers import BertModel, BertTokenizer
from flcore.trainmodel.clip_base import CustomCLIP_client


class clientTSPv3(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = args
        # torch.manual_seed(0)

        # initialize local vision model
        self.model = load_item(self.role, 'model', self.save_folder_name)

        if args.server_model == 'clip':
            clip_model, _ = clip.load('ViT-B/32', device=torch.device("cpu"))
            self.logit_scale = clip_model.logit_scale
            self.dtype = clip_model.dtype
        elif args.server_model == 'bert':
            bert_model_name = "bert-base-uncased"  # 或其他BERT模型名称
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
            self.dtype = torch.float32

        self.model.to('cpu')
        self.lamda = args.lamda
        self.trainloader = self.load_train_data()
        self.testloaderfull = self.load_test_data()

        # used for aligning the vision proto
        self.global_vision_proto = None
        self.local_vision_proto = None
        self.loss_mse = nn.MSELoss()
        self.global_text_proto = None

        self.logger = self.args.logger
        self.tensorboardLogger = self.args.tensorboardLogger
        self.epoch = 0

    def train(self):
        self.model.to(self.device)
        optimizer_visual_model = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.model.train()
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        total_loss1 = 0
        total_loss2 = 0
        total_loss3 = 0
        batch_num = 0

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(self.trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # cross entropy loss
                rep = self.model.base(x)
                classification_logit = self.model.head(rep)
                loss1 = self.loss(classification_logit, y)

                # align with the global text prototype
                image_features = rep.type(self.dtype)
                # print(f'type:{self.dtype}')
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = self.global_text_proto
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logit_scale = self.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                loss2 = F.cross_entropy(logits_per_image, y)


                if self.global_vision_proto is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_vision_proto[y_c]) != type([]):
                            proto_new[i, :] = self.global_vision_proto[y_c].data
                    loss3 = self.loss_mse(proto_new, rep)
                else:
                    loss3 = 0


                loss = loss1 + self.lamda * loss2 + self.args.vision_proto * loss3

                total_loss1 += loss1.item()
                total_loss2 += loss2.item()
                total_loss3 += loss3.item() if type(loss3) != type(0) else 0
                batch_num += 1

                optimizer_visual_model.zero_grad()
                loss.backward()
                optimizer_visual_model.step()


        self.local_vision_proto = self.collect_protos()

        avg_loss1 = total_loss1 / batch_num
        avg_loss2 = total_loss2 / batch_num
        avg_loss3 = total_loss3 / batch_num
        print(f'client {self.id} train loss1:{avg_loss1}, loss2:{avg_loss2}, loss3:{avg_loss3}')
        self.logger.info(f'client {self.id} train loss1:{avg_loss1}, loss2:{avg_loss2}, loss3:{avg_loss3}')
        if self.id in [i for i in range(10)]:
            client_train_loss_info = {
                'train_loss1': avg_loss1,
                'train_loss2': avg_loss2,
                'train_loss3': avg_loss3
            }
            self.tensorboardLogger.add_scalars_dict(prefix=f'train/client {self.id}', dic=client_train_loss_info, rnd=self.epoch)

        self.model.to('cpu')
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def collect_protos(self):
        trainloader = self.load_train_data()
        model = self.model
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

        return agg_func(protos)

    def test_metrics(self):
        # testloaderfull = self.load_test_data()
        # model = load_item(self.role, 'model', self.save_folder_name).visual_model
        # model.to(self.device)
        self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        losses = 0

        with torch.no_grad():
            for x, y in self.testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
                loss = self.loss(output, y)
                losses += loss.item() * y.shape[0]

        self.model.to('cpu')
        return test_acc, test_num, losses

    def train_metrics(self):
        # trainloader = self.load_train_data()
        # model = load_item(self.role, 'model', self.save_folder_name).visual_model
        # model.to(self.device)
        self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        train_acc = 0
        with torch.no_grad():
            for x, y in self.trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        self.model.to('cpu')
        return losses, train_num, train_acc

    def set_parameters(self, global_classifier, global_vision_proto, global_text_proto):
        if global_classifier is not None:
            self.model.head.load_state_dict(global_classifier.state_dict())
        if global_vision_proto is not None:
            self.global_vision_proto = copy.deepcopy(global_vision_proto)
        if global_text_proto is not None:
            self.global_text_proto = copy.deepcopy(global_text_proto)

    def set_logit_scale(self, global_logit_scale):
        self.logit_scale = copy.deepcopy(global_logit_scale)

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