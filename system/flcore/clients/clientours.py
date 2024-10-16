import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict
import clip
from flcore.trainmodel.clip_base import CustomCLIP_client


class clientOurs(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = args
        # torch.manual_seed(0)

        # initialize CLIP models
        visual_model = load_item(self.role, 'model', self.save_folder_name)
        clip_model, _ = clip.load('ViT-B/32', device=torch.device("cpu"))
        clip_model.to(self.device)
        self.model = CustomCLIP_client(self.args.classes, clip_model, visual_model, self.args.len_prompt, args.CSC).to(self.device)
        for name, param in self.model.named_parameters():
            if 'prompt_learner' in name or 'visual_model' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        # save_item(model, self.role, 'model', self.save_folder_name)
        self.model.to('cpu')
        self.lamda = args.lamda
        self.trainloader = self.load_train_data()
        self.testloaderfull = self.load_test_data()
        self.prompt_lr = args.prompt_lr

        # used for aligning the vision prompt
        if self.args.vision_proto > 1e-6:
            self.global_vision_prompt = None
            self.local_vision_prompt = None
            self.loss_mse = nn.MSELoss()

    def train(self):
        # trainloader = self.load_train_data()
        # model = load_item(self.role, 'model', self.save_folder_name)
        self.model.to(self.device)
        visual_model = self.model.visual_model
        prompts = self.model.prompt_learner
        optimizer_visual_model = torch.optim.SGD(visual_model.parameters(), lr=self.learning_rate)
        if self.args.len_prompt > 0 and self.args.update_prompt: optimizer_prompts = torch.optim.SGD(prompts.parameters(), lr=self.prompt_lr)
        # model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        protos = defaultdict(list)
        if self.args.alter == 0 or self.args.len_prompt == 0:
            total_loss1 = 0
            total_loss2 = 0
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

                    classification_logit = visual_model(x)
                    loss1 = self.loss(classification_logit, y)

                    clip_logits, rep = self.model(x)
                    loss2 = F.cross_entropy(clip_logits, y)

                    if abs(self.args.vision_proto) > 1e-6:
                        if self.global_vision_prompt is not None:
                            proto_new = copy.deepcopy(rep.detach())
                            for i, yy in enumerate(y):
                                y_c = yy.item()
                                if type(self.global_vision_prompt[y_c]) != type([]):
                                    proto_new[i, :] = self.global_vision_prompt[y_c].data
                            loss1 += self.loss_mse(proto_new, rep) * self.args.vision_proto

                        for i, yy in enumerate(y):
                            y_c = yy.item()
                            protos[y_c].append(rep[i, :].detach().data)

                    loss = loss1 + self.lamda * loss2

                    total_loss1 += loss1.item()
                    total_loss2 += loss2.item()
                    batch_num += 1

                    if self.args.len_prompt > 0 and self.args.update_prompt: optimizer_prompts.zero_grad()
                    optimizer_visual_model.zero_grad()
                    loss.backward()
                    if self.args.len_prompt > 0 and self.args.update_prompt: optimizer_prompts.step()
                    optimizer_visual_model.step()
            avg_loss1 = total_loss1 / batch_num
            avg_loss2 = total_loss2 / batch_num
            print(f'client {self.id} train loss1:{avg_loss1}, loss2:{avg_loss2}')

        else:
            for step in range(max_local_epochs - self.args.prompt_epoch):
                for i, (x, y) in enumerate(self.trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))

                    classification_logit = visual_model(x)
                    loss1 = self.loss(classification_logit, y)

                    clip_logits, rep = self.model(x)
                    loss2 = F.cross_entropy(clip_logits, y)

                    if abs(self.args.vision_proto) > 1e-6:
                        if self.global_vision_prompt is not None:
                            proto_new = copy.deepcopy(rep.detach())
                            for i, yy in enumerate(y):
                                y_c = yy.item()
                                if type(self.global_vision_prompt[y_c]) != type([]):
                                    proto_new[i, :] = self.global_vision_prompt[y_c].data
                            loss1 += self.loss_mse(proto_new, rep) * self.args.vision_proto

                        for i, yy in enumerate(y):
                            y_c = yy.item()
                            protos[y_c].append(rep[i, :].detach().data)

                    loss = loss1 + self.lamda * loss2

                    optimizer_visual_model.zero_grad()
                    loss.backward()
                    optimizer_visual_model.step()

            for step in range(self.args.prompt_epoch):
                for i, (x, y) in enumerate(self.trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))

                    clip_logits, _ = self.model(x)
                    loss = F.cross_entropy(clip_logits, y)

                    if self.args.update_prompt:optimizer_prompts.zero_grad()
                    loss.backward()
                    if self.args.update_prompt: optimizer_prompts.step()

        if abs(self.args.vision_proto) > 1e-6: self.local_vision_prompt = agg_func(protos)
        # save_item(visual_model, self.role, 'visual_model', self.save_folder_name)
        # save_item(prompts, self.role, 'prompts', self.save_folder_name)
        # save_item(model, self.role, 'model', self.save_folder_name)
        self.model.to('cpu')
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

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
                output = self.model.visual_model(x)

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
                output = self.model.visual_model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        self.model.to('cpu')
        return losses, train_num, train_acc

    def set_parameters(self, global_prompts, global_classifier, global_vision_prompt):
        if global_prompts is not None:
            self.model.prompt_learner.load_state_dict(global_prompts.state_dict())
        if global_classifier is not None:
            self.model.visual_model.head.load_state_dict(global_classifier.state_dict())
        if global_vision_prompt is not None:
            self.global_vision_prompt = copy.deepcopy(global_vision_prompt)

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