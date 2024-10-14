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

    def train(self):
        # trainloader = self.load_train_data()
        # model = load_item(self.role, 'model', self.save_folder_name)
        self.model.to(self.device)
        visual_model = self.model.visual_model
        prompts = self.model.prompt_learner
        optimizer_visual_model = torch.optim.SGD(visual_model.parameters(), lr=self.learning_rate)
        if self.args.len_prompt > 0: optimizer_prompts = torch.optim.SGD(prompts.parameters(), lr=self.prompt_lr)
        # model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        if self.args.alter == 0 or self.args.len_prompt == 0:
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

                    clip_logits = self.model(x)
                    loss2 = F.cross_entropy(clip_logits, y)

                    loss = loss1 + self.lamda * loss2

                    if self.args.len_prompt > 0: optimizer_prompts.zero_grad()
                    optimizer_visual_model.zero_grad()
                    loss.backward()
                    if self.args.len_prompt > 0: optimizer_prompts.step()
                    optimizer_visual_model.step()
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

                    clip_logits = self.model(x)
                    loss2 = F.cross_entropy(clip_logits, y)

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

                    clip_logits = self.model(x)
                    loss = F.cross_entropy(clip_logits, y)

                    optimizer_prompts.zero_grad()
                    loss.backward()
                    optimizer_prompts.step()


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

    def set_parameters(self, global_prompts, global_classifier):
        if global_prompts is not None:
            self.model.prompt_learner.load_state_dict(global_prompts.state_dict())
        if global_classifier is not None:
            self.model.visual_model.head.load_state_dict(global_classifier.state_dict())