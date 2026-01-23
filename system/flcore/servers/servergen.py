import copy
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clientgen import clientGen
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
import os


class FedGen(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientGen)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        # 通信开销和计算开销统计
        self.comm_costs = []  # 每轮的上行链路通信开销（MB）
        self.downlink_comm_costs = []  # 每轮的下行链路通信开销（MB）
        self.client_comp_costs = []  # 每轮的客户端计算开销（秒）
        self.server_comp_costs = []  # 每轮的服务器计算开销（秒）

        generative_model = Generative(
                                args.noise_dim, 
                                args.num_classes, 
                                args.hidden_dim, 
                                args.feature_dim, 
                                self.device
                            ).to(self.device)
        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            save_item(generative_model, self.role, 'generative_model', self.save_folder_name)
        self.loss = nn.CrossEntropyLoss()
        self.generator_learning_rate = args.generator_learning_rate
        
        self.qualified_labels = []
        for client in self.clients:
            for yy in range(self.num_classes):
                self.qualified_labels.extend([yy for _ in range(int(client.sample_per_class[yy].item()))])
        for client in self.clients:
            client.qualified_labels = self.qualified_labels

        self.server_epochs = args.server_epochs

        head = load_item(self.clients[0].role, 'model', self.clients[0].save_folder_name).head
        save_item(head, self.role, 'head', self.save_folder_name)

        # set logger
        if 'main.py' in self.caller_script:
            logger_path = f'../logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_nd{args.noise_dim}_glr{args.generator_learning_rate}_hd{args.hidden_dim}_se{args.server_epochs}_seed{args.seed}/'
        else:
            logger_path = f'../visualization_logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_nd{args.noise_dim}_glr{args.generator_learning_rate}_hd{args.hidden_dim}_se{args.server_epochs}_test({args.visualization_mode}_{args.visualization_dataset_type}_{args.test_data_mode})_seed{args.seed}/'
        
        self.set_loggers(logger_path)

        self.model_save_path = f'../save/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_nd{args.noise_dim}_glr{args.generator_learning_rate}_hd{args.hidden_dim}_se{args.server_epochs}_seed{args.seed}/'

        if 'main.py' in self.caller_script:
            self.final_log_path = f'../logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/summary.txt'
        else:
            self.final_log_path = f'../visualization_logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/test({args.visualization_mode}_{args.visualization_dataset_type}_{args.test_data_mode})/summary.txt'

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_parameters()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.logger.info(f"\n-------------Round number: {i}-------------")
                self.logger.info("\nEvaluate heterogeneous models")
                self.current_epoch = i
                self.evaluate()
                if self.best_epoch == i:
                    if self.args.save_model != 0:
                        self.save_model()

            for client in self.selected_clients:
                client.train()
            
            if self.args.compute_overhead:

                # 统计服务器计算开销
                server_start_time = time.time()

                self.receive_ids()
                self.train_generator()
                self.aggregate_parameters()

                server_comp_time = time.time() - server_start_time
                self.server_comp_costs.append(server_comp_time)

                # 统计上行链路通信开销（MB）
                comm_cost = self.calculate_communication_cost()
                self.comm_costs.append(comm_cost)

                # 统计下行链路通信开销（MB）
                downlink_comm_cost = self.calculate_downlink_communication_cost()
                self.downlink_comm_costs.append(downlink_comm_cost)
            else:
                self.receive_ids()
                self.train_generator()
                self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        hyperparameters = {
            'noise_dim': self.args.noise_dim,
            'generator_learning_rate': self.args.generator_learning_rate,
            'hidden_dim': self.args.hidden_dim,
            'server_epochs': self.args.server_epochs,
            'seed': self.args.seed
        }
        results = {
            'Best accuracy': self.best_acc,
            'Best epoch': self.best_epoch,
        }

        if self.args.compute_overhead:
            # 计算客户端平均计算开时间
            for client in self.clients:
                self.client_comp_costs.append(client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'])
            results['Average uplink communication cost per round'] = sum(self.comm_costs[1:])/len(self.comm_costs[1:])
            results['Average downlink communication cost per round'] = sum(self.downlink_comm_costs[1:])/len(self.downlink_comm_costs[1:])
            results['Average client computation cost per round'] = sum(self.client_comp_costs)/len(self.client_comp_costs)
            results['Average server computation cost per round'] = sum(self.server_comp_costs[1:])/len(self.server_comp_costs[1:])

        self.log_experiment_results(self.final_log_path, hyperparameters, results)

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()

    def calculate_downlink_communication_cost(self):
        """计算下行链路通信开销（MB）- FedGen发送生成器模型参数"""
        total_bytes = 0

        # 计算生成器模型参数大小
        try:
            generative_model = load_item(self.role, 'generative_model', self.save_folder_name)
            for param in generative_model.parameters():
                param_bytes = param.nelement() * 4  # float32计算
                total_bytes += param_bytes
        except Exception as e:
            print(f"Error calculating downlink communication cost: {e}")

        # 转换为MB
        total_mb = total_bytes / (1024 * 1024)

        return total_mb

    def calculate_communication_cost(self):
        """计算上行链路通信开销（MB）"""
        total_bytes = 0

        # 计算每个上传客户端的模型参数大小
        for cid in self.uploaded_ids:
            client = self.clients[cid]
            client_model = load_item(client.role, 'model', client.save_folder_name).head

            # 计算模型参数的总字节数
            for param in client_model.parameters():
                # 每个参数的字节数 = 元素数量 * 每个元素的字节数（float32 = 4字节）
                param_bytes = param.nelement() * 4  # 默认使用float32计算
                total_bytes += param_bytes

        # 转换为MB（1 MB = 1024 * 1024 字节）
        total_mb = total_bytes / (1024 * 1024)

        return total_mb

    def save_model(self):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # save client models
        for client in self.clients:
            client.save_model(save_dir=self.model_save_path)


    def receive_ids(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            head = load_item(client.role, 'model', client.save_folder_name).head
            self.uploaded_models.append(head)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def train_generator(self):
        generative_model = load_item(self.role, 'generative_model', self.save_folder_name)
        generative_optimizer = torch.optim.Adam(
            params=generative_model.parameters(),
            lr=self.generator_learning_rate, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        generative_model.train()

        for _ in range(self.server_epochs):
            labels = np.random.choice(self.qualified_labels, self.batch_size)
            labels = torch.LongTensor(labels).to(self.device)
            z = generative_model(labels)

            logits = 0
            for w, model in zip(self.uploaded_weights, self.uploaded_models):
                model.eval()
                logits += model(z) * w

            generative_optimizer.zero_grad()
            loss = self.loss(logits, labels)
            loss.backward()
            generative_optimizer.step()

    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)

        client = self.clients[self.uploaded_ids[0]]
        head = load_item(client.role, 'model', client.save_folder_name).head
        for param in head.parameters():
            param.data.zero_()
            
        for w, cid in zip(self.uploaded_weights, self.uploaded_ids):
            client = self.clients[cid]
            client_model = load_item(client.role, 'model', client.save_folder_name).head
            for server_param, client_param in zip(head.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * w

        save_item(head, self.role, 'head', self.save_folder_name)

# based on official code https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/trainmodel/generator.py
class Generative(nn.Module):
    def __init__(self, noise_dim, num_classes, hidden_dim, feature_dim, device) -> None:
        super().__init__()

        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.device = device

        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim + num_classes, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU()
        )

        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, labels):
        batch_size = labels.shape[0]
        eps = torch.rand((batch_size, self.noise_dim), device=self.device) # sampling from Gaussian

        y_input = F.one_hot(labels, self.num_classes)
        z = torch.cat((eps, y_input), dim=1)

        z = self.fc1(z)
        z = self.fc(z)

        return z