import time
import torch
import torch.nn as nn
from flcore.clients.clientgh import clientGH
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from torch.utils.data import DataLoader
import os


class FedGH(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientGH)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.CEloss = nn.CrossEntropyLoss()
        self.server_learning_rate = args.server_learning_rate
        self.server_epochs = args.server_epochs

        # 通信开销和计算开销统计
        self.comm_costs = []  # 每轮的上行链路通信开销（MB）
        self.downlink_comm_costs = []  # 每轮的下行链路通信开销（MB）
        self.client_comp_costs = []  # 每轮的客户端计算开销（秒）
        self.server_comp_costs = []  # 每轮的服务器计算开销（秒）

        head = load_item(self.clients[0].role, 'model', self.clients[0].save_folder_name).head
        save_item(head, 'Server', 'head', self.save_folder_name)

        # set logger
        if 'main.py' in self.caller_script:
            logger_path = f'../logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_slr{args.server_learning_rate}_seed{args.seed}/'
        else:
            logger_path = f'../visualization_logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_slr{args.server_learning_rate}_test({args.visualization_mode}_{args.visualization_dataset_type}_{args.test_data_mode})_seed{args.seed}/'
        
        self.set_loggers(logger_path)

        self.model_save_path = f'../save/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_slr{args.server_learning_rate}_seed{args.seed}/'

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
                client.collect_protos()

            if self.args.compute_overhead:

                # 统计服务器计算开销
                server_start_time = time.time()

                self.receive_protos()
                self.train_head()

                server_comp_time = time.time() - server_start_time
                self.server_comp_costs.append(server_comp_time)

                # 统计上行链路通信开销（MB）
                comm_cost = self.calculate_communication_cost()
                self.comm_costs.append(comm_cost)

                # 统计下行链路通信开销（MB）
                downlink_comm_cost = self.calculate_downlink_communication_cost()
                self.downlink_comm_costs.append(downlink_comm_cost)
            else:
                self.receive_protos()
                self.train_head()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        hyperparameters = {
            'server_learning_rate': self.args.server_learning_rate,
            'seed': self.args.seed
        }
        results = {
            'Best accuracy': self.best_acc,
            'Best epoch': self.best_epoch,
        }

        if self.args.compute_overhead:
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
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()

    def calculate_downlink_communication_cost(self):
        """计算下行链路通信开销（MB）- FedGH 发送头部模型参数"""
        total_bytes = 0

        # 计算发送给客户端的头部模型参数大小
        try:
            head = load_item('Server', 'head', self.save_folder_name)
            for param in head.parameters():
                param_bytes = param.nelement() * 4  # float32计算
                total_bytes += param_bytes
        except Exception as e:
            print(f"Error calculating downlink communication cost: {e}")

        # 转换为MB
        total_mb = total_bytes / (1024 * 1024)

        return total_mb

    def calculate_communication_cost(self):
        """计算上行链路通信开销（MB）- FedGH 接收原型数据"""
        total_bytes = 0

        # 计算每个上传客户端的原型数据大小
        for client in self.selected_clients:
            try:
                protos = load_item(client.role, 'protos', client.save_folder_name)
                for cc in protos.keys():
                    proto = protos[cc]
                    # 每个原型的字节数 = 元素数量 * 每个元素的字节数（float32 = 4字节）
                    proto_bytes = proto.nelement() * 4  # 默认使用float32计算
                    total_bytes += proto_bytes
            except Exception as e:
                print(f"Error calculating communication cost for client {client.id}: {e}")
                continue

        # 转换为MB（1 MB = 1024 * 1024 字节）
        total_mb = total_bytes / (1024 * 1024)

        return total_mb

    def save_model(self):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # save client models
        for client in self.clients:
            client.save_model(save_dir=self.model_save_path)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            protos = load_item(client.role, 'protos', client.save_folder_name)
            for cc in protos.keys():
                y = torch.tensor(cc, dtype=torch.int64, device=self.device)
                uploaded_protos.append((protos[cc], y))
            
        save_item(uploaded_protos, self.role, 'uploaded_protos', self.save_folder_name)
    
    def train_head(self):
        uploaded_protos = load_item(self.role, 'uploaded_protos', self.save_folder_name)
        proto_loader = DataLoader(uploaded_protos, self.batch_size, drop_last=False, shuffle=True)
        head = load_item('Server', 'head', self.save_folder_name)
        
        opt_h = torch.optim.SGD(head.parameters(), lr=self.server_learning_rate)

        for _ in range(self.server_epochs):
            for p, y in proto_loader:
                out = head(p)
                loss = self.CEloss(out, y)
                opt_h.zero_grad()
                loss.backward()
                opt_h.step()

        save_item(head, 'Server', 'head', self.save_folder_name)