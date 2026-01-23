import copy
import random
import time
from flcore.clients.clientmrl import clientMRL
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from flcore.trainmodel.models import BaseHeadSplit


class FedMRL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            global_model = BaseHeadSplit(args, 0, args.sub_feature_dim).to(args.device)
            save_item(global_model, self.role, 'global_model', self.save_folder_name)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientMRL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        # 通信开销和计算开销统计
        self.comm_costs = []  # 每轮的上行链路通信开销（MB）
        self.downlink_comm_costs = []  # 每轮的下行链路通信开销（MB）
        self.client_comp_costs = []  # 每轮的客户端计算开销（秒）
        self.server_comp_costs = []  # 每轮的服务器计算开销（秒）

        # set logger
        if 'main.py' in self.caller_script:
            logger_path = f'../logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_sub_feature{args.sub_feature_dim}_seed{args.seed}/'
        else:
            logger_path = f'../visualization_logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_sub_feature{args.sub_feature_dim}_test({args.visualization_mode}_{args.visualization_dataset_type}_{args.test_data_mode})_seed{args.seed}/'
        
        self.set_loggers(logger_path)

        self.model_save_path = f'../save/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_sub_feature{args.sub_feature_dim}_seed{args.seed}/'

        if 'main.py' in self.caller_script:
            self.final_log_path = f'../logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/summary.txt'
        else:
            self.final_log_path = f'../visualization_logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/test({args.visualization_mode}_{args.visualization_dataset_type}_{args.test_data_mode})/summary.txt'


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.current_epoch = i
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            if self.args.compute_overhead:
                # 统计服务器计算开销
                server_start_time = time.time()

                self.receive_ids()
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
                self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        hyperparameters = {
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

        self.save_results()
        
        
    def calculate_downlink_communication_cost(self):
        """计算下行链路通信开销（MB）- FedMRL 发送全局模型参数"""
        total_bytes = 0

        # 计算发送给客户端的全局模型参数大小
        try:
            global_model = load_item(self.role, 'global_model', self.save_folder_name)
            for param in global_model.parameters():
                param_bytes = param.nelement() * 4  # float32计算
                total_bytes += param_bytes
        except Exception as e:
            print(f"Error calculating downlink communication cost: {e}")

        # 转换为MB
        total_mb = total_bytes / (1024 * 1024)

        return total_mb

    def calculate_communication_cost(self):
        """计算上行链路通信开销（MB）- FedMRL 接收完整模型参数"""
        total_bytes = 0

        # 计算每个上传客户端的模型参数大小
        for cid in self.uploaded_ids:
            client = self.clients[cid]
            try:
                client_model = load_item(client.role, 'global_model', client.save_folder_name)

                # 计算模型参数的总字节数
                for param in client_model.parameters():
                    # 每个参数的字节数 = 元素数量 * 每个元素的字节数（float32 = 4字节）
                    param_bytes = param.nelement() * 4  # 默认使用float32计算
                    total_bytes += param_bytes
            except Exception as e:
                print(f"Error calculating communication cost for client {cid}: {e}")
                continue

        # 转换为MB（1 MB = 1024 * 1024 字节）
        total_mb = total_bytes / (1024 * 1024)

        return total_mb

    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)

        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        for param in global_model.parameters():
            param.data.zero_()
            
        for cid in self.uploaded_ids:
            client = self.clients[cid]
            client_model = load_item(client.role, 'global_model', client.save_folder_name)
            for server_param, client_param in zip(global_model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * 1/len(self.uploaded_ids)

        save_item(global_model, self.role, 'global_model', self.save_folder_name)
