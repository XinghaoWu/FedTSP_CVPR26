import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flcore.clients.clienttgp_cvpr26 import clientTGP_CVPR26
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from collections import defaultdict
from torch.utils.data import DataLoader

import os, copy


class FedTGP_CVPR26(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # save original dataset information for potential switch
        self.original_dataset = args.dataset
        self.switch_dataset = args.switch_dataset
        self.switch_round = args.switch_round
        self.dataset_switched = False
        # 最佳准确率统计
        self.best_acc_before_switch = 0.0
        self.best_epoch_before_switch = 0
        self.best_acc_after_switch = 0.0
        self.best_epoch_after_switch = 0

        # openset training parameters
        self.openset = args.openset
        self.first_stage_clients = args.first_stage_clients
        self.first_stage_rounds = args.first_stage_rounds
        # openset best accuracy tracking
        self.best_acc_first_stage = 0.0
        self.best_epoch_first_stage = 0
        self.best_acc_remaining = 0.0
        self.best_epoch_remaining = 0

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientTGP_CVPR26)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes

        # 通信开销和计算开销统计
        self.comm_costs = []  # 每轮的上行链路通信开销（MB）
        self.downlink_comm_costs = []  # 每轮的下行链路通信开销（MB）
        self.client_comp_costs = []  # 每轮的客户端计算开销（秒）
        self.server_comp_costs = []  # 每轮的服务器计算开销（秒）

        self.server_learning_rate = args.local_learning_rate
        self.batch_size = args.batch_size
        self.server_epochs = args.server_epochs
        self.margin_threthold = args.margin_threthold

        self.feature_dim = args.feature_dim
        self.server_hidden_dim = self.feature_dim

        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            PROTO = Trainable_prototypes(
                self.num_classes,
                self.server_hidden_dim,
                self.feature_dim,
                self.device
            ).to(self.device)
            save_item(PROTO, self.role, 'PROTO', self.save_folder_name)
            print(PROTO)
        self.CEloss = nn.CrossEntropyLoss()
        self.MSEloss = nn.MSELoss()

        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        self.min_gap = None
        self.max_gap = None

        # set logger
        if 'main.py' in self.caller_script:
            logger_path = (f'../logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_lamda{args.lamda}_se{args.server_epochs}_margin{args.margin_threthold}'
                        f'{f"_switch({args.switch_dataset}_{args.switch_round})" if args.switch_round != -1 else ""}'
                        f'{f"_openset({args.first_stage_clients}_{args.first_stage_rounds})" if args.openset == 1 else ""}_seed{args.seed}/')
        else:
            logger_path = (f'../visualization_logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_lamda{args.lamda}_se{args.server_epochs}_margin{args.margin_threthold}'
                        f'{f"_switch({args.switch_dataset}_{args.switch_round})" if args.switch_round != -1 else ""}'
                        f'{f"_openset({args.first_stage_clients}_{args.first_stage_rounds})" if args.openset == 1 else ""}_test({args.visualization_mode}_{args.visualization_dataset_type}_{args.test_data_mode})_seed{args.seed}/')

        self.set_loggers(logger_path)

        self.model_save_path = (f'../save/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_'
                                f'ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_lamda{args.lamda}_se{args.server_epochs}_margin{args.margin_threthold}'
                                f'{f"_switch({args.switch_dataset}_{args.switch_round})" if args.switch_round != -1 else ""}'
                                f'{f"_openset({args.first_stage_clients}_{args.first_stage_rounds})" if args.openset == 1 else ""}_seed{args.seed}')

        self.plot_path = (f'../plot/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_'
                          f'ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_lamda{args.lamda}_se{args.server_epochs}_margin{args.margin_threthold}'
                          f'{f"_switch({args.switch_dataset}_{args.switch_round})" if args.switch_round != -1 else ""}'
                          f'{f"_openset({args.first_stage_clients}_{args.first_stage_rounds})" if args.openset == 1 else ""}_seed{args.seed}')
        if 'main.py' in self.caller_script:
            self.final_log_path = (f'../logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/'
                               f'summary{f"_switch({args.switch_dataset}_{args.switch_round})" if args.switch_round != -1 else ""}'
                               f'{f"_openset({args.first_stage_clients}_{args.first_stage_rounds})" if args.openset == 1 else ""}.txt')
        else:
            self.final_log_path = (f'../visualization_logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/'
                               f'test({args.visualization_mode}_{args.visualization_dataset_type}_{args.test_data_mode})/summary{f"_switch({args.switch_dataset}_{args.switch_round})" if args.switch_round != -1 else ""}'
                               f'{f"_openset({args.first_stage_clients}_{args.first_stage_rounds})" if args.openset == 1 else ""}.txt')

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            # Switch dataset if it's the specified round and hasn't been switched yet
            if i == self.switch_round and not self.dataset_switched and self.switch_round != -1 and self.switch_dataset:
                print(f"\n-------------Switching dataset to {self.switch_dataset} at round {i}-------------")
                self.logger.info(f"\n-------------Switching dataset to {self.switch_dataset} at round {i}-------------")
                self.switch_client_data()
                self.dataset_switched = True

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

                self.receive_protos()
                self.update_Gen()

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
                self.update_Gen()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        hyperparameters = {
            'lamda': self.args.lamda,
            'server_epochs': self.args.server_epochs,
            'margin_threshold': self.args.margin_threthold,
            'seed': self.args.seed
        }
        results = {
            'Best accuracy': self.best_acc,
            'Best epoch': self.best_epoch,
        }
        # 添加切换前后最佳准确率统计
        if self.switch_round != -1:
            results['Best accuracy before switch'] = self.best_acc_before_switch
            results['Best epoch before switch'] = self.best_epoch_before_switch
            if self.dataset_switched:
                results['Best accuracy after switch'] = self.best_acc_after_switch
                results['Best epoch after switch'] = self.best_epoch_after_switch

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
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        

    def switch_client_data(self):
        """Switch client training and test data to the new dataset distribution"""
        # Update dataset information for data loading
        self.args.dataset = self.switch_dataset
        print(f"Server dataset switched to {self.switch_dataset}")
        self.logger.info(f"Server dataset switched to {self.switch_dataset}")

        # 由于客户端在每次训练/测试时都会直接调用 load_train_data() 和 load_test_data()，
        # 而这些方法使用 self.args.dataset，所以我们只需要更新客户端端的 args.dataset 即可
        for client in self.clients:
            client.args.dataset = self.args.dataset
        print("Client data will be loaded from new dataset on next training/testing round")
        self.logger.info("Client data will be loaded from new dataset on next training/testing round")

    def select_clients(self):
        """Override base class select_clients to support openset training"""
        if self.openset == 1 and self.current_epoch < self.first_stage_rounds:
            # First stage: only select from first_stage_clients
            available_clients = self.clients[:self.first_stage_clients]
            if self.random_join_ratio:
                num_join = np.random.choice(range(1, len(available_clients) + 1), 1, replace=False)[0]
            else:
                num_join = int(self.join_ratio * len(available_clients))
                num_join = max(1, num_join)
            selected_clients = list(np.random.choice(available_clients, num_join, replace=False))
            self.current_num_join_clients = len(selected_clients)  # Update for receive_ids
            return selected_clients
        else:
            # Second stage or normal mode: use base class implementation
            return super().select_clients()

    # 重写evaluate方法以统计切换前后最佳准确率
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        # 如果启用了openset模式,分别计算前first_stage_clients和后续clients的准确率
        if self.openset == 1:
            # First stage clients (0 to first_stage_clients-1)
            first_stage_acc = sum(stats[2][:self.first_stage_clients]) * 1.0 / sum(stats[1][:self.first_stage_clients])
            # Remaining clients (from first_stage_clients to end)
            remaining_acc = sum(stats[2][self.first_stage_clients:]) * 1.0 / sum(stats[1][self.first_stage_clients:]) if len(stats[2]) > self.first_stage_clients else 0.0

            # Update best accuracy for first stage clients
            if first_stage_acc >= self.best_acc_first_stage:
                self.best_acc_first_stage = first_stage_acc
                self.best_epoch_first_stage = self.current_epoch

            # Update best accuracy for remaining clients
            if remaining_acc >= self.best_acc_remaining:
                self.best_acc_remaining = remaining_acc
                self.best_epoch_remaining = self.current_epoch

            print(f"First stage clients (0-{self.first_stage_clients-1}) Test Acc: {first_stage_acc:.4f}, Best: {self.best_acc_first_stage:.4f} (Epoch {self.best_epoch_first_stage})")
            print(f"Remaining clients ({self.first_stage_clients}-{self.num_clients-1}) Test Acc: {remaining_acc:.4f}, Best: {self.best_acc_remaining:.4f} (Epoch {self.best_epoch_remaining})")
            self.logger.info(f"First stage clients (0-{self.first_stage_clients-1}) Test Acc: {first_stage_acc:.4f}, Best: {self.best_acc_first_stage:.4f} (Epoch {self.best_epoch_first_stage})")
            self.logger.info(f"Remaining clients ({self.first_stage_clients}-{self.num_clients-1}) Test Acc: {remaining_acc:.4f}, Best: {self.best_acc_remaining:.4f} (Epoch {self.best_epoch_remaining})")

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        # 统计全过程最佳准确率
        if test_acc >= self.best_acc:
            self.best_acc = test_acc
            self.best_epoch = self.current_epoch

        # 统计切换前最佳准确率
        if not self.dataset_switched:
            if test_acc >= self.best_acc_before_switch:
                self.best_acc_before_switch = test_acc
                self.best_epoch_before_switch = self.current_epoch
        # 统计切换后最佳准确率
        else:
            if test_acc >= self.best_acc_after_switch:
                self.best_acc_after_switch = test_acc
                self.best_epoch_after_switch = self.current_epoch

        # if loss == None:
        #     self.rs_train_loss.append(train_loss)
        # else:
        #     loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        print("Best Epoch: {}".format(self.best_epoch))
        print("Best Test Accurancy: {:.4f}".format(self.best_acc))

        # 打印切换前后最佳准确率
        if self.switch_round != -1:
            print("Best Acc before switch: {:.4f} (Epoch: {})".format(self.best_acc_before_switch, self.best_epoch_before_switch))
            if self.dataset_switched:
                print("Best Acc after switch: {:.4f} (Epoch: {})".format(self.best_acc_after_switch, self.best_epoch_after_switch))

        if self.logger is not None:
            self.logger.info("Averaged Train Loss: {:.4f}".format(train_loss))
            self.logger.info("Averaged Test Accurancy: {:.4f}".format(test_acc))
            self.logger.info("Averaged Test AUC: {:.4f}".format(test_auc))
            self.logger.info("Std Test Accurancy: {:.4f}".format(np.std(accs)))
            self.logger.info("Std Test AUC: {:.4f}".format(np.std(aucs)))
            self.logger.info("Best Epoch: {}".format(self.best_epoch))
            self.logger.info("Best Test Accurancy: {:.4f}".format(self.best_acc))
            # 打印切换前后最佳准确率
            if self.switch_round != -1:
                self.logger.info("Best Acc before switch: {:.4f} (Epoch: {})".format(self.best_acc_before_switch, self.best_epoch_before_switch))
                if self.dataset_switched:
                    self.logger.info("Best Acc after switch: {:.4f} (Epoch: {})".format(self.best_acc_after_switch, self.best_epoch_after_switch))
            # 记录所有客户端的测试准确率（按ID排序，数值列表格式）
            client_acc_pairs = list(zip(stats[0], accs))
            client_acc_pairs.sort(key=lambda x: x[0])  # 按客户端ID从小到大排序
            acc_list = [f"{acc:.4f}" for cid, acc in client_acc_pairs]
            self.logger.info("Test Accuracy list: [" + ", ".join(acc_list) + "]")

            train_info = {
                'train_loss': train_loss,
            }
            self.tensorboardLogger.add_scalars_dict(prefix='train', dic=train_info, rnd=self.current_epoch)

            test_info = {
                'test_acc': test_acc,
                'best_acc': self.best_acc
            }

            # 如果启用了openset模式,在tensorboard中记录分组准确率
            if self.openset == 1:
                test_info['first_stage_acc'] = first_stage_acc
                test_info['remaining_acc'] = remaining_acc
                test_info['best_acc_first_stage'] = self.best_acc_first_stage
                test_info['best_acc_remaining'] = self.best_acc_remaining

            self.tensorboardLogger.add_scalars_dict(prefix='test', dic=test_info, rnd=self.current_epoch)

    def calculate_downlink_communication_cost(self):
        """计算下行链路通信开销（MB）- FedTGP 发送可训练原型模型"""
        total_bytes = 0

        # 计算发送给客户端的可训练原型模型参数大小
        try:
            PROTO = load_item('Server', 'global_protos', self.save_folder_name)
            for param in PROTO.parameters():
                param_bytes = param.nelement() * 4  # float32计算
                total_bytes += param_bytes
        except Exception as e:
            print(f"Error calculating downlink communication cost: {e}")

        # 转换为MB
        total_mb = total_bytes / (1024 * 1024)

        return total_mb

    def calculate_communication_cost(self):
        """计算上行链路通信开销（MB）- FedTGP 接收原型数据"""
        total_bytes = 0

        # 计算每个上传客户端的原型数据大小
        for client in self.selected_clients:
            try:
                protos = load_item(client.role, 'protos', client.save_folder_name)
                for k in protos.keys():
                    proto = protos[k]
                    # 每个原型的字节数 = 元素数量 * 每个元素的字节数（float32 = 4字节）
                    proto_bytes = proto.nelement() * 4  # 默认使用float32计算
                    total_bytes += proto_bytes
            except Exception as e:
                print(f"Error calculating communication cost for client {client.id}: {e}")
                continue

        # 转换为MB（1 MB = 1024 * 1024 字节）
        total_mb = total_bytes / (1024 * 1024)

        return total_mb

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        uploaded_protos_per_client = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            protos = load_item(client.role, 'protos', client.save_folder_name)
            for k in protos.keys():
                self.uploaded_protos.append((protos[k], k))
            uploaded_protos_per_client.append(protos)

        # calculate class-wise minimum distance
        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        avg_protos = proto_cluster(uploaded_protos_per_client)
        for k1 in avg_protos.keys():
            for k2 in avg_protos.keys():
                if k1 > k2:
                    dis = torch.norm(avg_protos[k1] - avg_protos[k2], p=2)
                    self.gap[k1] = torch.min(self.gap[k1], dis)
                    self.gap[k2] = torch.min(self.gap[k2], dis)
        self.min_gap = torch.min(self.gap)
        for i in range(len(self.gap)):
            if self.gap[i] > torch.tensor(1e8, device=self.device):
                self.gap[i] = self.min_gap
        self.max_gap = torch.max(self.gap)
        print('class-wise minimum distance', self.gap)
        print('min_gap', self.min_gap)
        print('max_gap', self.max_gap)
            
    def update_Gen(self):
        PROTO = load_item(self.role, 'PROTO', self.save_folder_name)
        Gen_opt = torch.optim.SGD(PROTO.parameters(), lr=self.server_learning_rate)
        PROTO.train()
        for e in range(self.server_epochs):
            proto_loader = DataLoader(self.uploaded_protos, self.batch_size, 
                                      drop_last=False, shuffle=True)
            for proto, y in proto_loader:
                y = torch.Tensor(y).type(torch.int64).to(self.device)

                proto_gen = PROTO(list(range(self.num_classes)))

                features_square = torch.sum(torch.pow(proto, 2), 1, keepdim=True)
                centers_square = torch.sum(torch.pow(proto_gen, 2), 1, keepdim=True)
                features_into_centers = torch.matmul(proto, proto_gen.T)
                dist = features_square - 2 * features_into_centers + centers_square.T
                dist = torch.sqrt(dist)
                
                one_hot = F.one_hot(y, self.num_classes).to(self.device)
                gap2 = min(self.max_gap.item(), self.margin_threthold)
                dist = dist + one_hot * gap2
                loss = self.CEloss(-dist, y)

                Gen_opt.zero_grad()
                loss.backward()
                Gen_opt.step()

        print(f'Server loss: {loss.item()}')
        self.uploaded_protos = []
        save_item(PROTO, self.role, 'PROTO', self.save_folder_name)

        PROTO.eval()
        global_protos = defaultdict(list)
        for class_id in range(self.num_classes):
            global_protos[class_id] = PROTO(torch.tensor(class_id, device=self.device)).detach()
        save_item(global_protos, self.role, 'global_protos', self.save_folder_name)

    def save_model(self):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # save server proto
        PROTO = load_item(self.role, 'PROTO', self.save_folder_name)
        torch.save(PROTO, f'{self.model_save_path}/global_proto.pth')

        # save client models
        for client in self.clients:
            client.save_model(save_dir=self.model_save_path)

    def load_model(self):
        if not os.path.exists(self.model_save_path):
            raise ValueError(f'No model to load: {self.model_save_path}')

        # load server proto with map_location to handle device mismatch
        self.global_proto = torch.load(f'{self.model_save_path}/global_proto.pth', map_location=self.device)

        # load client models
        for c in self.clients:
            c.load_model(save_dir=self.model_save_path)

        print('Loaded checkpoint models successfully')
        self.logger.info('Loaded checkpoint models successfully')

    def load_global_prototype(self):
        if not os.path.exists(self.model_save_path):
            raise ValueError(f'No model to load: {self.model_save_path}')

        # load server proto with map_location to handle device mismatch
        self.global_proto = torch.load(f'{self.model_save_path}/global_proto.pth', map_location=self.device)

        print('Loaded checkpoint global protos successfully')
        self.logger.info('Loaded checkpoint global protos successfully')

    def get_global_protos(self):
        PROTO = self.global_proto
        global_protos = defaultdict(list)
        for class_id in range(self.num_classes):
            global_protos[class_id] = PROTO(torch.tensor(class_id, device=self.device)).detach()
        global_protos = torch.stack([global_protos[class_id] for class_id in range(self.num_classes)])
        return global_protos

    # def visualize_global_prototype_similarity(self):
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #     import matplotlib
    #     matplotlib.rcParams['pdf.fonttype'] = 42
    #
    #     self.load_model()
    #     global_protos = self.get_global_protos()
    #     print(global_protos)
    #
    #     global_protos = global_protos / global_protos.norm(dim=-1, keepdim=True)
    #     if self.args.similarity_mode == 'cosine':
    #         similarity_matrix = global_protos @ global_protos.T
    #     elif self.args.similarity_mode == 'euclidean':
    #         similarity_matrix = torch.cdist(global_protos, global_protos, p=2)
    #     else:
    #         raise ValueError(f'Invalid similarity mode: {self.args.similarity_mode}')
    #
    #     # Convert to numpy for visualization
    #     similarity_matrix_np = similarity_matrix.detach().cpu().numpy()
    #     print(similarity_matrix_np)
    #
    #     # Get class names from self.args.classes
    #     class_names = self.args.classes if hasattr(self.args, 'classes') else [f'Class {i}' for i in
    #                                                                            range(similarity_matrix_np.shape[0])]
    #
    #     # Plot heatmap with values
    #     plt.figure(figsize=(10, 8))
    #     if self.args.similarity_mode == 'cosine':
    #         sns.heatmap(similarity_matrix_np, annot=True, fmt=".2f", cmap='Blues', cbar=True,
    #                     xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1)
    #     else:
    #         sns.heatmap(similarity_matrix_np, annot=True, fmt=".2f", cmap='Blues', cbar=True,
    #                     xticklabels=class_names, yticklabels=class_names)
    #     plt.title('Global Prototype Similarity Heatmap')
    #     plt.xlabel('Prototypes')
    #     plt.ylabel('Prototypes')
    #     plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    #     plt.tight_layout()  # Adjust layout to prevent label cutoff
    #     plt.show()
    #
    #     return similarity_matrix



def proto_cluster(protos_list):
    proto_clusters = defaultdict(list)
    for protos in protos_list:
        for k in protos.keys():
            proto_clusters[k].append(protos[k])

    for k in proto_clusters.keys():
        protos = torch.stack(proto_clusters[k])
        proto_clusters[k] = torch.mean(protos, dim=0).detach()

    return proto_clusters
            

class Trainable_prototypes(nn.Module):
    def __init__(self, num_classes, server_hidden_dim, feature_dim, device):
        super().__init__()

        self.device = device

        self.embedings = nn.Embedding(num_classes, feature_dim)
        layers = [nn.Sequential(
            nn.Linear(feature_dim, server_hidden_dim), 
            nn.ReLU()
        )]
        self.middle = nn.Sequential(*layers)
        self.fc = nn.Linear(server_hidden_dim, feature_dim)

    def forward(self, class_id):
        class_id = torch.tensor(class_id, device=self.device)

        emb = self.embedings(class_id)
        mid = self.middle(emb)
        out = self.fc(mid)

        return out