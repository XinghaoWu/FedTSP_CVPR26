import time
import numpy as np
from flcore.clients.clientalignstruct import clientAlignStruct
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from utils.data_utils import read_client_data
from threading import Thread
from collections import defaultdict
import os, copy
import torch
from tqdm import tqdm


class AlignFedStruct(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAlignStruct)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes

        # set logger
        if 'main.py' in self.caller_script:
            logger_path = f'../logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_struct({args.struct_loss_type})_lamda{args.lamda}_finallamda{args.final_lamda}_gamma{args.gamma}_seed{args.seed}/'
        else:
            logger_path = f'../visualization_logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_struct({args.struct_loss_type})_lamda{args.lamda}_finallamda{args.final_lamda}_gamma{args.gamma}_test({args.visualization_mode}_{args.visualization_dataset_type}_{args.test_data_mode})_seed{args.seed}/'

        self.set_loggers(logger_path)

        self.model_save_path = (f'../save/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_'
                                f'ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_struct({args.struct_loss_type})_lamda{args.lamda}_finallamda{args.final_lamda}_gamma{args.gamma}_seed{args.seed}')

        self.plot_path = (f'../plot/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_'
                          f'ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_struct({args.struct_loss_type})_lamda{args.lamda}_finallamda{args.final_lamda}_gamma{args.gamma}_seed{args.seed}')

        if 'main.py' in self.caller_script:
            self.final_log_path = f'../logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/summary.txt'
        else:
            self.final_log_path = f'../visualization_logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/test({args.visualization_mode}_{args.visualization_dataset_type}_{args.test_data_mode})/summary.txt'

    def train(self):
        print(f'generate global prototype')
        self.global_proto = generate_target(self.num_classes, self.args.feature_dim, 100000, lr=0.1, tau=0.5, device=self.args.device)
        print(f'global prototype generated: {self.global_proto}')
        self.send_global_protos()

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
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

            self.receive_ids()
            self.aggregate_parameters()
            self.send_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        hyperparameters = {
            'lamda': self.args.lamda,
            'gamma': self.args.gamma,
            'final lamda': self.args.final_lamda,
            'struct_loss': self.args.struct_loss_type,
            'seed': self.args.seed
        }
        results = {
            'Best accuracy': self.best_acc,
            'Best epoch': self.best_epoch,
        }
        self.log_experiment_results(self.final_log_path, hyperparameters, results)

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)

        # aggregate global classifier
        client = self.clients[self.uploaded_ids[0]]
        global_classifier = copy.deepcopy(client.model.head)
        for param in global_classifier.parameters():
            param.data.zero_()
        for w, cid in zip(self.uploaded_weights, self.uploaded_ids):
            client = self.clients[cid]
            client_classifier = copy.deepcopy(client.model.head)
            for server_param, client_param in zip(global_classifier.parameters(), client_classifier.parameters()):
                server_param.data += client_param.data.clone() * w
        self.global_classifier = global_classifier

    def send_parameters(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            # send global classifier
            client.set_parameters(self.global_classifier)
            # sync global training round
            client.global_round = self.current_epoch + 1

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def send_global_protos(self):
        for client in self.clients:
            client.global_prototype = copy.deepcopy(self.global_proto)

    def save_model(self):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # save server proto
        try:
            PROTO = self.global_proto
            torch.save(PROTO, f'{self.model_save_path}/global_proto.pth')
        except:
            print('No global protos to save')

        # save client models
        for client in self.clients:
            client.save_model(save_dir=self.model_save_path)

    def load_model(self):
        if not os.path.exists(self.model_save_path):
            raise ValueError(f'No model to load: {self.model_save_path}')

        # load server proto
        self.global_proto = torch.load(f'{self.model_save_path}/global_proto.pth')

        # load client models
        for c in self.clients:
            c.load_model(save_dir=self.model_save_path)

        print('Loaded checkpoint models successfully')
        self.logger.info('Loaded checkpoint models successfully')

    def load_global_prototype(self):
        if not os.path.exists(self.model_save_path):
            raise ValueError(f'No model to load: {self.model_save_path}')

        # load server proto
        self.global_proto = torch.load(f'{self.model_save_path}/global_proto.pth')

        print('Loaded checkpoint global protos successfully')
        self.logger.info('Loaded checkpoint global protos successfully')

    def get_global_protos(self):
        global_protos = self.global_proto
        # global_protos = torch.stack([global_protos[class_id] for class_id in range(self.num_classes)])
        return global_protos



def generate_target(points_num, feature_dim, iter_rounds, lr=0.1, tau=0.5, device='cuda:0'):
    """
    points_num: number of points distributed on the hypersphere
    feature_dim: dimension of the feature
    iter_rounds: rounds of iteration during the optimization
    tau: hyper parameters on exponent
    """
    from torch.autograd import Variable
    import torch.nn as nn
    import torch.nn.functional as F
    vector = torch.zeros(points_num, feature_dim, requires_grad=True).to(device)
    #initialize the vector by uniform distribution
    vector_uniform = Variable(nn.init.uniform_(vector, a=-1, b=1), requires_grad=True).to(device)
    # map the vector to unit hypersphere
    input_vector_norm = F.normalize(vector_uniform, p=2, dim=1).to(device)
    for i in tqdm(range(iter_rounds)):
        input_vector_norm = F.normalize(vector_uniform, p=2, dim=1)
        # print(input_vector_norm, type(input_vector_norm))
        # matrix of vector dot
        vector_dot = torch.matmul(input_vector_norm, torch.transpose(input_vector_norm, 0, 1))
        # print(vector_dot, vector_dot.shape)
        # calculate loss function
        # 1/points_num * sum_{i=1}^{points_num} log sum_{j=1}^{points_num} e^{dot(vector_i, vector_j) / tau}
        vector_exp_sum = torch.sum(torch.exp(vector_dot / tau), dim=0, keepdim=True)
        # print(vector_exp_sum, vector_exp_sum.shape)
        vector_log_sum = torch.sum(torch.log(vector_exp_sum), dim=1)
        # print(vector_log_sum, vector_log_sum.shape)
        loss = torch.div(vector_log_sum, points_num)

        optimizer = torch.optim.SGD([vector_uniform], lr=lr)
        loss.backward()
        optimizer.step()
        vector_uniform.grad.data.zero_()

    return input_vector_norm.detach()