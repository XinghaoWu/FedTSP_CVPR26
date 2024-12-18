import time
import numpy as np
from flcore.clients.clientproto import clientProto
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from utils.data_utils import read_client_data
from threading import Thread
from collections import defaultdict
import os, copy
import torch


class FedProto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientProto)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes

        # set logger
        logger_path = f'../logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_lamda{args.lamda}_seed{args.seed}/'
        self.set_loggers(logger_path)

        self.model_save_path = (f'../save/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_'
                                f'ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_lamda{args.lamda}_seed{args.seed}')

        self.plot_path = (f'../plot/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_'
                          f'ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_lamda{args.lamda}_seed{args.seed}')

        self.final_log_path = f'../logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/summary.txt'

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

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

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_protos()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        hyperparameters = {
            'lamda': self.args.lamda,
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
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()

    def save_model(self):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # save server proto
        try:
            PROTO = load_item(self.role, 'global_protos', self.save_folder_name)
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
    #     print(similarity_matrix_np.shape)
    #     print(similarity_matrix_np.min(), similarity_matrix_np.max())
    #     print(len(self.args.classes), similarity_matrix_np.shape[0])
    #
    #     # Get class names from self.args.classes
    #     class_names = self.args.classes if hasattr(self.args, 'classes') else [f'Class {i}' for i in
    #                                                                            range(similarity_matrix_np.shape[0])]
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

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            protos = load_item(client.role, 'protos', client.save_folder_name)
            uploaded_protos.append(protos)
            
        global_protos = proto_aggregation(uploaded_protos)
        save_item(global_protos, self.role, 'global_protos', self.save_folder_name)
    

# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221
def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label