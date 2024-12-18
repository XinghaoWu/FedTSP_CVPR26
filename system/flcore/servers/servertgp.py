import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clienttgp import clientTGP
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from collections import defaultdict
from torch.utils.data import DataLoader

import os, copy


class FedTGP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientTGP)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes

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
        logger_path = f'../logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_lamda{args.lamda}_se{args.server_epochs}_margin{args.margin_threthold}_seed{args.seed}/'
        self.set_loggers(logger_path)

        self.model_save_path = (f'../save/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_'
                                f'ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_lamda{args.lamda}_se{args.server_epochs}_margin{args.margin_threthold}_seed{args.seed}')

        self.plot_path = (f'../plot/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_'
                          f'ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_lamda{args.lamda}_se{args.server_epochs}_margin{args.margin_threthold}_seed{args.seed}')

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
            self.update_Gen()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        

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