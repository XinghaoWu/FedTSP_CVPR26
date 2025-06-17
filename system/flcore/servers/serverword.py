import time
import numpy as np
from flcore.clients.clientword import clientWord
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from utils.data_utils import read_client_data
from threading import Thread
from collections import defaultdict
import os, copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json


class FedWord(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientWord)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes

        # obtain the classes
        dataset_json_dir = f'../dataset/{args.dataset}/config.json'
        with open(dataset_json_dir, 'r') as f:
            data_config = json.load(f)
        self.args.classes = data_config['classes']

        # set logger
        if 'main.py' in self.caller_script:
            logger_path = f'../logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_lamda{args.lamda}_finallamda{args.final_lamda}_seed{args.seed}/'
        else:
            logger_path = f'../visualization_logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_lamda{args.lamda}_finallamda{args.final_lamda}_test({args.visualization_mode}_{args.visualization_dataset_type}_{args.test_data_mode})_seed{args.seed}/'
        
        self.set_loggers(logger_path)

        self.model_save_path = (f'../save/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_'
                                f'ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_lamda{args.lamda}_finallamda{args.final_lamda}_seed{args.seed}')

        self.plot_path = (f'../plot/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_'
                          f'ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_lamda{args.lamda}_finallamda{args.final_lamda}_seed{args.seed}')

        if 'main.py' in self.caller_script:
            self.final_log_path = f'../logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/summary.txt'
        else:
            self.final_log_path = f'../visualization_logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/test({args.visualization_mode}_{args.visualization_dataset_type}_{args.test_data_mode})/summary.txt'

    def train(self):
        print(f'generate global prototype')
        wordnet_similarity = compute_wordnet_similarity_natrix(self.args.classes)
        print(f'Classes: {self.args.classes}')
        print(f'WordNet similarity: {wordnet_similarity}')
        self.global_proto = generate_semantic_prototypes(wordnet_similarity, self.args.feature_dim, 100000, lr=0.1, device=self.args.device)
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
            'final lamda': self.args.final_lamda,
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

def compute_wordnet_similarity_natrix(class_names):
    from nltk.corpus import wordnet as wn
    K = len(class_names)
    sim_matrix = torch.zeros(K, K)

    for i in range(K):
        for j in range(K):
            syn1 = wn.synsets(class_names[i], pos=wn.NOUN)[0]
            syn2 = wn.synsets(class_names[j], pos=wn.NOUN)[0]
            sim = syn1.wup_similarity(syn2)
            sim_matrix[i, j] = sim if sim is not None else 0.0
    return sim_matrix


def generate_semantic_prototypes(sim_matrix, feature_dim, iter_rounds=1000, lr=0.1, device='cuda:0'):
    """
    sim_matrix: [K, K] WordNet similarity matrix, range [0, 1]
    feature_dim: dimension of the feature vectors
    """
    from scipy.stats import spearmanr, pearsonr


    K = sim_matrix.shape[0]
    sim_matrix = torch.tensor(sim_matrix, dtype=torch.float32).to(device)
    
    # Learnable prototype matrix
    prototypes = torch.randn(K, feature_dim, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([prototypes], lr=lr)

    for _ in tqdm(range(iter_rounds)):
        proto_norm = F.normalize(prototypes, p=2, dim=1)  # Ensure on hypersphere
        cos_sim = torch.matmul(proto_norm, proto_norm.T)  # [K, K] cosine similarity
        loss = F.mse_loss(cos_sim, sim_matrix)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Normalize prototypes
    final_prototypes = F.normalize(prototypes, p=2, dim=1).detach()
    final_cos_sim = torch.matmul(final_prototypes, final_prototypes.T)

    # Calculate difference
    mse_diff = F.mse_loss(final_cos_sim, sim_matrix).item()
    diff_matrix = (final_cos_sim - sim_matrix).cpu().numpy()

    # Flatten upper triangle (excluding diagonal) for correlation
    triu_indices = torch.triu_indices(K, K, offset=1)
    cos_flat = final_cos_sim[triu_indices[0], triu_indices[1]].cpu().numpy()
    sim_flat = sim_matrix[triu_indices[0], triu_indices[1]].cpu().numpy()

    spearman_corr = spearmanr(cos_flat, sim_flat).correlation
    pearson_corr = pearsonr(cos_flat, sim_flat)[0]

    print(f"[Info] Final MSE between prototype sim and WordNet sim: {mse_diff:.4f}")
    print(f"[Info] Spearman correlation: {spearman_corr:.4f}")
    print(f"[Info] Pearson correlation:  {pearson_corr:.4f}")

    return final_prototypes