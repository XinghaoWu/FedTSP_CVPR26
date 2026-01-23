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

        # 通信开销和计算开销统计
        self.comm_costs = []  # 每轮的上行链路通信开销（MB）
        self.downlink_comm_costs = []  # 每轮的下行链路通信开销（MB）
        self.client_comp_costs = []  # 每轮的客户端计算开销（秒）
        self.server_comp_costs = []  # 每轮的服务器计算开销（秒）

        # set logger
        if 'main.py' in self.caller_script:
            logger_path = f'../logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_lamda{args.lamda}_seed{args.seed}/'
        else:
            logger_path = f'../visualization_logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_lamda{args.lamda}_test({args.visualization_mode}_{args.visualization_dataset_type}_{args.test_data_mode})_seed{args.seed}/'
        self.set_loggers(logger_path)

        self.model_save_path = (f'../save/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_'
                                f'ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_lamda{args.lamda}_seed{args.seed}')

        self.plot_path = (f'../plot/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_'
                          f'ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_lamda{args.lamda}_seed{args.seed}')

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

    def calculate_downlink_communication_cost(self):
        """计算下行链路通信开销（MB）- FedProto 发送全局原型"""
        total_bytes = 0

        # 计算发送给客户端的全局原型大小
        try:
            global_protos = load_item(self.role, 'global_protos', self.save_folder_name)
            for class_id in global_protos.keys():
                proto = global_protos[class_id]
                proto_bytes = proto.nelement() * 4  # float32计算
                total_bytes += proto_bytes
        except Exception as e:
            print(f"Error calculating downlink communication cost: {e}")

        # 转换为MB
        total_mb = total_bytes / (1024 * 1024)

        return total_mb

    def calculate_communication_cost(self):
        """计算上行链路通信开销（MB）- FedProto 接收原型而非完整模型参数"""
        total_bytes = 0

        # 计算每个上传客户端的原型数据大小
        for client in self.selected_clients:
            # 原型存储在文件中，通过 load_item 加载
            try:
                protos = load_item(client.role, 'protos', client.save_folder_name)
                for class_id in protos.keys():
                    proto = protos[class_id]
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
    
    def test_cka_sensitivity(self):
        self.load_model()
        client_proto_dict = {}
        selected_client = [0, 8]
        for client in selected_client:
            prototype = self.clients[client].get_local_prototpye()
            client_proto_dict[client] = prototype
        
        # swap prototypes of one client
        temp_prototype = client_proto_dict[selected_client[-1]]
        swap_idx_i = [0, 1, 2]
        swap_idx_j = [5, 6, 7]
        
        for i, j in zip(swap_idx_i, swap_idx_j):
            temp = temp_prototype[i]
            temp_prototype[i] = temp_prototype[j]
            temp_prototype[j] = temp
        
        client_proto_dict[selected_client[-1]] = temp_prototype

        results = compute_prototype_similarity_missing_safe(client_proto_dict, selected_client, metrics=('pearson', 'spearman', 'cka'))
        cosine_matrix_dis = compute_cosine_distance_matrix(client_proto_dict)
        mse_matrix_dis = compute_mse_distance_matrix(client_proto_dict)
        results['cosine'] = cosine_matrix_dis
        results['mse'] = mse_matrix_dis
        print(results)

    
    def get_prototype_semantic_similarity(self, metrics=('pearson', 'spearman', 'cka'), round=None):
        if round is None:   # post-training validation
            self.load_model()

        client_proto_dict = {}
        client_protos = []
        for client in self.clients:
            prototype = client.get_local_prototpye()
            client_proto_dict[client.id] = prototype
            client_protos.append(prototype)
        
        # 增加计算全局prototype，将其作为最后一个客户端计算相似度
        global_prototye = proto_aggregation(client_protos)
        client_proto_dict[len(self.clients)] = global_prototye

        # # 预先存储每个客户端的“上三角向量”和中心化 Gram
        # tri_vec, gram_c = {}, {}
        # for cid, P in client_proto_dict.items():
        #     tri_vec[cid] = upper_tri(cosine_matrix(P))
        #     print(f'client {cid}, similarity matrix:  {tri_vec[cid]}')
        #     if 'cka' in metrics:
        #         gram_c[cid] = centered_gram(P - P.mean(0, keepdims=True))

        # # 初始化结果容器（NxN，对角线置 1）
        # results = {m: np.eye(n) for m in metrics}

        # # 双重循环计算 pair-wise
        # for i, ci in enumerate(client_ids):
        #     for j, cj in enumerate(client_ids):
        #         if j <= i:
        #             continue  # 跳过下三角和对角线
        #         if 'pearson' in metrics:
        #             r, _ = pearsonr(tri_vec[ci], tri_vec[cj])
        #             results['pearson'][i, j] = results['pearson'][j, i] = r
        #         if 'spearman' in metrics:
        #             r, _ = spearmanr(tri_vec[ci], tri_vec[cj])
        #             results['spearman'][i, j] = results['spearman'][j, i] = r
        #         if 'cka' in metrics:
        #             Kx, Ky = gram_c[ci], gram_c[cj]
        #             hsic_xy = (Kx * Ky).sum()
        #             cka = hsic_xy / np.sqrt((Kx * Kx).sum() * (Ky * Ky).sum() + 1e-10)
        #             results['cka'][i, j] = results['cka'][j, i] = cka

        client_ids = list(client_proto_dict.keys())
        results = compute_prototype_similarity_missing_safe(client_proto_dict, client_ids, metrics=('pearson', 'spearman', 'cka'))

        
        # ===== Cosine & MSE Difference Matrix =====
        cosine_matrix_dis = compute_cosine_distance_matrix(client_proto_dict)
        mse_matrix_dis = compute_mse_distance_matrix(client_proto_dict)
        
        results['cosine'] = cosine_matrix_dis
        results['mse'] = mse_matrix_dis

        print(results)
        client_ids = [i for i in range(self.args.num_clients)]
        plot_similarity_heatmap(results['pearson'], client_ids, self.plot_path, f'{self.args.dataset}_{self.args.model_family}_Pearson', round=round)
        plot_similarity_heatmap(results['spearman'], client_ids, self.plot_path, f'{self.args.dataset}_{self.args.model_family}_Spearman', round=round)
        plot_similarity_heatmap(results['cka'], client_ids, self.plot_path, f'{self.args.dataset}_{self.args.model_family}_cka', round=round)
        plot_similarity_heatmap(cosine_matrix_dis, client_ids, self.plot_path, f'{self.args.dataset}_{self.args.model_family}_Cosine_Distance', round=round)
        plot_similarity_heatmap(mse_matrix_dis, client_ids, self.plot_path, f'{self.args.dataset}_{self.args.model_family}_MSE_Distance', round=round)
        return client_ids, results
        

def compute_prototype_similarity_missing_safe(client_proto_dict, client_ids, metrics=('pearson', 'spearman', 'cka')):
    """Computes similarity between clients based on shared class subset."""
    from scipy.stats import pearsonr, spearmanr
    n = len(client_ids)
    results = {m: np.full((n, n), np.nan) for m in metrics}

    tri_vec, gram_c = {}, {}

    for cid in client_ids:
        proto = client_proto_dict[cid]
        labels = sorted(proto.keys())
        if len(labels) < 2:
            continue  # not enough to form a similarity matrix

        # build class-class cosine similarity matrix
        P = protos_to_matrix(proto, labels)
        cosine_sim = cosine_matrix(P)
        tri_vec[cid] = upper_tri(cosine_sim)

        if 'cka' in metrics:
            gram_c[cid] = centered_gram(P - P.mean(0, keepdims=True))

    for i, ci in enumerate(client_ids):
        for j, cj in enumerate(client_ids):
            if j <= i:
                continue

            proto_i, proto_j = client_proto_dict[ci], client_proto_dict[cj]
            shared_labels = get_shared_classes(proto_i, proto_j)
            if len(shared_labels) < 2:
                continue  # too few to form meaningful comparison

            # Cosine similarity matrix upper triangle
            Pi = protos_to_matrix(proto_i, shared_labels)
            Pj = protos_to_matrix(proto_j, shared_labels)
            Si = upper_tri(cosine_matrix(Pi))
            Sj = upper_tri(cosine_matrix(Pj))

            if 'pearson' in metrics and len(shared_labels) > 2:
                r, _ = pearsonr(Si, Sj)
                results['pearson'][i, j] = results['pearson'][j, i] = r
            if 'spearman' in metrics and len(shared_labels) > 2:
                r, _ = spearmanr(Si, Sj)
                results['spearman'][i, j] = results['spearman'][j, i] = r
            if 'cka' in metrics:
                Pi_centered = Pi - Pi.mean(0, keepdims=True)
                Pj_centered = Pj - Pj.mean(0, keepdims=True)
                Kx = centered_gram(Pi_centered)
                Ky = centered_gram(Pj_centered)
                hsic_xy = (Kx * Ky).sum()
                cka_val = hsic_xy / np.sqrt((Kx * Kx).sum() * (Ky * Ky).sum() + 1e-10)
                results['cka'][i, j] = results['cka'][j, i] = cka_val

    # Set diagonals to 1.0
    for m in metrics:
        np.fill_diagonal(results[m], 1.0)

    return results


def cosine_matrix(protos: np.ndarray) -> np.ndarray:
    """类-类余弦相似度矩阵，K×K"""
    protos = protos / np.linalg.norm(protos, axis=1, keepdims=True)  # L2 归一化
    return protos @ protos.T                                         # K·K^T

def upper_tri(mat: np.ndarray) -> np.ndarray:
    """提取上三角(不含对角)并展平，长度=K(K-1)/2"""
    idx = np.triu_indices_from(mat, k=1)
    return mat[idx]

def centered_gram(F: np.ndarray) -> np.ndarray:
    """F: K×D → 线性核 K×K 并中心化"""
    Kmat = F @ F.T
    n = Kmat.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ Kmat @ H

def get_shared_classes(proto_i, proto_j):
    """Returns the shared class labels between two prototype dictionaries."""
    return sorted(set(proto_i.keys()) & set(proto_j.keys()))

def protos_to_matrix(proto_dict, label_list):
    """Converts a prototype dictionary to a matrix aligned to a given label list."""
    return np.stack([proto_dict[k].detach().cpu().numpy() for k in label_list], axis=0)


def compute_cosine_distance_matrix(client_proto_dict):
    """Returns a pairwise cosine distance matrix between clients, handling missing classes."""
    client_ids = list(client_proto_dict.keys())
    n = len(client_ids)
    cosine_diff = np.zeros((n, n))
    for i, ci in enumerate(client_ids):
        for j, cj in enumerate(client_ids):
            if j < i:
                continue
            proto_i, proto_j = client_proto_dict[ci], client_proto_dict[cj]
            shared_labels = get_shared_classes(proto_i, proto_j)
            if len(shared_labels) == 0:
                cosine_diff[i, j] = cosine_diff[j, i] = np.nan
                continue
            Pi = protos_to_matrix(proto_i, shared_labels)
            Pj = protos_to_matrix(proto_j, shared_labels)
            Pi = Pi / np.linalg.norm(Pi, axis=1, keepdims=True)
            Pj = Pj / np.linalg.norm(Pj, axis=1, keepdims=True)
            cos_sim = np.sum(Pi * Pj, axis=1)
            avg_cos = cos_sim.mean()
            cosine_diff[i, j] = cosine_diff[j, i] = avg_cos
    return cosine_diff

def compute_mse_distance_matrix(client_proto_dict, eps=1e-10):
    """Returns a pairwise MSE matrix between clients, handling missing classes."""
    client_ids = list(client_proto_dict.keys())
    n = len(client_ids)
    mse_diff = np.zeros((n, n))
    for i, ci in enumerate(client_ids):
        for j, cj in enumerate(client_ids):
            if j < i:
                continue
            proto_i, proto_j = client_proto_dict[ci], client_proto_dict[cj]
            shared_labels = get_shared_classes(proto_i, proto_j)
            if len(shared_labels) == 0:
                mse_diff[i, j] = mse_diff[j, i] = 0
                continue
            Pi = protos_to_matrix(proto_i, shared_labels)
            Pj = protos_to_matrix(proto_j, shared_labels)
            mse = ((Pi - Pj) ** 2).mean()
            mse_diff[i, j] = mse_diff[j, i] = mse
    min_val = mse_diff.min()
    max_val = mse_diff.max()
    # return 1.0 - (mse_diff - min_val) / (max_val - min_val + eps)
    return mse_diff

def plot_similarity_heatmap(matrix, client_ids, save_path, title="Similarity Heatmap", cmap='viridis', round=None):
    """
    Draws a heatmap of a pairwise similarity matrix.

    Args:
        matrix (np.ndarray): (N, N) similarity matrix.
        client_ids (list of str): List of client identifiers of length N.
        title (str): Title for the plot.
        cmap (str): Matplotlib colormap to use.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    # Infer value range from matrix content
    # vmin, vmax = (0, 1) if np.all((matrix >= 0) & (matrix <= 1)) else (-1, 1)
    vmin, vmax = (0, 1)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(client_ids)))
    ax.set_yticks(np.arange(len(client_ids)))
    ax.set_xticklabels(client_ids, rotation=90, fontsize=6)
    ax.set_yticklabels(client_ids, fontsize=6)
    # Grid for readability
    ax.set_xticks(np.arange(-.5, len(client_ids), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(client_ids), 1), minor=True)
    ax.grid(which="minor", color="w", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    if round is not None:
        save_path = os.path.join(save_path, f'{title}_{round}.png')
    else:
        save_path = os.path.join(save_path, f'{title}.png')
    plt.savefig(save_path)
    plt.close()

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