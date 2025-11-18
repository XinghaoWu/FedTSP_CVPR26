import time
import numpy as np
from flcore.clients.clientprotoeval import clientProtoEval
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from utils.data_utils import read_client_data
from threading import Thread
from collections import defaultdict
import os, copy
import torch
from scipy.stats import pearsonr, spearmanr


class FedProtoEval(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientProtoEval)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes

        # tag suffix for paths
        tag_suffix = f'_tag({args.tag})' if hasattr(args, 'tag') and args.tag else ''

        # set logger
        if 'main.py' in self.caller_script:
            logger_path = f'../logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_lamda{args.lamda}{tag_suffix}_seed{args.seed}/'
        else:
            logger_path = f'../visualization_logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_lamda{args.lamda}{tag_suffix}_test({args.visualization_mode}_{args.visualization_dataset_type}_{args.test_data_mode})_seed{args.seed}/'
        self.set_loggers(logger_path)

        self.model_save_path = (f'../save/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_'
                                f'ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_lamda{args.lamda}{tag_suffix}_seed{args.seed}')

        self.plot_path = (f'../plot/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_'
                          f'ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_lamda{args.lamda}{tag_suffix}_seed{args.seed}')

        if 'main.py' in self.caller_script:
            self.final_log_path = f'../logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/summary.txt'
        else:
            self.final_log_path = f'../visualization_logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/test({args.visualization_mode}_{args.visualization_dataset_type}_{args.test_data_mode})/summary.txt'

    def train(self):
        self.feature_metric_change = []
        self.average_feature_metric_change = []
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
            client_feature_metric_change = {}
            for client in self.selected_clients:
                # before_proto = client.get_local_prototpye()
                client.train()
                # after_proto = client.get_local_prototpye()
                # abs_log_s, theta, cka = self.get_feature_metric_change(before_proto, after_proto)
                abs_log_s, theta, cka = 0, 0, 0
                client_feature_metric_change[client.id] = (abs_log_s, theta, cka)
            self.feature_metric_change.append(client_feature_metric_change)
            # 计算所有客户端三个指标的平均值
            abs_log_s_mean = np.mean([v[0] for v in client_feature_metric_change.values()])
            theta_mean = np.mean([v[1] for v in client_feature_metric_change.values()])
            cka_mean = np.mean([v[2] for v in client_feature_metric_change.values()])
            self.average_feature_metric_change.append((abs_log_s_mean, theta_mean, cka_mean))
            print(f'Average feature metric change: {self.average_feature_metric_change[-1]}')
            self.logger.info(f'Average feature metric change: {self.average_feature_metric_change[-1]}')
            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_protos()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        self.save_model(tag='last')
        hyperparameters = {
            'lamda': self.args.lamda,
            'tag': self.args.tag,
            'seed': self.args.seed
        }
        # 计算所有轮次特征指标变化的均值
        if self.average_feature_metric_change:
            avg_abs_log_s = np.mean([v[0] for v in self.average_feature_metric_change])
            avg_theta = np.mean([v[1] for v in self.average_feature_metric_change])
            avg_cka = np.mean([v[2] for v in self.average_feature_metric_change])
        else:
            avg_abs_log_s = avg_theta = avg_cka = np.nan
            
        results = {
            'Best accuracy': self.best_acc,
            'Best epoch': self.best_epoch,
            'Average abs_log_s': avg_abs_log_s,
            'Average theta': avg_theta,
            'Average cka': avg_cka,
        }
        self.log_experiment_results(self.final_log_path, hyperparameters, results)

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()

    def save_model(self, tag='best'):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # save server proto
        try:
            PROTO = load_item(self.role, 'global_protos', self.save_folder_name)
            if tag == 'best':
                torch.save(PROTO, f'{self.model_save_path}/global_proto.pth')
            elif tag == 'last':
                torch.save(PROTO, f'{self.model_save_path}/global_proto_last.pth')
            else:
                raise ValueError(f'Invalid tag: {tag}')
        except:
            print('No global protos to save')

        # save client models
        for client in self.clients:
            client.save_model(save_dir=self.model_save_path, tag=tag)
        
        # save feature metric change
        torch.save(self.feature_metric_change, f'{self.model_save_path}/feature_metric_change.pth')

    def load_model(self, tag='best'):
        if not os.path.exists(self.model_save_path):
            raise ValueError(f'No model to load: {self.model_save_path}')

        # load server proto
        if tag == 'best':
            self.global_proto = torch.load(f'{self.model_save_path}/global_proto.pth', map_location=self.device)
        elif tag == 'last':
            self.global_proto = torch.load(f'{self.model_save_path}/global_proto_last.pth', map_location=self.device)
        else:
            raise ValueError(f'Invalid tag: {tag}')

        # load client models
        for c in self.clients:
            c.load_model(save_dir=self.model_save_path, tag=tag)

        print('Loaded checkpoint models successfully')
        self.logger.info('Loaded checkpoint models successfully')

    def load_global_prototype(self):
        if not os.path.exists(self.model_save_path):
            raise ValueError(f'No model to load: {self.model_save_path}')

        # load server proto
        self.global_proto = torch.load(f'{self.model_save_path}/global_proto.pth', map_location=self.device)

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

        results = compute_prototype_similarity_missing_safe(client_proto_dict, selected_client, metrics=('pearson', 'spearman', 'cka', 'cosine', 'mse'))
        print(results)

    def testing_feature_difference(self, round=None):
        """
        ① 计算 |log s|、θ、CKA 三张矩阵
        ② 基于 self.args.model_family (示例: 'HtFE9') 把 pair 分成同构 vs. 异构
        ③ 打印三种指标的整体均值以及同构/异构均值
        """
        if round is None:
            self.load_model(tag='last')

        # -------- 1. 收集原型 --------
        client_proto_dict = {}
        for client in self.clients:
            client_proto_dict[client.id] = client.get_local_prototpye()
        client_ids = list(client_proto_dict.keys())          # 默认按 id 排序

        # -------- 2. 计算指标矩阵 --------
        metric_mats = compute_coord_metric_matrices(client_proto_dict, client_ids)
        print(f'metric_mats: {metric_mats}')

        # # -------- 3. 针对模型架构生成 arch_id 列表 --------
        # # self.args.model_family 形如 'HtFE4' → 提取数字
        # import re
        # m = re.search(r'HtFE(\d+)', self.args.model_family)
        # if m is None:
        #     raise ValueError(f"model_family '{self.args.model_family}' 不符合 HtFE# 格式")
        # num_arch = int(m.group(1))
        # arch_ids = [cid % num_arch for cid in client_ids]    # id 取模确定架构
        # print(f'arch_ids: {arch_ids}')

        # -------- 4. 打印统计 --------
        print("=== Overall Mean (上三角, 排除 NaN) ===")
        for name, mat in metric_mats.items():
            valid = mat[np.triu_indices_from(mat, k=1)]
            valid = valid[~np.isnan(valid)]
            print(f"{name:10}: {valid.mean():.4f}  (N={valid.size})")

        # print("\n=== Homogeneous vs. Heterogeneous ===")
        # for name, mat in metric_mats.items():
        #     stats = compute_hom_het_stats(mat, arch_ids)
        #     hom_mean, hom_N = stats['hom_mean']
        #     het_mean, het_N = stats['het_mean']
        #     print(f"{name:10}: hom {hom_mean:.4f} (N={hom_N}) | "
        #         f"het {het_mean:.4f} (N={het_N})")

        # -------- 5. 写入结果到txt --------
        result_dir = '../result'
        os.makedirs(result_dir, exist_ok=True)

        # 构建实验组文件名（不包含lamda）
        exp_group_name = (f'{self.args.dataset}_{self.args.model_family}_{self.args.algorithm}_'
                         f'gr{self.args.global_rounds}_ep{self.args.local_epochs}_'
                         f'bs{self.args.batch_size}_nc{self.args.num_clients}_'
                         f'lr{self.args.local_learning_rate}_tag({self.args.tag})_seed{self.args.seed}')
        result_file = os.path.join(result_dir, f'{exp_group_name}.txt')

        # 追加写入结果
        with open(result_file, 'a') as f:
            f.write(f'\nlamda={self.args.lamda}\n')
            f.write("=== Overall Mean (上三角, 排除 NaN) ===\n")
            for name, mat in metric_mats.items():
                valid = mat[np.triu_indices_from(mat, k=1)]
                valid = valid[~np.isnan(valid)]
                f.write(f"{name:10}: {valid.mean():.4f}  (N={valid.size})\n")
            f.write('-' * 50 + '\n')

        # -------- 6. 返回供后续绘图 / 保存 --------
        return metric_mats
            
    def get_feature_metric_change(self, before_proto, after_proto):
        shared = get_shared_classes(before_proto, after_proto)
        if len(shared) < 2:
            # 若共享类不足 2，统计意义不够；保持 NaN
            return 0, 0, 1

        Pi = protos_to_matrix(before_proto, shared)  # (K,D)
        Pj = protos_to_matrix(after_proto, shared)

        abs_log_s, theta, _, _, cka = procrustes_coord_metrics(Pi, Pj)
        return abs_log_s, theta, cka

    
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
        # client_proto_dict[len(self.clients)] = global_prototye

        client_ids = list(client_proto_dict.keys())
        results = compute_prototype_similarity_missing_safe(client_proto_dict, client_ids, metrics=('pearson', 'spearman', 'cka'))

        
        # ===== Cosine & MSE Difference Matrix =====
        cosine_matrix_dis = compute_cosine_distance_matrix(client_proto_dict)
        mse_matrix_dis = compute_mse_distance_matrix(client_proto_dict)
        
        results['cosine'] = cosine_matrix_dis
        results['mse'] = mse_matrix_dis

        # print('原始相似度:')
        # print(results)
        
        # 计算原始相似度的平均值
        original_averages = compute_similarity_averages(results)
        print('原始相似度平均值（排除对角线）:')
        for metric, avg in original_averages.items():
            print(f'{metric}: {avg:.4f}')
        
        # ===== Procrustes对齐后的相似度计算 =====
        print('开始Procrustes对齐...')
        results_procrustes = compute_prototype_similarity_procrustes(client_proto_dict, client_ids, metrics=('pearson', 'spearman', 'cka', 'cosine', 'mse'))
        
        # 计算Procrustes对齐后相似度的平均值
        procrustes_averages = compute_similarity_averages(results_procrustes)
        print('Procrustes对齐后相似度平均值（排除对角线）:')
        for metric, avg in procrustes_averages.items():
            print(f'{metric}: {avg:.4f}')
        
        client_ids_plot = [i for i in range(self.args.num_clients)]
        # plot_similarity_heatmap(results['pearson'], client_ids_plot, self.plot_path, f'{self.args.dataset}_{self.args.model_family}_Pearson', round=round)
        # plot_similarity_heatmap(results['spearman'], client_ids_plot, self.plot_path, f'{self.args.dataset}_{self.args.model_family}_Spearman', round=round)
        # plot_similarity_heatmap(results['cka'], client_ids_plot, self.plot_path, f'{self.args.dataset}_{self.args.model_family}_cka', round=round)
        # plot_similarity_heatmap(cosine_matrix_dis, client_ids_plot, self.plot_path, f'{self.args.dataset}_{self.args.model_family}_Cosine_Distance', round=round)
        # plot_similarity_heatmap(mse_matrix_dis, client_ids_plot, self.plot_path, f'{self.args.dataset}_{self.args.model_family}_MSE_Distance', round=round)

        
        return client_ids, results, results_procrustes
        

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


def compute_prototype_similarity_procrustes(client_proto_dict, client_ids, metrics=('pearson', 'spearman', 'cka')):
    """Computes similarity between clients after Procrustes alignment."""
    n = len(client_ids)
    results = {m: np.full((n, n), np.nan) for m in metrics}

    for i, ci in enumerate(client_ids):
        for j, cj in enumerate(client_ids):
            if j <= i:
                continue

            proto_i, proto_j = client_proto_dict[ci], client_proto_dict[cj]
            shared_labels = get_shared_classes(proto_i, proto_j)
            if len(shared_labels) < 2:
                continue  # too few to form meaningful comparison

            # 转换为numpy矩阵
            Pi = protos_to_matrix(proto_i, shared_labels)
            Pj = protos_to_matrix(proto_j, shared_labels)
            
            # Procrustes对齐：将Pi对齐到Pj的空间
            Pi_aligned = procrustes_align(Pi, Pj)
            
            # 计算对齐后的相似度矩阵
            Si = upper_tri(cosine_matrix(Pi_aligned))
            Sj = upper_tri(cosine_matrix(Pj))

            if 'pearson' in metrics and len(shared_labels) > 2:
                r, _ = pearsonr(Si, Sj)
                results['pearson'][i, j] = results['pearson'][j, i] = r
            if 'spearman' in metrics and len(shared_labels) > 2:
                r, _ = spearmanr(Si, Sj)
                results['spearman'][i, j] = results['spearman'][j, i] = r
            if 'cka' in metrics:
                Pi_centered = Pi_aligned - Pi_aligned.mean(0, keepdims=True)
                Pj_centered = Pj - Pj.mean(0, keepdims=True)
                Kx = centered_gram(Pi_centered)
                Ky = centered_gram(Pj_centered)
                hsic_xy = (Kx * Ky).sum()
                cka_val = hsic_xy / np.sqrt((Kx * Kx).sum() * (Ky * Ky).sum() + 1e-10)
                results['cka'][i, j] = results['cka'][j, i] = cka_val
            if 'cosine' in metrics and len(shared_labels) > 0:
                Pi_aligned_norm = Pi_aligned / np.linalg.norm(Pi_aligned, axis=1, keepdims=True)
                Pj_norm = Pj / np.linalg.norm(Pj, axis=1, keepdims=True)
                cos_sim = np.sum(Pi_aligned_norm * Pj_norm, axis=1)
                avg_cos = cos_sim.mean()
                results['cosine'][i, j] = results['cosine'][j, i] = avg_cos
            if 'mse' in metrics and len(shared_labels) > 0:
                mse = ((Pi_aligned - Pj) ** 2).mean()
                results['mse'][i, j] = results['mse'][j, i] = mse

    # Set diagonals appropriately
    for m in metrics:
        if m == 'mse':
            np.fill_diagonal(results[m], 0.0)  # MSE距离，自己与自己的距离为0
        else:
            np.fill_diagonal(results[m], 1.0)  # 相似度指标，自己与自己的相似度为1

    return results


def procrustes_align(X, Y):
    """
    使用Procrustes分析将X对齐到Y的空间
    
    Args:
        X: 需要对齐的矩阵 (K, D)
        Y: 目标空间的矩阵 (K, D)
    
    Returns:
        X_aligned: 对齐后的X矩阵
    """
    # 中心化
    X_centered = X - X.mean(0, keepdims=True)
    Y_centered = Y - Y.mean(0, keepdims=True)
    
    # 计算SVD
    M = X_centered.T @ Y_centered
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    
    # 计算旋转矩阵
    R = Vt.T @ U.T
    
    # 应用旋转和对齐
    X_aligned = (X_centered @ R) + Y.mean(0, keepdims=True)
    
    return X_aligned


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


def compute_similarity_averages(similarity_results):
    """
    计算各个相似度指标的平均值，排除对角线元素（自己与自己的相似度）
    
    Args:
        similarity_results (dict): 包含不同相似度指标的字典，每个指标对应一个矩阵
        
    Returns:
        dict: 每个指标的平均值
    """
    averages = {}
    for metric_name, matrix in similarity_results.items():
        if matrix is None or np.isnan(matrix).all():
            averages[metric_name] = np.nan
            continue
            
        # 创建上三角矩阵的掩码（排除对角线）
        n = matrix.shape[0]
        mask = np.triu(np.ones((n, n)), k=1)  # k=1 排除对角线
        
        # 应用掩码并计算平均值
        masked_values = matrix[mask.astype(bool)]
        # 排除NaN值
        valid_values = masked_values[~np.isnan(masked_values)]
        
        if len(valid_values) > 0:
            averages[metric_name] = np.mean(valid_values)
        else:
            averages[metric_name] = np.nan
    
    return averages

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


# ---------- 单对客户端的缩放 |log s|、旋转角 θ、CKA ----------
def procrustes_coord_metrics(P: np.ndarray, Q: np.ndarray):
    """
    Args
    ----
        P, Q : (K, D) numpy arrays (同一组 shared_labels)
    Returns
    -------
        abs_log_s : float  |log s|
        theta_deg : float  平均旋转角 (°)
        cka       : float  线性 CKA ∈ [0,1]
    """
    # --- 去均值 ---
    Pc = P - P.mean(0, keepdims=True)
    Qc = Q - Q.mean(0, keepdims=True)

    # --- SVD 求最优旋转 & 缩放 ---
    M = Pc.T @ Qc                       # (D,D)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt                          # 旋转
    # s = S.sum() / (np.linalg.norm(Pc)**2 + 1e-12)
    s = S.sum() / (np.linalg.norm(Qc)**2 + 1e-12)   # 20250911 修改为Qc

    abs_log_s = abs(np.log(s + 1e-12))  # 对称尺度差异

    # --- Frobenius 范数差异 ---
    norm_diff = abs(np.log(np.linalg.norm(Pc, 'fro') + 1e-12) - np.log(np.linalg.norm(Qc, 'fro') + 1e-12))
    # P_norm = np.log(np.linalg.norm(Pc, 'fro') + 1e-12)
    # Q_norm = np.log(np.linalg.norm(Qc, 'fro') + 1e-12)
    # norm_diff = abs(P_norm - Q_norm) / min(P_norm, Q_norm)

    # # --- 旋转角：用对角平均近似 principal angle ---
    # cos_theta = np.clip(np.diag(R).mean(), -1.0, 1.0)
    # theta_deg = np.degrees(np.arccos(cos_theta))

    from scipy.linalg import subspace_angles, svd
    def mean_principal_angle(P, Q, k=None):
        # from scipy.linalg import subspace_angles, svd

        """
        P, Q : (K, D)  numpy arrays  (同一组 shared labels)
        k    : 取多少维的列空间做比较；默认取 rank(PQ) 的最小值
        """
        # 1) 去均值
        Pc = P - P.mean(0, keepdims=True)
        Qc = Q - Q.mean(0, keepdims=True)

        # 2) 取列空间 (左奇异向量)
        Ui, _, _ = svd(Pc, full_matrices=False)
        Uj, _, _ = svd(Qc, full_matrices=False)

        if k is None:
            k = min(Ui.shape[1], Uj.shape[1])
        Ui, Uj = Ui[:, :k], Uj[:, :k]

        # 3) principal angles
        angles_rad = subspace_angles(Ui, Uj)       # 返回长度 = k
        return np.degrees(angles_rad).mean()
    
    def mean_principal_angle_right(P, Q, k=None, energy=0.90, weight=True):
        # 1) 去均值
        Pc = P - P.mean(0, keepdims=True)
        Qc = Q - Q.mean(0, keepdims=True)
        # 2) 取行空间（右奇异向量 V ∈ R^{D×r}）
        _, S_p, Vt_p = svd(Pc, full_matrices=False)
        _, S_q, Vt_q = svd(Qc, full_matrices=False)
        V_p, V_q = Vt_p.T, Vt_q.T  # 形状：(D, r)
        
        if k is None:
            # # 与左侧保持同样的“轻量默认”：取 k=2；你也可以改成能量阈值策略
            # k = min(V_p.shape[1], V_q.shape[1], 2)

            # 选择覆盖 energy 的维度数
            def choose_k(S, thr=energy, eps=1e-12):
                e = (S**2)
                c = np.cumsum(e) / (e.sum() + eps)
                return int(np.searchsorted(c, thr)) + 1

            kp = choose_k(S_p, energy); kq = choose_k(S_q, energy)
            k = min(kp, kq)
            k = max(k, 3)
            # print(f'k: {k}', end=' ')

        # V_p, V_q = V_p[:, :k], V_q[:, :k]
        # angles_rad = subspace_angles(V_p, V_q)
        # return float(np.degrees(angles_rad).mean())

        Vp, Vq = V_p[:, :k], V_q[:, :k]

        # 角度（逐维）+ 按能量加权平均
        ang = np.degrees(subspace_angles(Vp, Vq))  # len = k
        # 用几何均值权重（稳健）
        if weight:
            w = np.sqrt((S_p[:k]**2) * (S_q[:k]**2))
        else:
            w = np.ones(k)
        w = w / (w.sum() + 1e-12)
        # print(f'right angle: {float((ang * w).sum())}')
        return float((ang * w).sum())
 
    def right_whiten(X, eps=1e-6):
        Xm = X - X.mean(0, keepdims=True)
        C = Xm.T @ Xm
        # 对称平方根逆
        evals, evecs = np.linalg.eigh(C + eps * np.eye(C.shape[0]))
        W = evecs @ np.diag(1.0 / np.sqrt(np.clip(evals, eps, None))) @ evecs.T
        return Xm @ W

    theta_deg = mean_principal_angle(P, Q, k=min(2, P.shape[1]))
    # theta_deg_right = mean_principal_angle_right(P, Q, k=min(2, P.shape[1]))
    # print('right angle: ')
    theta_deg_right = mean_principal_angle_right(P, Q, weight=True)
    # print(f'right angle whiten: ')
    Pw, Qw = right_whiten(P), right_whiten(Q)
    theta_deg_right_whiten = mean_principal_angle_right(Pw, Qw, weight=False)

    # --- 线性 CKA ---
    def linear_cka(X, Y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.float32)
        X_centered = X - X.mean(dim=0, keepdim=True)
        Y_centered = Y - Y.mean(dim=0, keepdim=True)

        # Compute Gram matrices (similarity matrices)
        gram_X = X_centered @ X_centered.T
        gram_Y = Y_centered @ Y_centered.T

        # Compute the CKA score
        cka_value = torch.trace(gram_X @ gram_Y) / (torch.norm(gram_X) * torch.norm(gram_Y))
        # return cka_value
        return float(cka_value)

    cka_val = linear_cka(P, Q)
    return abs_log_s, norm_diff, theta_deg, theta_deg_right, theta_deg_right_whiten, cka_val


# ---------- 全客户端两两计算矩阵 ----------
def compute_coord_metric_matrices(client_proto_dict, client_ids):
    """
    Returns
    -------
        metrics : dict
            {
              'abs_log_s': np.ndarray (n,n),
              'norm_diff': np.ndarray (n,n),
              'theta'    : np.ndarray (n,n),
              'theta_right'    : np.ndarray (n,n),
              'cka'      : np.ndarray (n,n),
            }
    """
    n = len(client_ids)
    mats = {m: np.full((n, n), np.nan) for m in ('abs_log_s', 'norm_diff', 'theta', 'theta_right', 'theta_right_whiten', 'cka')}

    for i, ci in enumerate(client_ids):
        for j, cj in enumerate(client_ids):
            if j <= i:
                continue
            proto_i, proto_j = client_proto_dict[ci], client_proto_dict[cj]
            shared = get_shared_classes(proto_i, proto_j)
            if len(shared) < 2:
                # 若共享类不足 2，统计意义不够；保持 NaN
                continue

            Pi = protos_to_matrix(proto_i, shared)  # (K,D)
            Pj = protos_to_matrix(proto_j, shared)

            abs_log_s, norm_diff, theta, theta_right, theta_right_whiten, cka = procrustes_coord_metrics(Pi, Pj)

            # 对称填充
            mats['abs_log_s'][i, j] = mats['abs_log_s'][j, i] = abs_log_s
            mats['norm_diff'][i, j] = mats['norm_diff'][j, i] = norm_diff
            mats['theta'][i, j]     = mats['theta'][j, i]     = theta
            mats['theta_right'][i, j] = mats['theta_right'][j, i] = theta_right
            mats['theta_right_whiten'][i, j] = mats['theta_right_whiten'][j, i] = theta_right_whiten
            mats['cka'][i, j]       = mats['cka'][j, i]       = cka

    # 对角线：自己与自己
    np.fill_diagonal(mats['abs_log_s'], 0.0)
    np.fill_diagonal(mats['norm_diff'], 0.0)
    np.fill_diagonal(mats['theta'],      0.0)
    np.fill_diagonal(mats['theta_right'], 0.0)
    np.fill_diagonal(mats['theta_right_whiten'], 0.0)
    np.fill_diagonal(mats['cka'],        1.0)
    return mats


from typing import Dict, Tuple, List

def compute_hom_het_stats(metric_mat: np.ndarray,
                          arch_ids: List[int]) -> Dict[str, Tuple[float, int]]:
    """
    Args
    ----
        metric_mat : (n,n) numpy 矩阵，主对角已设 0/1，可能含 NaN
        arch_ids   : 长度 n 的列表，每个客户端的“架构编号”
                     — 同构: arch_ids[i] == arch_ids[j]
                     — 异构: arch_ids[i] != arch_ids[j]

    Returns
    -------
        dict{ 'hom_mean': (均值, 样本数),
              'het_mean': (均值, 样本数) }
        若某组无有效样本，均值设为 np.nan，样本数 0
    """
    n = metric_mat.shape[0]
    assert len(arch_ids) == n

    tri_idx = np.triu_indices(n, k=1)          # 上三角排除对角
    # print(f'tri_idx: {tri_idx}')
    vals    = metric_mat[tri_idx]
    i_idx, j_idx = tri_idx

    hom_mask = (np.array(arch_ids)[i_idx] == np.array(arch_ids)[j_idx])
    het_mask = ~hom_mask
    # print(f'hom_mask: {hom_mask}')
    # print(f'het_mask: {het_mask}')

    def _stat(mask):
        v = vals[mask]
        v = v[~np.isnan(v)]
        return (v.mean() if v.size else np.nan, v.size)

    return {'hom_mean': _stat(hom_mask),
            'het_mean': _stat(het_mask)}
