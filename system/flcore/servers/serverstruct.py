import time
import numpy as np
from flcore.clients.clientstruct import clientStruct
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from utils.data_utils import read_client_data
from threading import Thread
from collections import defaultdict
import os, copy
import torch


class FedStruct(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # Check if we should skip existing experiments
        if hasattr(args, 'skip_exist') and args.skip_exist == 1:
            # Construct logger_path to check existence
            if 'main.py' in self.caller_script:
                logger_path = f'../logs/{args.dataset}/{args.model_family}/{args.algorithm}_v{args.version}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_struct({args.struct_loss_type})_rot{args.rotation}_lamda{args.lamda}_gamma{args.gamma}_beta{args.beta}_seed{args.seed}/'
            else:
                logger_path = f'../visualization_logs/{args.dataset}/{args.model_family}/{args.algorithm}_v{args.version}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_struct({args.struct_loss_type})_rot{args.rotation}_lamda{args.lamda}_gamma{args.gamma}_beta{args.beta}_test({args.visualization_mode}_{args.visualization_dataset_type}_{args.test_data_mode})_seed{args.seed}/'

            if os.path.exists(logger_path):
                print(f"\n[SKIP] Logger path already exists: {logger_path}")
                print(f"[SKIP] Skipping this hyperparameter combination (skip_exist=1)")
                self.skip_training = True
                return

        self.skip_training = False

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientStruct)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes

        # set logger
        if 'main.py' in self.caller_script:
            logger_path = f'../logs/{args.dataset}/{args.model_family}/{args.algorithm}_v{args.version}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_struct({args.struct_loss_type})_rot{args.rotation}_lamda{args.lamda}_gamma{args.gamma}_beta{args.beta}_seed{args.seed}/'
        else:
            logger_path = f'../visualization_logs/{args.dataset}/{args.model_family}/{args.algorithm}_v{args.version}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_struct({args.struct_loss_type})_rot{args.rotation}_lamda{args.lamda}_gamma{args.gamma}_beta{args.beta}_test({args.visualization_mode}_{args.visualization_dataset_type}_{args.test_data_mode})_seed{args.seed}/'
        self.set_loggers(logger_path)

        self.model_save_path = (f'../save/{args.dataset}/{args.model_family}/{args.algorithm}_v{args.version}/gr{args.global_rounds}_'
                                f'ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_struct({args.struct_loss_type})_rot{args.rotation}_lamda{args.lamda}_gamma{args.gamma}_beta{args.beta}_seed{args.seed}')

        self.plot_path = (f'../plot/{args.dataset}/{args.model_family}/{args.algorithm}_v{args.version}/gr{args.global_rounds}_'
                          f'ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/lr({args.local_learning_rate})_struct({args.struct_loss_type})_rot{args.rotation}_lamda{args.lamda}_gamma{args.gamma}_beta{args.beta}_seed{args.seed}')

        if 'main.py' in self.caller_script:
            self.final_log_path = f'../logs/{args.dataset}/{args.model_family}/{args.algorithm}_v{args.version}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/summary.txt'
        else:
            self.final_log_path = f'../visualization_logs/{args.dataset}/{args.model_family}/{args.algorithm}_v{args.version}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/test({args.visualization_mode}_{args.visualization_dataset_type}_{args.test_data_mode})/summary.txt'
        
        self.global_proto = None

    def train(self):
        # Check if training should be skipped
        if hasattr(self, 'skip_training') and self.skip_training:
            print("[SKIP] Training skipped for this hyperparameter combination.")
            return

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
                        # self.get_prototype_semantic_similarity(round=i)
            
            # if i % 5 == 0:
            #     self.get_prototype_semantic_similarity(round=i)

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_protos()
            self.send_protos()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if i - self.best_epoch >= self.args.tolerance:
                print(f'The best accuracy has not changed for {self.args.tolerance} rounds; stopping automatically.')
                self.logger.info(f'The best accuracy has not changed for {self.args.tolerance} rounds; stopping automatically.')
                break

        hyperparameters = {
            'lamda': self.args.lamda,
            'gamma': self.args.gamma,
            'beta': self.args.beta,
            'rotation': self.args.rotation,
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
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()

    def save_model(self):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # save server proto
        try:
            # PROTO = load_item(self.role, 'global_protos', self.save_folder_name)
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
        global_protos = torch.stack([global_protos[class_id] for class_id in range(self.num_classes)])
        return global_protos

    def send_protos(self):
        for client in self.clients:
            client.global_proto = copy.deepcopy(self.global_proto)

    def receive_protos(self):
        assert len(self.selected_clients) > 0
        self.uploaded_ids = []
        uploaded_proto_dicts, weights = [], []
        self.label_order = list(range(self.num_classes))

        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            # uploaded_proto_dicts.append(load_item(client.role, 'protos', client.save_folder_name))
            uploaded_proto_dicts.append(copy.deepcopy(client.local_proto))
            weights.append(float(client.train_samples))

        # ---------------- 非 version 3：保持原逻辑 ----------------
        if self.args.version not in [3, 4, 5] or self.args.rotation == 0:
            global_protos = proto_aggregation(uploaded_proto_dicts)
            # save_item(global_protos, self.role, 'global_protos', self.save_folder_name)
            self.global_proto = copy.deepcopy(global_protos)
            return

        # ---------------- version 3：等权逐类聚合 ----------------
        # P_g_dict = load_item('Server', 'global_protos', self.save_folder_name)
        P_g_dict = copy.deepcopy(self.global_proto)

        # 首轮无全局 → 先用 FedProto 均值 boot-strap，再退出
        if P_g_dict is None:
            # save_item(proto_aggregation(uploaded_proto_dicts),
            #         self.role, 'global_protos', self.save_folder_name)
            self.global_proto = proto_aggregation(uploaded_proto_dicts)
            print("[BOOTSTRAP] global_protos saved (FedProto mean).")
            return

        print('Orthogonal-Procrustes (per-class equal mean).')

        # 全局原型 (C,D) & 中心化
        P_g, _ = dict_to_mat(P_g_dict, self.label_order, self.device)
        P_g_c  = P_g - P_g.mean(0, keepdim=True)
        g_mean = P_g.mean(0, keepdim=True)          # (1,D) 用于加回平移

        # 桶：label -> list[tensor(D,)]
        agg_bucket = defaultdict(list)

        # ----- 遍历客户端：旋转对齐 + 填充桶 -----
        for proto_dict in uploaded_proto_dicts:
            P_k, mask = dict_to_mat(proto_dict, self.label_order,
                                    self.device, fill_mat=P_g)   # (C,D)
            P_k_c = P_k - P_k.mean(0, keepdim=True)
            # print(f'!!!!!!!!!!!!!!!!!!! P_g:{P_g} !!!!!!!!!!!!!!!!!!!!!!')
            # print(f'!!!!!!!!!!!!!!!!!!! P_k:{P_k} !!!!!!!!!!!!!!!!!!!!!!')

            # 仅用真实行估计旋转 R_small
            idx = mask.nonzero(as_tuple=True)[0]
            # print(f'idx numel:{idx.numel()}')
            if idx.numel() >= 2:
                # 取共有类别的子矩阵 (C',D)
                Pk_sub = P_k_c[idx]                              # (C',D)
                Pg_sub = P_g_c[idx]                              # (C',D)

                # 经典特征空间 Procrustes：min_R || Pk_sub R - Pg_sub ||_F
                M = Pk_sub.T @ Pg_sub                            # (D,D)
                # print(f'!!!!!!!!!!!!!!!!!!! M:{M} !!!!!!!!!!!!!!!!!!!!!!')
                try:
                    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
                    R_D = U @ Vt                                     # (D,D) 旋转矩阵
                    P_k_align = P_k_c @ R_D                          # (C,D)
                except torch._C._LinAlgError:
                    # print(f'!!!!!!!!!!!!!!!!!!! SVD失败，添加正则化重新拟合 !!!!!!!!!!!!!!!!!!!!!!')
                    # 添加正则化重试
                    eps = 1e-6
                    M_reg = M + eps * torch.eye(M.shape[0], device=M.device)
                    try:
                        U, _, Vt = torch.linalg.svd(M_reg, full_matrices=False)
                        R_D = U @ Vt
                        P_k_align = P_k_c @ R_D
                    except torch._C._LinAlgError:
                        # print(f'!!!!!!!!!!!!!!!!!!! 正则化后仍然失败 !!!!!!!!!!!!!!!!!!!!!!')
                        # 仍然失败，不做旋转
                        P_k_align = P_k_c
            else:
                # 共享类太少，无法估计稳定旋转 → 不做旋转
                P_k_align = P_k_c                                # (C,D)

            # 加回全局平移
            P_k_align = P_k_align + g_mean                       # (C,D)

            # 只把真实行放进桶
            for lbl_idx in idx.tolist():
                agg_bucket[lbl_idx].append(P_k_align[lbl_idx])

        # ----- 逐类等权平均 + 行级 EMA -----
        beta = getattr(self.args, "beta", 0.1)
        new_Pg = {}

        for lbl_idx in self.label_order:
            if lbl_idx in agg_bucket:                             # 至少有一个客户端拥有该类
                mean_vec = torch.stack(agg_bucket[lbl_idx], 0).mean(0)
                old_vec  = P_g[lbl_idx]
                new_Pg[lbl_idx] = ((1 - beta) * old_vec + beta * mean_vec).detach().cpu()
            else:
                # 仍无人持有该类 → 直接保持旧值
                new_Pg[lbl_idx] = P_g[lbl_idx].detach().cpu()

        # save_item(new_Pg, self.role, 'global_protos', self.save_folder_name)
        self.global_proto = copy.deepcopy(new_Pg)
        print(f"[UPDATE] global_protos saved (β={beta}).")

    
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

    def visualize_prototypes_tsne(self):
        """
        1. 加载checkpoint
        2. 收集每个客户端的Local Prototype
        3. 使用t-SNE可视化, 用颜色区分不同客户端, 用marker区分类别
        """
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        # 1. 加载checkpoint
        self.load_model()
        print("Loaded checkpoint successfully")

        # 2. 收集每个客户端的Local Prototype
        client_proto_dict = {}
        all_prototypes = []
        all_client_ids = []
        all_class_labels = []

        for client in self.clients:
            prototype = client.get_local_prototpye()
            client_proto_dict[client.id] = prototype

            # 收集该客户端的所有prototype
            for class_id, proto_tensor in prototype.items():
                all_prototypes.append(proto_tensor.detach().cpu().numpy())
                all_client_ids.append(client.id)
                all_class_labels.append(class_id)

        print(f"Collected {len(all_prototypes)} prototypes from {len(self.clients)} clients")

        # 3. 使用t-SNE降维
        all_prototypes = np.array(all_prototypes)
        print(f"Prototype shape: {all_prototypes.shape}")

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_prototypes)-1))
        prototypes_2d = tsne.fit_transform(all_prototypes)

        # 4. 可视化
        fig, ax = plt.subplots(figsize=(12, 10))

        # 定义颜色映射(每个客户端一种颜色)
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.clients)))

        # 定义marker映射(每个类别一种marker)
        markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'p', 'H',
                   'd', '<', '>', '1', '2', '3', '4', '8', 'h', '+']
        num_classes = self.num_classes

        # 为每个客户端-类别组合绘制点
        for client_id in range(len(self.clients)):
            for class_id in range(num_classes):
                # 找到属于该客户端和类别的点
                mask = np.array([(cid == client_id and cls == class_id)
                                for cid, cls in zip(all_client_ids, all_class_labels)])

                if mask.any():
                    ax.scatter(prototypes_2d[mask, 0],
                              prototypes_2d[mask, 1],
                              c=[colors[client_id]],
                              marker=markers[class_id % len(markers)],
                              s=100,
                              alpha=0.7,
                              edgecolors='black',
                              linewidths=0.5,
                              label=f'Client {client_id}, Class {class_id}' if class_id == 0 else '')

        # 创建图例: 颜色表示客户端
        from matplotlib.patches import Patch
        legend_elements_clients = [Patch(facecolor=colors[i], edgecolor='black', label=f'Client {i}')
                                  for i in range(len(self.clients))]

        # 创建图例: marker表示类别
        from matplotlib.lines import Line2D
        legend_elements_classes = [Line2D([0], [0], marker=markers[i % len(markers)], color='w',
                                         markerfacecolor='gray', markersize=10,
                                         label=f'Class {i}', markeredgecolor='black')
                                  for i in range(num_classes)]

        # 添加两个图例
        first_legend = ax.legend(handles=legend_elements_clients,
                                loc='upper left',
                                bbox_to_anchor=(1.02, 1),
                                title='Clients',
                                fontsize=8)
        ax.add_artist(first_legend)

        ax.legend(handles=legend_elements_classes,
                 loc='upper left',
                 bbox_to_anchor=(1.02, 0.5),
                 title='Classes',
                 fontsize=8)

        ax.set_title('t-SNE Visualization of Client Prototypes\n(Color=Client, Marker=Class)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        os.makedirs(self.plot_path, exist_ok=True)
        save_path = os.path.join(self.plot_path, 'prototype_tsne_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE visualization saved to: {save_path}")
        plt.close()

        return client_proto_dict, prototypes_2d

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

    def compute_effective_dimension(self):
        """
        计算所有客户端local prototype的有效维度。
        流程：
        1. 加载客户端的checkpoint
        2. 收集所有客户端的local prototype
        3. 叠放在一起（形成矩阵）
        4. 归一化（L2 norm）
        5. 去均值（center）
        6. SVD分解
        7. 计算有效维度

        Returns:
            dict: {
                'effective_dim': float,           # 有效维度（奇异值的贡献比例）
                'singular_values': np.ndarray,   # 所有奇异值
                'cumsum_ratio': np.ndarray,      # 奇异值累计贡献比例
                'total_samples': int,            # 总的原型向量数
                'feature_dim': int,              # 特征维度
                'data_matrix_shape': tuple,      # 数据矩阵的形状
            }
        """
        # 1. 加载checkpoint
        print("[INFO] Loading client checkpoints...")
        self.load_model()

        # 2. 收集所有客户端的local prototype
        print("[INFO] Collecting client prototypes...")
        all_prototypes = []

        for client in self.clients:
            prototype = client.get_local_prototpye()  # dict: {class_id: tensor}

            # 将该客户端的所有class prototype提取为向量
            for class_id in sorted(prototype.keys()):
                proto_vector = prototype[class_id].detach().cpu().numpy()
                all_prototypes.append(proto_vector)

        # 3. 叠放在一起形成矩阵 (N, D)
        # N = 总prototype数量, D = 特征维度
        data_matrix = np.vstack(all_prototypes)
        print(f"[INFO] Data matrix shape: {data_matrix.shape} (samples={data_matrix.shape[0]}, features={data_matrix.shape[1]})")

        # 4. L2归一化（每个样本单独归一化）
        print("[INFO] Normalizing data (L2 norm)...")
        norms = np.linalg.norm(data_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 避免除零
        data_matrix_normalized = data_matrix / norms

        # 5. 去均值（中心化）
        print("[INFO] Centering data (removing mean)...")
        mean_vec = data_matrix_normalized.mean(axis=0, keepdims=True)
        data_matrix_centered = data_matrix_normalized - mean_vec

        # 6. SVD分解
        print("[INFO] Computing SVD...")
        U, singular_values, Vt = np.linalg.svd(data_matrix_centered, full_matrices=False)

        # 7. 计算有效维度
        # 有效维度定义：找到使得累计方差贡献比例达到某个阈值（如90%）的维度数
        variance_explained = singular_values ** 2 / (singular_values ** 2).sum()
        cumsum_variance = np.cumsum(variance_explained)

        # 找到达到90%累计方差的维度
        threshold = 0.9
        effective_dim = np.argmax(cumsum_variance >= threshold) + 1

        print(f"\n[RESULTS] Effective Dimension Analysis:")
        print(f"  - Total samples: {data_matrix.shape[0]}")
        print(f"  - Feature dimension: {data_matrix.shape[1]}")
        print(f"  - Top 10 singular values: {singular_values[:10]}")
        print(f"  - Effective dimension (90% variance): {effective_dim}")
        print(f"  - Variance explained by effective_dim: {cumsum_variance[effective_dim-1]:.4f}")

        results = {
            'effective_dim': int(effective_dim),
            'singular_values': singular_values,
            'cumsum_ratio': cumsum_variance,
            'variance_explained': variance_explained,
            'total_samples': data_matrix.shape[0],
            'feature_dim': data_matrix.shape[1],
            'data_matrix_shape': data_matrix.shape,
            'mean_vec': mean_vec,
            'U': U,
            'Vt': Vt,
        }

        # 打印更多统计信息
        print(f"\n[STATISTICS]")
        for threshold in [0.8, 0.85, 0.9, 0.95, 0.99]:
            dim = np.argmax(cumsum_variance >= threshold) + 1
            print(f"  - Dimension for {threshold*100:.0f}% variance: {dim}")

        return results

# --------------- 工具：dict <-> tensor ----------------       
def dict_to_mat(proto_dict, label_order, device, fill_mat=None):
    C = len(label_order)
    # 推断 D（若 dict 为空，回退到 fill_mat.shape[1]）
    if proto_dict:
        D = next(iter(proto_dict.values())).numel()
    else:
        assert fill_mat is not None, "Cannot infer prototype dim."
        D = fill_mat.shape[1]

    mat   = torch.zeros(C, D, device=device)
    mask  = torch.zeros(C, dtype=torch.bool, device=device)

    for idx, lbl in enumerate(label_order):
        if lbl in proto_dict:
            mat[idx]  = proto_dict[lbl].to(device)
            mask[idx] = True
        elif fill_mat is not None:
            mat[idx]  = fill_mat[idx]           # 用上一轮 P_g 行占位
    return mat, mask


def mat_to_dict(mat, label_order):
    """(C, D) -> {label: tensor(D,)}"""
    return {lbl: mat[idx].detach().cpu() for idx, lbl in enumerate(label_order)}
# -----------------------------------------------------

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