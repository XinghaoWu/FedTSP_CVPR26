import numpy as np
import json
import torch
from nltk.corpus import wordnet as wn
from scipy.stats import spearmanr


fallback = {
    "aquarium_fish": "fish.n.01",
    "lawn_mower": "lawn_mower.n.01",
    "maple_tree": "maple.n.02",
    "oak_tree": "oak.n.02",
    "palm_tree": "palm.n.04",
    "pine_tree": "pine.n.02",
    "willow_tree": "willow.n.02",
    "sweet_pepper": "bell_pepper.n.01",
    "pickup_truck": "pickup.n.01",
}


# ---------- 工具函数 ----------
def tri_vec(M):
    return M[np.triu_indices_from(M, k=1)]

# def delta_general(S, groups):
#     intra_vals = []
#     for g in groups:
#         block = S[np.ix_(g, g)]
#         intra_vals.append(block[np.triu_indices_from(block, k=1)])
#     intra = np.concatenate(intra_vals).mean()

#     inter_vals = []
#     for i in range(len(groups)):
#         for j in range(i + 1, len(groups)):
#             inter_vals.append(S[np.ix_(groups[i], groups[j])].ravel())
#     inter = np.concatenate(inter_vals).mean()

#     return intra - inter

def delta_general(S, groups):
    # 同组：先各自求均值
    group_means = []
    for g in groups:
        block = S[np.ix_(g, g)]
        group_means.append(block[np.triu_indices_from(block, 1)].mean())
    intra = np.mean(group_means)     # 对等权重

    # 异组：所有跨组 pair 一次拼接
    inter_vals = []
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            inter_vals.append(S[np.ix_(groups[i], groups[j])].ravel())
    inter = np.concatenate(inter_vals).mean()

    return intra - inter


# ---------- 主函数 ----------
def analyze_semantic_structure(sim_matrix_path, plot_config_path):
    # 1. 加载原型相似度矩阵
    sim_matrix = torch.load(sim_matrix_path)
    sim_matrix = np.array(sim_matrix)
    assert sim_matrix.shape[0] == sim_matrix.shape[1]

    # 2. 加载 TSNE class 和 coarse superclass 信息
    with open(plot_config_path, 'r') as f:
        config = json.load(f)
    tsne_classes = config['TSNE class']
    superclass_dict = config['superclass']

    # 3. 构造类名 → 索引映射
    class_name_to_idx = {name: idx for idx, name in enumerate(tsne_classes)}

    # 4. coarse-level Δ_sym 计算（20 coarse 类）
    groups = []
    for coarse in superclass_dict.values():
        group = [class_name_to_idx[name] for name in coarse]
        groups.append(group)
    delta_sym = delta_general(sim_matrix, groups)
    print(f"\n✅ Δ_sym (coarse-level) = {delta_sym:.4f}")

    # 5. fine-level Spearman ρ 计算（100 类）
    synsets = []
    unresolved_classes = []
    for name in tsne_classes:
        name_w = name.replace('_', ' ')
        synset_list = wn.synsets(name_w, pos=wn.NOUN)
        if synset_list:
            synsets.append(synset_list[0])
        else:
            print(f"❗️No WordNet synset for: {name_w}")
            # temp_synset_list = wn.synsets(name, pos=wn.NOUN)
            # if temp_synset_list:
            #     synsets.append(temp_synset_list[0])
            #     print(f"{name} in synset")
            # else:
            #     print(f"❗️No WordNet synset for: {name}")

                # alt = fallback.get(name, None)
                # print(f"Change name from fallbact to {alt}")
                # temp_synsets = wn.synset(alt)
                # if temp_synsets:
                #     print(f'new name in synset')
                #     synsets.append(temp_synsets)
                # else:
                #     print(f"❗️No WordNet synset for {alt} in fallback")
            synsets.append(None)
            unresolved_classes.append(name)

    # 构造 WordNet 相似度矩阵
    C = len(tsne_classes)
    W = np.zeros((C, C))
    vmask  = np.zeros((C, C), dtype=bool)   # True ⇔ 两端都有 synset
    for i in range(C):
        for j in range(i, C):
            if synsets[i] is not None and synsets[j] is not None:
                sim = synsets[i].wup_similarity(synsets[j])
                W[i, j] = W[j, i] = sim if sim is not None else 0.0
                vmask[i, j] = vmask[j, i] = True

    mask_triu = vmask[np.triu_indices(C, 1)]
    s_vec = tri_vec(sim_matrix)[mask_triu]
    w_vec = tri_vec(W)[mask_triu]
    rho, p_val = spearmanr(s_vec, w_vec)

    print(f"\n✅ Spearman ρ = {rho:.4f}   p-value = {p_val:.3e}")
    if unresolved_classes:
        print(f"\n⚠️ 仍有 {len(unresolved_classes)} 个类无 synset：{unresolved_classes}")

    rho, p_val = spearmanr(tri_vec(sim_matrix), tri_vec(W))
    print(f"\n✅ Spearman ρ = {rho:.4f}   p-value = {p_val:.3e}")
    if unresolved_classes:
        print(f"\n⚠️ Warning: {len(unresolved_classes)} classes not found in WordNet:")
        print(unresolved_classes)

# ---------- 运行示例 ----------
if __name__ == "__main__":
    # sim_matrix_path = './save/Cifar100_dir_0.1_balance_20_HtFE2_FedTSPv4_clip_global_prototype_similarity_matrix.pt'
    # sim_matrix_path = './save/Cifar100_dir_0.1_balance_20_HtFE2_AlignFed_global_prototype_similarity_matrix.pt'
    # sim_matrix_path = './save/Cifar100_dir_0.1_balance_20_HtFE2_FedTGP_global_prototype_similarity_matrix.pt'
    sim_matrix_path = './save/Cifar100_dir_0.1_balance_20_HtFE2_FedProto_global_prototype_similarity_matrix.pt'
    plot_config_path = './data_config/Cifar100_dir_0.1_balance_20/plot_config.json'
    analyze_semantic_structure(sim_matrix_path, plot_config_path)