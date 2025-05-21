import numpy as np
import torch

# S : (C, C) numpy array of cosine similarities
# a_idx, v_idx : lists of class indices
# def delta_sem(S, a_idx, v_idx):
#     AA = S[np.ix_(a_idx, a_idx)].mean()          # animal–animal
#     AV = S[np.ix_(a_idx, v_idx)].mean()          # animal–vehicle
#     return AA - AV

# def delta_sem(S, a_idx, v_idx):
#     # --- 1. Animal–Animal (exclude self) ---------------------------
#     AA_block = S[np.ix_(a_idx, a_idx)]
#     triu = np.triu_indices_from(AA_block, k=1)   # k=1  => exclude diagonal
#     AA = AA_block[triu].mean()                  # ½∑_{i≠j}  ➔ 已除以对数

#     # --- 2. Animal–Vehicle ----------------------------------------
#     AV = S[np.ix_(a_idx, v_idx)].mean()

#     return AA - AV

def delta_sym(S, a_idx, v_idx):
    # --- 1. Animal–Animal ----------------------------
    AA_block = S[np.ix_(a_idx, a_idx)]
    AA = AA_block[np.triu_indices_from(AA_block, k=1)]
    # --- 2. Vehicle–Vehicle --------------------------
    VV_block = S[np.ix_(v_idx, v_idx)]
    VV = VV_block[np.triu_indices_from(VV_block, k=1)]
    # intra = np.concatenate([AA, VV]).mean()
    intra = (AA.mean() + VV.mean()) / 2

    # --- 3. Animal–Vehicle ---------------------------
    AV = S[np.ix_(a_idx, v_idx)].mean()

    return intra - AV

S = torch.load('./save/Cifar10_dir_0.1_balance_20_HtFE2_FedTSPv4_bert_global_prototype_similarity_matrix.pt')
S = np.array(S)
# Example
a_idx = [0, 1, 2, 3, 4, 5]      # bird, cat, deer, dog, frog, horse
v_idx = [6, 7, 8, 9]            # ship, truck, automobile, airplane
Δ_sem = delta_sym(S, a_idx, v_idx)
print(f"Δ_sym= {Δ_sem:.4f}")

import nltk
# nltk.download('wordnet'); nltk.download('omw-1.4')

import numpy as np
from nltk.corpus import wordnet as wn
from scipy.stats import spearmanr

# 1 ─────────── 明确 CIFAR-10 → synset 映射（手动选最常用词义）
synset_map = {    
    'bird'       : wn.synset('bird.n.01'),
    'cat'        : wn.synset('cat.n.01'),
    'deer'       : wn.synset('deer.n.01'),
    'dog'        : wn.synset('dog.n.01'),
    'horse'      : wn.synset('horse.n.01'),
    'frog'       : wn.synset('frog.n.01'),
    'ship'       : wn.synset('ship.n.01'),
    'truck'      : wn.synset('truck.n.01'),
    'automobile' : wn.synset('car.n.01'),        # 用 car 更常见
    'airplane'   : wn.synset('airplane.n.01'),
}

labels = list(synset_map.keys())                # 排序一定要固定
print(labels)

# 2 ─────────── 生成 WordNet 相似度矩阵
C = len(labels)
W = np.zeros((C, C))
for i in range(C):
    for j in range(i, C):
        sim = synset_map[labels[i]].wup_similarity(synset_map[labels[j]])
        W[i, j] = W[j, i] = sim if sim is not None else 0.0

# 3 ─────────── 工具：取上三角向量
def tri_vec(M):
    return M[np.triu_indices_from(M, k=1)]

print(S)
print(W)

# 4 ─────────── 计算 ρ
rho, p_val = spearmanr(tri_vec(S), tri_vec(W))
print(f"Spearman ρ = {rho:.4f}   p = {p_val:.3e}")