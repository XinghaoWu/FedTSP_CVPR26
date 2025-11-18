import torch
import torch.optim as optim
import numpy as np
from scipy.linalg import subspace_angles, svd


def cosine_similarity(P, Q, detach_norm=False):
    """
    Compute cosine similarity between P and Q
    P, Q: shape (k, D)
    detach_norm: whether to detach the norm in denominator
    """
    # Flatten to vectors
    P_flat = P.flatten()
    Q_flat = Q.flatten()

    # Compute dot product
    dot_product = torch.sum(P_flat * Q_flat)

    # Compute norms
    if detach_norm:
        P_norm = torch.sqrt(torch.sum(P_flat ** 2)).detach()
        Q_norm = torch.sqrt(torch.sum(Q_flat ** 2)).detach()
    else:
        P_norm = torch.sqrt(torch.sum(P_flat ** 2))
        Q_norm = torch.sqrt(torch.sum(Q_flat ** 2))

    # Compute cosine similarity
    cosine_sim = dot_product / (P_norm * Q_norm + 1e-8)

    return cosine_sim


def mse_loss(P, Q):
    """
    Compute MSE (Mean Squared Error) between P and Q
    P, Q: shape (k, D)
    Returns MSE value
    """
    return torch.mean((P - Q) ** 2)


def linear_cka(P, Q, detach_norm=False):
    """
    Compute linear CKA similarity between P and Q
    P, Q: shape (k, D)
    detach_norm: whether to detach the norm in denominator
    """
    # P: (k, D), Q: (k, D)
    # Gram matrix: P @ P.T and Q @ Q.T

    # Center the representations
    P_centered = P - P.mean(dim=0, keepdim=True)
    Q_centered = Q - Q.mean(dim=0, keepdim=True)

    # Compute Gram matrices
    P_gram = P_centered @ P_centered.T  # (k, k)
    Q_gram = Q_centered @ Q_centered.T  # (k, k)

    # Compute HSIC
    hsic_pq = torch.sum(P_gram * Q_gram)

    if detach_norm:
        hsic_pp = torch.sum(P_gram * P_gram).detach()
        hsic_qq = torch.sum(Q_gram * Q_gram).detach()
    else:
        hsic_pp = torch.sum(P_gram * P_gram)
        hsic_qq = torch.sum(Q_gram * Q_gram)

    # Compute CKA
    cka = hsic_pq / (torch.sqrt(hsic_pp * hsic_qq) + 1e-8)

    return cka


def compute_norm(X):
    """Compute Frobenius norm of tensor X"""
    return torch.sqrt(torch.sum(X ** 2)).item()


def compute_procrustes_metrics(P, Q):
    """
    Compute Procrustes-based metrics: scaling, rotation, and norm difference
    P, Q: torch tensors of shape (k, D)
    Returns:
        abs_log_s: absolute log scaling difference
        norm_diff: Frobenius norm difference (log scale)
        theta_deg_left: mean principal angle of left singular vectors (degrees)
        theta_deg_right: mean principal angle of right singular vectors (degrees, weighted)
        theta_deg_right_whiten: mean principal angle after whitening (degrees)
    """
    # Convert to numpy for scipy operations
    P_np = P.detach().cpu().numpy() if isinstance(P, torch.Tensor) else P
    Q_np = Q.detach().cpu().numpy() if isinstance(Q, torch.Tensor) else Q

    # Center the representations
    Pc = P_np - P_np.mean(0, keepdims=True)
    Qc = Q_np - Q_np.mean(0, keepdims=True)

    # --- Compute optimal rotation & scaling via SVD ---
    M = Pc.T @ Qc  # (D, D)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    # R = U @ Vt  # Rotation matrix (not used here but available if needed)
    s = S.sum() / (np.linalg.norm(Qc)**2 + 1e-12)

    # Scaling difference
    abs_log_s = abs(np.log(s + 1e-12))

    # Frobenius norm difference
    norm_diff = abs(np.log(np.linalg.norm(Pc, 'fro') + 1e-12) -
                    np.log(np.linalg.norm(Qc, 'fro') + 1e-12))

    # --- Rotation angles ---
    def mean_principal_angle_left(P, Q, k=None):
        """
        Compute mean principal angle using left singular vectors (column space)
        """
        # Center
        Pc = P - P.mean(0, keepdims=True)
        Qc = Q - Q.mean(0, keepdims=True)

        # Get left singular vectors
        Ui, _, _ = svd(Pc, full_matrices=False)
        Uj, _, _ = svd(Qc, full_matrices=False)

        if k is None:
            k = min(Ui.shape[1], Uj.shape[1], 2)
        Ui, Uj = Ui[:, :k], Uj[:, :k]

        # Compute principal angles
        angles_rad = subspace_angles(Ui, Uj)
        return np.degrees(angles_rad).mean()

    def mean_principal_angle_right(P, Q, k=None, energy=0.90, weight=True):
        """
        Compute mean principal angle using right singular vectors (row space)
        """
        # Center
        Pc = P - P.mean(0, keepdims=True)
        Qc = Q - Q.mean(0, keepdims=True)

        # Get right singular vectors
        _, S_p, Vt_p = svd(Pc, full_matrices=False)
        _, S_q, Vt_q = svd(Qc, full_matrices=False)
        V_p, V_q = Vt_p.T, Vt_q.T  # Shape: (D, r)

        if k is None:
            # Choose k that covers 'energy' proportion of variance
            def choose_k(S, thr=energy, eps=1e-12):
                e = (S**2)
                c = np.cumsum(e) / (e.sum() + eps)
                return int(np.searchsorted(c, thr)) + 1

            kp = choose_k(S_p, energy)
            kq = choose_k(S_q, energy)
            k = min(kp, kq)
            k = max(k, 3)

        Vp, Vq = V_p[:, :k], V_q[:, :k]

        # Compute angles
        ang = np.degrees(subspace_angles(Vp, Vq))

        # Weighted average
        if weight:
            w = np.sqrt((S_p[:k]**2) * (S_q[:k]**2))
        else:
            w = np.ones(k)
        w = w / (w.sum() + 1e-12)
        return float((ang * w).sum())

    def right_whiten(X, eps=1e-6):
        """
        Whiten the data using eigendecomposition
        """
        Xm = X - X.mean(0, keepdims=True)
        C = Xm.T @ Xm
        # Symmetric square root inverse
        evals, evecs = np.linalg.eigh(C + eps * np.eye(C.shape[0]))
        W = evecs @ np.diag(1.0 / np.sqrt(np.clip(evals, eps, None))) @ evecs.T
        return Xm @ W

    # Compute rotation angles
    theta_deg_left = mean_principal_angle_left(P_np, Q_np, k=min(2, P_np.shape[1]))
    theta_deg_right = mean_principal_angle_right(P_np, Q_np, weight=True)

    # Compute rotation angle after whitening
    Pw, Qw = right_whiten(P_np), right_whiten(Q_np)
    theta_deg_right_whiten = mean_principal_angle_right(Pw, Qw, weight=False)

    return abs_log_s, norm_diff, theta_deg_left, theta_deg_right, theta_deg_right_whiten


def train_alignment_to_prototype(P_init, Q_init, method, num_epochs=100, lr=0.01):
    """
    Train alignment of P and Q to their prototype (mean)
    method: 'cosine', 'cosine_detach', 'cka', 'cka_detach', 'mse'

    Key difference: P and Q are aligned to prototype = (P + Q) / 2,
    not directly to each other.
    """
    P = P_init.clone().requires_grad_(True)
    Q = Q_init.clone().requires_grad_(True)

    optimizer = optim.SGD([P, Q], lr=lr)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Compute prototype as the mean of P and Q (detached, no gradient)
        prototype = ((P + Q) / 2).detach()

        # Compute loss for P to prototype alignment
        if method == 'cosine':
            loss_P = 1 - cosine_similarity(P, prototype, detach_norm=False)
            loss_Q = 1 - cosine_similarity(Q, prototype, detach_norm=False)
        elif method == 'cosine_detach':
            loss_P = 1 - cosine_similarity(P, prototype, detach_norm=True)
            loss_Q = 1 - cosine_similarity(Q, prototype, detach_norm=True)
        elif method == 'cka':
            loss_P = 1 - linear_cka(P, prototype, detach_norm=False)
            loss_Q = 1 - linear_cka(Q, prototype, detach_norm=False)
        elif method == 'cka_detach':
            loss_P = 1 - linear_cka(P, prototype, detach_norm=True)
            loss_Q = 1 - linear_cka(Q, prototype, detach_norm=True)
        elif method == 'mse':
            loss_P = mse_loss(P, prototype)
            loss_Q = mse_loss(Q, prototype)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Total loss: sum of both alignment losses
        loss = loss_P + loss_Q

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f} "
                  f"(P: {loss_P.item():.6f}, Q: {loss_Q.item():.6f})")

    return P.detach(), Q.detach()


def print_statistics(P_init, Q_init, P_final, Q_final, method_name):
    """Print statistics"""
    print(f"\n{'='*60}")
    print(f"Method: {method_name}")
    print(f"{'='*60}")

    # Compute norms
    P_init_norm = compute_norm(P_init)
    P_final_norm = compute_norm(P_final)
    Q_init_norm = compute_norm(Q_init)
    Q_final_norm = compute_norm(Q_final)

    # Compute similarities
    with torch.no_grad():
        cosine_init = cosine_similarity(P_init, Q_init, detach_norm=False).item()
        cosine_final = cosine_similarity(P_final, Q_final, detach_norm=False).item()
        cka_init = linear_cka(P_init, Q_init, detach_norm=False).item()
        cka_final = linear_cka(P_final, Q_final, detach_norm=False).item()

    # Compute Procrustes metrics
    abs_log_s_init, norm_diff_init, theta_left_init, theta_right_init, theta_whiten_init = \
        compute_procrustes_metrics(P_init, Q_init)
    abs_log_s_final, norm_diff_final, theta_left_final, theta_right_final, theta_whiten_final = \
        compute_procrustes_metrics(P_final, Q_final)

    # Print results
    print(f"\n1. Norm of P:")
    print(f"   Before: {P_init_norm:.6f}")
    print(f"   After:  {P_final_norm:.6f}")
    print(f"   Change: {P_final_norm - P_init_norm:+.6f} ({(P_final_norm/P_init_norm - 1)*100:+.2f}%)")

    print(f"\n2. Norm of Q:")
    print(f"   Before: {Q_init_norm:.6f}")
    print(f"   After:  {Q_final_norm:.6f}")
    print(f"   Change: {Q_final_norm - Q_init_norm:+.6f} ({(Q_final_norm/Q_init_norm - 1)*100:+.2f}%)")

    print(f"\n3. Cosine Similarity (between P and Q):")
    print(f"   Before: {cosine_init:.6f}")
    print(f"   After:  {cosine_final:.6f}")
    print(f"   Change: {cosine_final - cosine_init:+.6f}")

    print(f"\n4. CKA Similarity (between P and Q):")
    print(f"   Before: {cka_init:.6f}")
    print(f"   After:  {cka_final:.6f}")
    print(f"   Change: {cka_final - cka_init:+.6f}")

    print(f"\n5. Scaling Difference (|log s|):")
    print(f"   Before: {abs_log_s_init:.6f}")
    print(f"   After:  {abs_log_s_final:.6f}")
    print(f"   Change: {abs_log_s_final - abs_log_s_init:+.6f}")

    print(f"\n6. Norm Difference (log scale):")
    print(f"   Before: {norm_diff_init:.6f}")
    print(f"   After:  {norm_diff_final:.6f}")
    print(f"   Change: {norm_diff_final - norm_diff_init:+.6f}")

    print(f"\n7. Rotation Angle - Left (degrees):")
    print(f"   Before: {theta_left_init:.4f}")
    print(f"   After:  {theta_left_final:.4f}")
    print(f"   Change: {theta_left_final - theta_left_init:+.4f}")

    print(f"\n8. Rotation Angle - Right (degrees, weighted):")
    print(f"   Before: {theta_right_init:.4f}")
    print(f"   After:  {theta_right_final:.4f}")
    print(f"   Change: {theta_right_final - theta_right_init:+.4f}")

    print(f"\n9. Rotation Angle - Right Whitened (degrees):")
    print(f"   Before: {theta_whiten_init:.4f}")
    print(f"   After:  {theta_whiten_final:.4f}")
    print(f"   Change: {theta_whiten_final - theta_whiten_init:+.4f}")
    print()


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Manually specify dimensions
    k = 10  # Number of samples
    D = 512  # Feature dimension

    print(f"Initializing representation vectors P and Q with shape: ({k}, {D})")

    # Initialize P and Q with different correlation structures to get lower CKA similarity
    # Strategy: P has low-rank structure, Q has different structure/high-rank

    # P: Low-rank structure (few dominant components)
    # Generate P as a sum of few random rank-1 matrices
    rank_P = 3  # Low rank
    P_init = torch.zeros(k, D)
    for i in range(rank_P):
        u = torch.randn(k, 1)
        v = torch.randn(1, D)
        P_init += u @ v
    P_init = P_init * 0.5 / np.sqrt(rank_P)  # Scale to smaller norm

    # Q: High-rank structure with different pattern
    # Option 1: Use full-rank random with block structure
    Q_init = torch.randn(k, D) * 2.0

    # Add block-diagonal structure to Q to make it different from P
    # This changes the correlation pattern
    block_size = D // 4
    for i in range(k):
        for j in range(4):
            start_idx = j * block_size
            end_idx = min((j + 1) * block_size, D)
            # Add strong correlation within blocks
            Q_init[i, start_idx:end_idx] += torch.randn(1).item() * 1.5

    print(f"Initial norm of P: {compute_norm(P_init):.6f}")
    print(f"Initial norm of Q: {compute_norm(Q_init):.6f}")
    print(f"Initial norm ratio (Q/P): {compute_norm(Q_init)/compute_norm(P_init):.2f}x")

    # Check initial CKA similarity
    with torch.no_grad():
        initial_cka = linear_cka(P_init, Q_init, detach_norm=False).item()
        initial_cosine = cosine_similarity(P_init, Q_init, detach_norm=False).item()
    print(f"Initial CKA similarity: {initial_cka:.6f}")
    print(f"Initial Cosine similarity: {initial_cosine:.6f}")

    # Compute initial prototype
    prototype_init = (P_init + Q_init) / 2
    print(f"Initial prototype norm: {compute_norm(prototype_init):.6f}")

    # Training parameters
    num_epochs = 3000
    lr = 10

    # Define five methods
    methods = [
        ('cosine', '1-cosine alignment to prototype'),
        ('cosine_detach', '1-cosine alignment to prototype (detach denominator)'),
        ('cka', '1-CKA alignment to prototype'),
        ('cka_detach', '1-CKA alignment to prototype (detach denominator)'),
        ('mse', 'MSE alignment to prototype')
    ]

    # Train and evaluate each method
    for method, method_name in methods:
        print(f"\n\nStarting training: {method_name}")
        print("-" * 60)

        # Train
        P_final, Q_final = train_alignment_to_prototype(
            P_init, Q_init,
            method=method,
            num_epochs=num_epochs,
            lr=lr
        )

        # Print statistics
        print_statistics(P_init, Q_init, P_final, Q_final, method_name)


if __name__ == "__main__":
    main()
