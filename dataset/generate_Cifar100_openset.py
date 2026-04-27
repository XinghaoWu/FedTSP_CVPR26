# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang
#
# Open-set split for CIFAR-100:
# Keep original noniid split logic (dir/pat) inside each client group,
# but restrict class set per group:
#   - first `client_ratio` clients see only first `class_ratio` classes
#   - remaining clients see only remaining classes
#
# This script is intentionally minimal and reuses dataset/utils/dataset_utils.py.

import argparse
import os
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from utils.dataset_utils import check, separate_data, split_data, save_file


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--niid', type=str, default="noniid", help="non-iid distribution")
    parser.add_argument('--balance', type=str, default="balance", help="balance data size per client")
    parser.add_argument('--partition', type=str, default="dir", help="partition distribution, dir|pat")
    parser.add_argument('--num_users', type=int, default=40, help="number of users")
    parser.add_argument('--alpha', type=float, default=1.0, help="the degree of imbalance")
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--class_ratio', type=float, default=0.8, help="ratio of classes in first group")
    parser.add_argument('--client_ratio', type=float, default=0.8, help="ratio of clients in first group")
    args = parser.parse_args()
    args.alpha = args.alpha if args.partition == 'dir' else int(args.alpha)
    return args


def _openset_group_indices(num_clients, client_ratio):
    n_major_clients = int(round(num_clients * client_ratio))
    n_major_clients = min(max(n_major_clients, 1), num_clients - 1)
    major_clients = list(range(n_major_clients))
    minor_clients = list(range(n_major_clients, num_clients))
    return major_clients, minor_clients


def _openset_class_ranges(num_classes, class_ratio):
    n_major_classes = int(round(num_classes * class_ratio))
    n_major_classes = min(max(n_major_classes, 1), num_classes - 1)
    major_classes = list(range(n_major_classes))
    minor_classes = list(range(n_major_classes, num_classes))
    return major_classes, minor_classes


def _remap_labels_to_compact(y, selected_classes):
    """
    Map labels in `selected_classes` to 0..K-1 (required by separate_data which assumes labels are 0..K-1).
    """
    selected_classes = list(selected_classes)
    mapping = {c: i for i, c in enumerate(selected_classes)}
    y_new = np.array([mapping[int(v)] for v in y], dtype=np.int64)
    return y_new, mapping


def _remap_labels_back(y_compact, mapping):
    inv = {v: k for k, v in mapping.items()}
    return np.array([inv[int(v)] for v in y_compact], dtype=np.int64)


def _concat_clients_to_global(X_group, y_group, client_ids):
    X_all = np.concatenate([X_group[cid] for cid in client_ids], axis=0) if client_ids else np.array([])
    y_all = np.concatenate([y_group[cid] for cid in client_ids], axis=0) if client_ids else np.array([], dtype=np.int64)
    return X_all, y_all


def _split_group_with_original_logic(dataset_image, dataset_label, client_ids, class_ids, niid, balance, partition, alpha, seed):
    """
    Run original separate_data/split_data for a subset of clients and subset of classes.
    Returns dicts keyed by original client id.
    """
    if len(client_ids) == 0:
        return {}, {}, {}

    # Filter samples by class subset
    mask = np.isin(dataset_label, np.array(class_ids))
    X_sub = dataset_image[mask]
    y_sub = dataset_label[mask]

    # Remap labels to 0..K-1 for dataset_utils.separate_data()
    y_compact, mapping = _remap_labels_to_compact(y_sub, class_ids)

    # Use original logic to split among |client_ids| clients, with K=|class_ids|
    np.random.seed(seed)
    X_splits, y_splits, statistic = separate_data(
        (X_sub, y_compact),
        num_clients=len(client_ids),
        num_classes=len(class_ids),
        niid=niid,
        balance=balance,
        partition=partition,
        alpha=alpha,
    )

    # Split train/test using original helper
    train_data, test_data = split_data(X_splits, y_splits)

    # Map back labels to original CIFAR-100 ids and assign to original client ids
    train_by_client = {}
    test_by_client = {}
    stat_by_client = {}
    for local_idx, orig_cid in enumerate(client_ids):
        train_by_client[orig_cid] = {
            'x': train_data[local_idx]['x'],
            'y': _remap_labels_back(train_data[local_idx]['y'], mapping),
        }
        test_by_client[orig_cid] = {
            'x': test_data[local_idx]['x'],
            'y': _remap_labels_back(test_data[local_idx]['y'], mapping),
        }

        # Convert statistic back to original label ids
        stat = []
        for compact_label, cnt in statistic[local_idx]:
            stat.append((int(_remap_labels_back(np.array([compact_label]), mapping)[0]), int(cnt)))
        stat_by_client[orig_cid] = stat

    return train_by_client, test_by_client, stat_by_client


def generate_dataset(niid, balance, partition, args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    num_clients = args.num_users

    dir_path = (
        f"Cifar100_openset_{args.partition}_{args.alpha}_{args.balance}_{args.num_users}"
        f"_c{args.class_ratio}_u{args.client_ratio}_seed{args.seed}/"
    )

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR100(
        root="rawdata/Cifar100", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR100(
        root="rawdata/Cifar100", train=False, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

    classes = trainset.classes

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []
    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    major_clients, minor_clients = _openset_group_indices(num_clients, args.client_ratio)
    major_classes, minor_classes = _openset_class_ranges(num_classes, args.class_ratio)

    # Run original noniid split inside each group, restricted to its class subset
    train_major, test_major, stat_major = _split_group_with_original_logic(
        dataset_image,
        dataset_label,
        client_ids=major_clients,
        class_ids=major_classes,
        niid=niid,
        balance=balance,
        partition=partition,
        alpha=args.alpha,
        seed=args.seed,
    )
    train_minor, test_minor, stat_minor = _split_group_with_original_logic(
        dataset_image,
        dataset_label,
        client_ids=minor_clients,
        class_ids=minor_classes,
        niid=niid,
        balance=balance,
        partition=partition,
        alpha=args.alpha,
        seed=args.seed + 1,  # decouple randomness between groups
    )

    # Merge into full client list
    train_data = []
    test_data = []
    statistic = []
    for cid in range(num_clients):
        if cid in train_major:
            train_data.append(train_major[cid])
            test_data.append(test_major[cid])
            statistic.append(stat_major[cid])
        else:
            train_data.append(train_minor[cid])
            test_data.append(test_minor[cid])
            statistic.append(stat_minor[cid])

    save_file(
        config_path,
        train_path,
        test_path,
        train_data,
        test_data,
        num_clients,
        num_classes,
        statistic,
        niid,
        balance,
        partition=f"{partition}_openset",
        alpha=args.alpha,
        classes=classes,
    )


if __name__ == "__main__":
    args = args_parser()
    niid = True if args.niid == "noniid" else False
    balance = True if args.balance == "balance" else False
    partition = args.partition if args.partition != "-" else None
    generate_dataset(niid, balance, partition, args)
