import time
import numpy as np
from flcore.clients.clientprotocka import clientProtoCKA
from flcore.servers.serverprotoeval import FedProtoEval
from flcore.servers.serverprotoeval import compute_prototype_similarity_missing_safe, compute_prototype_similarity_procrustes, procrustes_align, cosine_matrix, upper_tri, centered_gram, get_shared_classes, protos_to_matrix, compute_similarity_averages, compute_cosine_distance_matrix, compute_mse_distance_matrix, plot_similarity_heatmap, proto_aggregation, procrustes_coord_metrics, compute_coord_metric_matrices, compute_hom_het_stats
from flcore.clients.clientbase import load_item, save_item
from utils.data_utils import read_client_data
from threading import Thread
from collections import defaultdict
import os, copy
import torch
from scipy.stats import pearsonr, spearmanr


class FedProtoCKA(FedProtoEval):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientProtoCKA)


    