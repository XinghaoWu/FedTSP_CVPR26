import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from flcore.trainmodel.models import BaseHeadSplit
import json


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.args = args
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.role = 'Client_' + str(self.id)
        self.save_folder_name = args.save_folder_name_full

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            model = BaseHeadSplit(args, self.id).to(self.device)
            save_item(model, self.role, 'model', self.save_folder_name)

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()

        # used for TSNE visualization
        self.marker = ['o', 'd', 's', 'D', '^', '<', '*', '>', 'v', 'p']


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False, num_workers=self.args.num_workers)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False, num_workers=self.args.num_workers)

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        # model.to(self.device)
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        # model.to(self.device)
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
    
    # used for evaluation on specific dataset
    def evaluation(self, dataloader):
        self.model.to(self.device)
        self.model.eval()
        test_loader = dataloader
        test_acc = 0
        test_num = 0
        losses = 0

        with torch.no_grad():
            for x, y in test_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
                loss = self.loss(output, y)
                losses += loss.item() * y.shape[0]

        self.model.to('cpu')
        return test_acc, test_num, losses

    # obtain top 5 accuracy on specific dataset
    def top5_accuracy(self, dataloader):
        self.model.to(self.device)
        self.model.eval()
        test_loader = dataloader
        test_num = 0
        losses = 0
        top5_acc = 0  # Initialize Top-5 Accuracy counter

        with torch.no_grad():
            for x, y in test_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                # # Top-1 Accuracy
                # test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

                # Top-5 Accuracy
                _, top5_pred = torch.topk(output, k=5, dim=1)  # Get top 5 predictions
                top5_acc += torch.sum(torch.any(top5_pred == y.view(-1, 1), dim=1)).item()

                test_num += y.shape[0]
                loss = self.loss(output, y)
                losses += loss.item() * y.shape[0]

        self.model.to('cpu')
        return top5_acc, test_num, losses
        

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_save_path = os.path.join(save_dir, f'local_model_client_{self.id}.pth')
        torch.save(self.model, model_save_path)

    def load_model(self, save_dir):
        model_save_path = os.path.join(save_dir, f'local_model_client_{self.id}.pth')
        self.model = torch.load(model_save_path).to(self.device)

    def get_features_and_labels(self):
        all_features = []
        all_labels = []

        dataloader = self.load_train_data() if self.args.visualization_dataset_type == 'train' else self.load_test_data()
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            for x, y in dataloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                labels = y.to(self.device)
                features = self.model.base(x)

                all_features.append(features)  # Keep as torch tensors for global norm
                all_labels.append(labels.cpu().numpy())

        # Concatenate features and labels across batches
        all_features = torch.cat(all_features, dim=0)  # Concatenate all batches
        all_labels = np.concatenate(all_labels, axis=0)

        # Normalize all features globally
        all_features = all_features / all_features.norm(dim=-1, keepdim=True)
        all_features = all_features.cpu().numpy()  # Convert to numpy array for t-SNE

        self.model.to('cpu')
        return all_features, all_labels


def save_item(item, role, item_name, item_path=None):
    if not os.path.exists(item_path):
        os.makedirs(item_path)
    torch.save(item, os.path.join(item_path, role + "_" + item_name + ".pt"))

def load_item(role, item_name, item_path=None):
    try:
        return torch.load(os.path.join(item_path, role + "_" + item_name + ".pt"))
    except FileNotFoundError:
        print(role, item_name, 'Not Found')
        return None
