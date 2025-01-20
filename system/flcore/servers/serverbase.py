import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import shutil
import json

from utils.data_utils import read_client_data
from torch.utils.data import DataLoader
from flcore.clients.clientbase import load_item, save_item
from utils.log_utils import set_logger, Logger
import sys


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.top_cnt = 100
        self.auto_break = args.auto_break
        self.role = 'Server'
        if args.save_folder_name == 'temp':
            # args.save_folder_name_full = f'{args.save_folder_name}/{args.dataset}/{args.algorithm}/{time.time()}/'
            args.save_folder_name_full = f'{args.save_folder_name}/{args.dataset}/{args.algorithm}/{args.seed}/{time.time()}/'
        elif 'temp' in args.save_folder_name:
            args.save_folder_name_full = args.save_folder_name
        else:
            args.save_folder_name_full = f'{args.save_folder_name}/{args.dataset}/{args.algorithm}/'
        self.save_folder_name = args.save_folder_name_full

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_test_loss = []
        self.rs_train_acc = []
        self.rs_train_loss = []
        self.best_acc = 0.0
        self.best_epoch = 0
        self.current_epoch = 0

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.tensorboardLogger = None
        self.logger = None

        # obtain caller script
        self.caller_script = os.path.basename(sys.argv[0])
        print(f'Caller: {self.caller_script}')

        if 'visualization.py' in self.caller_script:
            # obtain the classes
            dataset_json_dir = f'../dataset/{args.dataset}/config.json'
            with open(dataset_json_dir, 'r') as f:
                data_config = json.load(f)
            self.args.classes = data_config['classes']

            # obtain tsne classes
            dataset_tsne_json_dir = f'../dataset/{args.dataset}/plot_config.json'
            if os.path.exists(dataset_tsne_json_dir):
                with open(dataset_tsne_json_dir, 'r', encoding='utf-8') as f:
                    data_config = json.load(f)
                self.args.tsne_classes = data_config['TSNE class']
            else:
                self.args.tsne_classes = None

    # set logger and tensorboard
    def set_loggers(self, logFilePath):
        self.tensorboardLogger = Logger(logFilePath)
        self.logger = set_logger(logFilePath + 'textlog.log')

    def log_experiment_results(self, log_file, hyperparameters, results):
        summary_dir = os.path.dirname(log_file)
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        with open(log_file, 'a') as f:
            f.write('Hyperparameters: \n')
            for key, value in hyperparameters.items():
                f.write(str(key) + ' : ' + str(value) + '\n')
            f.write('Results: \n')
            for key, value in results.items():
                f.write(str(key) + ' : ' + str(value) + '\n')
            f.write('='*50 + '\n')


    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_parameters(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters()

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_ids(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)

        client = self.clients[self.uploaded_ids[0]]
        global_model = load_item(client.role, 'model', client.save_folder_name)
        for param in global_model.parameters():
            param.data.zero_()
            
        for w, cid in zip(self.uploaded_weights, self.uploaded_ids):
            client = self.clients[cid]
            client_model = load_item(client.role, 'model', client.save_folder_name)
            for server_param, client_param in zip(global_model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * w

        save_item(global_model, self.role, 'global_model', self.save_folder_name)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            # algo = algo + "_" + self.goal + "_" + str(self.times)
            algo = algo + "_" + self.goal + "_" + str(self.args.seed)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
        
        if 'temp' in self.save_folder_name:
            try:
                shutil.rmtree(self.save_folder_name)
                print('Deleted.')
            except:
                print('Already deleted.')

    def test_metrics(self):        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            print(f'Client {c.id}: Acc: {ct*1.0/ns}, AUC: {auc}')
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)
            print(f'Client {c.id}: Loss: {cl*1.0/ns}')

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if test_acc >= self.best_acc:
            self.best_acc = test_acc
            self.best_epoch = self.current_epoch
        
        # if loss == None:
        #     self.rs_train_loss.append(train_loss)
        # else:
        #     loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        print("Best Epoch: {}".format(self.best_epoch))
        print("Best Test Accurancy: {:.4f}".format(self.best_acc))

        if self.logger is not None:
            self.logger.info("Averaged Train Loss: {:.4f}".format(train_loss))
            self.logger.info("Averaged Test Accurancy: {:.4f}".format(test_acc))
            self.logger.info("Averaged Test AUC: {:.4f}".format(test_auc))
            self.logger.info("Std Test Accurancy: {:.4f}".format(np.std(accs)))
            self.logger.info("Std Test AUC: {:.4f}".format(np.std(aucs)))
            self.logger.info("Best Epoch: {}".format(self.best_epoch))
            self.logger.info("Best Test Accurancy: {:.4f}".format(self.best_acc))

            train_info = {
                'train_loss': train_loss,
            }
            self.tensorboardLogger.add_scalars_dict(prefix='train', dic=train_info, rnd=self.current_epoch)

            test_info = {
                'test_acc': test_acc,
                'best_acc': self.best_acc
            }
            self.tensorboardLogger.add_scalars_dict(prefix='test', dic=test_info, rnd=self.current_epoch)

    # evaluate selected clients on the global dataset
    def evaluate_after_training(self, type='global'):
        test_acc_for_each_client = []
        total_correct = 0
        total_samples = 0
        if type == 'global':
            test_dataset_global = []
            for client_id in range(self.num_clients):
                test_data = read_client_data(self.dataset, client_id, is_train=False)
                test_dataset_global.extend(test_data)
            test_data_loader_global = DataLoader(test_dataset_global, 200, drop_last=False, shuffle=False)

            # evaluate on each client
            for client in self.clients:
                acc_num, total_num, _ = client.evaluation(test_data_loader_global)
                total_correct += acc_num
                total_samples += total_num
                print(acc_num, total_num)
                test_acc_for_each_client.append(acc_num / total_num)
                print(f'client {client.id}, acc {acc_num / total_num}')
                self.logger.info(f'client {client.id}, acc {acc_num / total_num}')
        elif type == 'local':
            # evaluate on each client
            for client in self.clients:
                test_data = read_client_data(self.dataset, client.id, is_train=False)
                test_data_loader_local = DataLoader(test_data, 200, drop_last=False, shuffle=False)
                acc_num, total_num, _ = client.evaluation(test_data_loader_local)
                total_correct += acc_num
                total_samples += total_num
                print(acc_num, total_num)
                test_acc_for_each_client.append(acc_num / total_num)
                print(f'client {client.id}, acc {acc_num / total_num}')
                self.logger.info(f'client {client.id}, acc {acc_num / total_num}')
        else:
            raise NotImplementedError
        print(f'Accuracy for each client: {test_acc_for_each_client}')
        print(f'Average accuracy: {total_correct / total_samples}')
        self.logger.info(f'Accuracy for each client: {test_acc_for_each_client}')
        self.logger.info(f'Average accuracy: {total_correct / total_samples}')

        hyperparameters = {
            'seed': self.args.seed,
            'type': type
        }
        results = {
            'Average Accuracy': total_correct / total_samples
        }
        self.log_experiment_results(self.final_log_path, hyperparameters, results)

    # evaluate top 5 accuracy on the local/global dataset
    def top5_accuracy(self, type='local'):
        test_acc_for_each_client = []
        total_correct = 0
        total_samples = 0
        if type == 'global':
            test_dataset_global = []
            for client_id in range(self.num_clients):
                test_data = read_client_data(self.dataset, client_id, is_train=False)
                test_dataset_global.extend(test_data)
            test_data_loader_global = DataLoader(test_dataset_global, 200, drop_last=False, shuffle=False)

            # evaluate on each client
            for client in self.clients:
                acc_num, total_num, _ = client.top5_accuracy(test_data_loader_global)
                total_correct += acc_num
                total_samples += total_num
                print(acc_num, total_num)
                test_acc_for_each_client.append(acc_num / total_num)
                print(f'client {client.id}, acc {acc_num / total_num}')
                self.logger.info(f'client {client.id}, acc {acc_num / total_num}')
        elif type == 'local':
            # evaluate on each client
            for client in self.clients:
                test_data = read_client_data(self.dataset, client.id, is_train=False)
                test_data_loader_local = DataLoader(test_data, 200, drop_last=False, shuffle=False)
                acc_num, total_num, _ = client.top5_accuracy(test_data_loader_local)
                total_correct += acc_num
                total_samples += total_num
                print(acc_num, total_num)
                test_acc_for_each_client.append(acc_num / total_num)
                print(f'client {client.id}, acc {acc_num / total_num}')
                self.logger.info(f'client {client.id}, acc {acc_num / total_num}')
        else:
            raise NotImplementedError
        print(f'Accuracy for each client: {test_acc_for_each_client}')
        print(f'Average accuracy: {total_correct / total_samples}')
        self.logger.info(f'Accuracy for each client: {test_acc_for_each_client}')
        self.logger.info(f'Average accuracy: {total_correct / total_samples}')

        hyperparameters = {
            'seed': self.args.seed,
            'type': type
        }
        results = {
            'Average Accuracy': total_correct / total_samples
        }
        self.log_experiment_results(self.final_log_path, hyperparameters, results)


    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def load_model(self):
        if not os.path.exists(self.model_save_path):
            raise ValueError(f'No model to load: {self.model_save_path}')

        # load client models
        for c in self.clients:
            c.load_model(save_dir=self.model_save_path)

    def get_global_protos(self):
        raise NotImplementedError

    def visualize_global_protos_superclass_similarity(self):
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        zh_font = fm.FontProperties(fname="C:/Windows/Fonts/simhei.ttf")  # Windows 示例路径
        import seaborn as sns
        import matplotlib
        matplotlib.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'Times New Roman'
        self.load_model()

        # obtain tsne classes
        dataset_tsne_json_dir = f'../dataset/{self.args.dataset}/plot_config.json'
        if os.path.exists(dataset_tsne_json_dir):
            with open(dataset_tsne_json_dir, 'r', encoding='utf-8') as f:
                data_config = json.load(f)
            tsne_super_classes = data_config['superclass']
            tsne_super_class_names = data_config['superclassname']
            print(tsne_super_classes)
            print(tsne_super_class_names)
        else:
            raise NotImplementedError

        global_protos = self.get_global_protos()
        global_protos = global_protos / global_protos.norm(dim=-1, keepdim=True)

        # # select partial class to visualize
        # if self.args.tsne_classes is not None:
        #     plot_proto = []
        #     for plot_class in self.args.tsne_classes:
        #         for idx, class_name in enumerate(self.args.classes):
        #             if class_name == plot_class:
        #                 plot_proto.append(global_protos[idx])
        #                 break
        #     global_protos = torch.stack(plot_proto)

        if self.args.similarity_mode == 'cosine':
            similarity_matrix = global_protos @ global_protos.T
        elif self.args.similarity_mode == 'euclidean':
            similarity_matrix = torch.cdist(global_protos, global_protos, p=2)
        else:
            raise ValueError(f'Invalid similarity mode: {self.args.similarity_mode}')

        if self.args.scaler == 'minmax':
            mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
            non_diag_elements = similarity_matrix[mask]
            min_val = non_diag_elements.min()
            max_val = non_diag_elements.max()
            normalized_values = (non_diag_elements - min_val) / (max_val - min_val)
            similarity_matrix[mask] = normalized_values

        superclassnumber = len(tsne_super_classes)
        superclasssimilarity = torch.zeros(superclassnumber, superclassnumber)
        count = torch.zeros(superclassnumber, superclassnumber)
        for i in range(self.args.num_classes):
            classname_i = self.args.classes[i]
            superclassname_i = None
            # obtain superclass of i
            for k in range(superclassnumber):
                if classname_i in tsne_super_classes[str(k)]:
                    superclassname_i = k
                    break
            print(f'classname:{classname_i}, superclassname:{superclassname_i}')
            for j in range(self.args.num_classes):
                if j == i: continue
                # obtain superclass of j
                classname_j = self.args.classes[j]
                superclassname_j = None
                for k in range(superclassnumber):
                    if classname_j in tsne_super_classes[str(k)]:
                        superclassname_j = k
                        break
                print(f'{superclassname_i}, {superclassname_j}, {similarity_matrix[i][j]}')
                superclasssimilarity[superclassname_i][superclassname_j] += similarity_matrix[i][j].cpu()
                count[superclassname_i][superclassname_j] += 1

        superclasssimilarity = superclasssimilarity / count
        tsne_super_class_names = [tsne_super_class_names[str(i)] for i in range(superclassnumber)]

        # Plot heatmap with values
        plt.figure(figsize=(10, 8))
        if self.args.similarity_mode == 'cosine':
            sns.heatmap(superclasssimilarity, annot=True, fmt=".2f", cmap='Blues', cbar=True,
                        xticklabels=tsne_super_class_names, yticklabels=tsne_super_class_names, vmin=0, vmax=1, annot_kws={"size": 16})
        else:
            sns.heatmap(superclasssimilarity, annot=True, fmt=".2f", cmap='Blues', cbar=True,
                        xticklabels=tsne_super_class_names, yticklabels=tsne_super_class_names)
        # plt.title('Global Prototype Similarity Heatmap')
        plt.xlabel('Prototypes', fontsize=18)
        plt.ylabel('Prototypes', fontsize=18)
        plt.xticks(fontproperties=zh_font,rotation=45, ha='right', fontsize=16)  # Rotate x-axis labels for better readability
        plt.yticks(fontproperties=zh_font,rotation=0, fontsize=16)
        plt.tight_layout()  # Adjust layout to prevent label cutoff
        plt.show()
        plt.close()

        # Compute the mean of diagonal and non-diagonal elements
        diagonal_mean = torch.diag(superclasssimilarity).mean().item()
        non_diagonal_mask = ~torch.eye(superclasssimilarity.size(0), dtype=bool)
        non_diagonal_mean = superclasssimilarity[non_diagonal_mask].mean().item()

        # Plot a bar chart for diagonal and non-diagonal means
        plt.figure(figsize=(4, 4))
        bars = plt.bar(['Diagonal', 'Non-Diagonal'], [diagonal_mean, non_diagonal_mean], color=['deepskyblue', 'darkorange'])
        # Add values on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom',
                     fontsize=14)
        plt.ylabel('Mean Similarity', fontsize=18)
        # plt.title('Mean Similarity: Diagonal vs Non-Diagonal', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()  # Adjust layout for better visual appearance
        plt.show()




    def visualize_global_prototype_similarity(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import matplotlib
        matplotlib.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'Times New Roman'
        self.load_model()

        global_protos = self.get_global_protos()
        global_protos = global_protos / global_protos.norm(dim=-1, keepdim=True)

        # select partial class to visualize
        if self.args.tsne_classes is not None:
            plot_proto = []
            for plot_class in self.args.tsne_classes:
                for idx, class_name in enumerate(self.args.classes):
                    if class_name == plot_class:
                        plot_proto.append(global_protos[idx])
                        break
            global_protos = torch.stack(plot_proto)

        if self.args.similarity_mode == 'cosine':
            similarity_matrix = global_protos @ global_protos.T
        elif self.args.similarity_mode == 'euclidean':
            similarity_matrix = torch.cdist(global_protos, global_protos, p=2)
        else:
            raise ValueError(f'Invalid similarity mode: {self.args.similarity_mode}')

        # Convert to numpy for visualization
        similarity_matrix_np = similarity_matrix.detach().cpu().numpy()
        np.fill_diagonal(similarity_matrix_np, np.nan)
        if self.args.scaler == 'minmax':
            mask = ~np.eye(similarity_matrix_np.shape[0], dtype=bool)
            non_diag_elements = similarity_matrix_np[mask]
            min_val = non_diag_elements.min()
            max_val = non_diag_elements.max()
            normalized_values = (non_diag_elements - min_val) / (max_val - min_val)
            similarity_matrix_np[mask] = normalized_values
        print(similarity_matrix_np)

        print(self.args)
        # Get class names from self.args.classes
        class_names = self.args.tsne_classes if hasattr(self.args, 'tsne_classes') else [f'Class {i}' for i in
                                                                               range(similarity_matrix_np.shape[0])]

        # Plot heatmap with values
        plt.figure(figsize=(10, 8))
        if self.args.similarity_mode == 'cosine':
            sns.heatmap(similarity_matrix_np, annot=True, fmt=".2f", cmap='Blues', cbar=True,
                        xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1, annot_kws={"size": 16})
        else:
            sns.heatmap(similarity_matrix_np, annot=True, fmt=".2f", cmap='Blues', cbar=True,
                        xticklabels=class_names, yticklabels=class_names)
        # plt.title('Global Prototype Similarity Heatmap')
        plt.xlabel('Prototypes', fontsize=18)
        plt.ylabel('Prototypes', fontsize=18)
        plt.xticks(rotation=45, ha='right', fontsize=16)  # Rotate x-axis labels for better readability
        plt.yticks(rotation=0, fontsize=16)
        plt.tight_layout()  # Adjust layout to prevent label cutoff
        plt.savefig(f'{self.args.algorithm}.png', dpi=480)
        plt.show()

        return similarity_matrix
