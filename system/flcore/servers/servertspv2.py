import time
import numpy as np
from flcore.clients.clienttspv2 import clientTSPv2
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from utils.data_utils import read_client_data
from threading import Thread
from collections import defaultdict
import json
import copy
from tqdm import tqdm
import clip
from flcore.trainmodel.clip_base import TextEncoder_server
import torch
import torch.nn.functional as F

'''
The main difference between FedTSPv2 and FedOurs is FedTSPv2 set text encoder on the server
'''
class FedTSPv2(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # set logger
        logger_path = (f'../logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_'
                       f'ep{args.local_epochs}_nc{args.num_clients}/lr({args.local_learning_rate}_{args.prompt_lr})_'
                       f'lamda(t{args.lamda}_v{args.vision_proto})_prompt(CSC{args.CSC}_len{args.len_prompt}_random{args.prompt_random_init})_'
                       f'EMA{args.EMA_alpha}_EMAp{args.prompt_EMA_alpha}_promptep{args.prompt_epoch}_gap{args.server_training_freq}_pcls{args.p_classifier}/')
        self.set_loggers(logger_path)
        self.args.logger = self.logger
        self.args.tensorboardLogger = self.tensorboardLogger

        # obtain the classes
        dataset_json_dir = f'../dataset/{args.dataset}/config.json'
        with open(dataset_json_dir, 'r') as f:
            data_config = json.load(f)
        self.args.classes = data_config['classes']

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientTSPv2)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes

        # load server model
        clip_model, _ = clip.load('ViT-B/32', device=torch.device("cpu"))
        clip_model.to(self.device)
        self.global_model = TextEncoder_server(self.args.classes, clip_model, self.args.len_prompt, self.args.CSC, self.args.prompt_random_init).to(self.device)
        for name, param in self.global_model.named_parameters():
            if 'prompt_learner' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        self.prompt_lr = args.prompt_lr

        # for vision prototype alignment
        self.global_vision_protos = None

        # for text prorotype alignment
        self.global_text_protos = None

        self.global_classifier = None

    def train(self):
        for i in tqdm(range(self.global_rounds + 1)):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.logger.info(f"\n-------------Round number: {i}-------------")
                self.logger.info("\nEvaluate heterogeneous models")
                self.epoch = i
                self.evaluate()

            # self.global_text_protos = self.global_model.get_text_prototypes()
            new_global_text_protos = self.global_model.get_text_prototypes()
            if self.global_text_protos is None:
                self.global_text_protos = new_global_text_protos
            else:
                self.global_text_protos = self.args.EMA_alpha * self.global_text_protos + (1 - self.args.EMA_alpha) * new_global_text_protos
            self.send_parameters()

            print(f'Round {i}, local training starts.')
            self.logger.info(f'Round {i}, local training starts.')

            # client training
            for client in self.selected_clients:
                client.train()

            self.receive_ids()

            self.aggregate_parameters()

            # server training to optimize the text prompt
            if self.args.len_prompt > 0 and i % self.args.server_training_freq == 0:
                print(f'Rounds {i}, server training starts.')
                self.logger.info(f'Rounds {i}, server training starts.')
                prompts = self.global_model.prompt_learner
                old_prompts = copy.deepcopy(prompts)
                optimizer_prompts = torch.optim.SGD(prompts.parameters(), lr=self.prompt_lr)
                for j in range(self.args.prompt_epoch):
                    clip_logits = self.global_model(self.global_vision_protos)
                    loss = F.cross_entropy(clip_logits, torch.tensor([i for i in range(self.num_classes)]).to(self.device))
                    optimizer_prompts.zero_grad()
                    loss.backward()
                    optimizer_prompts.step()
                    print(f'Prompt training epoch {j}, loss: {loss.item()}')
                    self.logger.info(f'Prompt training epoch {j}, loss: {loss.item()}')
                for param, old_param in zip(prompts.parameters(), old_prompts.parameters()):
                    param.data = self.args.prompt_EMA_alpha * old_param.data + (1 - self.args.prompt_EMA_alpha) * param.data




            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        self.save_results()

    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)

        # aggregate global classifier
        if self.args.p_classifier == 0:
            client = self.clients[self.uploaded_ids[0]]
            global_classifier = copy.deepcopy(client.model.head)
            for param in global_classifier.parameters():
                param.data.zero_()
            for w, cid in zip(self.uploaded_weights, self.uploaded_ids):
                client = self.clients[cid]
                client_classifier = copy.deepcopy(client.model.head)
                for server_param, client_param in zip(global_classifier.parameters(), client_classifier.parameters()):
                    server_param.data += client_param.data.clone() * w
            self.global_classifier = global_classifier
        else:
            self.global_classifier = None

        # aggregate global vision prototype
        print('Aggregate vision prototypes')
        uploaded_protos = []
        for client in self.selected_clients:
            protos = client.local_vision_proto
            uploaded_protos.append(protos)
        self.global_vision_protos = proto_aggregation(uploaded_protos)

    def send_parameters(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_classifier, self.global_vision_protos, self.global_text_protos)
            client.epoch = self.epoch

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_losses = []
        for c in self.clients:
            correct_num, test_num, total_loss = c.test_metrics()
            tot_correct.append(correct_num)
            # print(f'Client {c.id}: Test Acc: {correct_num * 1.0 / test_num}, Test Loss: {total_loss * 1.0 / test_num}')
            tot_losses.append(total_loss)
            num_samples.append(test_num)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_losses

    def train_metrics(self):
        num_samples = []
        tot_correct = []
        tot_losses = []
        for c in self.clients:
            total_loss, train_num, correct_num = c.train_metrics()
            num_samples.append(train_num)
            tot_losses.append(total_loss)
            # print(f'Client {c.id}: Train Acc: {correct_num * 1.0 / train_num}, Train Loss: {total_loss * 1.0 / train_num}')
            tot_correct.append(correct_num)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_loss = sum(stats[3]) * 1.0 / sum(stats[1])
        test_accs = [a / n for a, n in zip(stats[2], stats[1])]
        test_losses = [a / n for a, n in zip(stats[3], stats[1])]

        if test_acc >= self.best_acc:
            self.best_acc = test_acc
            self.best_epoch = self.epoch

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        train_acc = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        train_loss = sum(stats_train[3])*1.0 / sum(stats_train[1])
        train_accs = [a / n for a, n in zip(stats_train[2], stats_train[1])]
        train_losses = [a / n for a, n in zip(stats_train[3], stats_train[1])]

        # if loss == None:
        #     self.rs_train_loss.append(train_loss)
        # else:
        #     loss.append(train_loss)

        # print("Averaged Train Loss: {:.4f}".format(train_loss))

        print(f'Train acc per client:{train_accs}')
        print(f'Test acc per clients: {test_accs}')

        print("Averaged Train Accurancy: {:.4f}".format(train_acc), end=', ')
        print("Std Train Accurancy: {:.4f}".format(np.std(train_accs)), end=', ')
        print("Averaged Train Loss: {:.4f}".format(train_loss), end=', ')
        print("Std Train Loss: {:.4f}".format(np.std(train_losses)))

        print("Averaged Test Accurancy: {:.4f}".format(test_acc), end=', ')
        print("Std Test Accurancy: {:.4f}".format(np.std(test_accs)), end=', ')
        print("Averaged Test Loss: {:.4f}".format(test_loss), end=', ')
        print("Std Test Loss: {:.4f}".format(np.std(test_losses)))
        print("Best Epoch: ", self.best_epoch)
        print("Best Test Accurancy: {:.4f}".format(self.best_acc))

        self.logger.info(f'Train acc per client:{train_accs}')
        self.logger.info(f'Test acc per clients: {test_accs}')

        self.logger.info("Averaged Train Accurancy: {:.4f}".format(train_acc))
        self.logger.info("Std Train Accurancy: {:.4f}".format(np.std(train_accs)))
        self.logger.info("Averaged Train Loss: {:.4f}".format(train_loss))
        self.logger.info("Std Train Loss: {:.4f}".format(np.std(train_losses)))

        self.logger.info("Averaged Test Accurancy: {:.4f}".format(test_acc))
        self.logger.info("Std Test Accurancy: {:.4f}".format(np.std(test_accs)))
        self.logger.info("Averaged Test Loss: {:.4f}".format(test_loss))
        self.logger.info("Std Test Loss: {:.4f}".format(np.std(test_losses)))
        self.logger.info(f"Best Epoch: {self.best_epoch}")
        self.logger.info("Best Test Accurancy: {:.4f}".format(self.best_acc))

        train_info = {
            'train_acc': train_acc,
            'train_loss': train_loss,
        }
        self.tensorboardLogger.add_scalars_dict(prefix='train', dic=train_info, rnd=self.epoch)

        test_info = {
            'test_acc': test_acc,
            'test_loss': test_loss,
            'best_acc': self.best_acc
        }
        self.tensorboardLogger.add_scalars_dict(prefix='test', dic=test_info, rnd=self.epoch)

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