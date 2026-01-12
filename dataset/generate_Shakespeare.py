import json
import os
import numpy as np
import random
from utils.language_utils import word_to_indices, letter_to_index


random.seed(1)
np.random.seed(1)
# You need to download LEAF project first, see
data_path_train = "utils/LEAF/data/shakespeare/data/train/all_data_niid_02_keep_0_train_9.json"
data_path_test = "utils/LEAF/data/shakespeare/data/test/all_data_niid_02_keep_0_test_9.json"
dir_path = "Shakespeare_0.02_118/"

# https://github.com/TalwalkarLab/leaf/blob/0d30b4d18c36551ee54e0076915b0c49c5dd9cd6/models/shakespeare/stacked_lstm.py#L40
def process_x(raw_x_batch):
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch

def process_y(raw_y_batch):
    y_batch = [letter_to_index(c) for c in raw_y_batch]
    y_batch = np.array(y_batch)
    return y_batch

# Allocate data to users
def generate_dataset(dir_path):
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(data_path_train) as f:
        raw_train_data = json.load(f)
    with open(data_path_test) as f:
        raw_test_data = json.load(f)

    train_data_ = []
    train_data_len = []
    test_data_ = []

    for k, v in raw_train_data['user_data'].items():
        train_data_.append({'x': process_x(v['x']), 'y': process_y(v['y'])})
        train_data_len.append(len(train_data_[-1]['x']))
    for k, v in raw_test_data['user_data'].items():
        test_data_.append({'x': process_x(v['x']), 'y': process_y(v['y'])})

    train_data = []
    test_data = []

    inds = sorted(range(len(train_data_len)), key=lambda k: train_data_len[k])
    for ind in inds:
        train_data.append(train_data_[ind])
        test_data.append(test_data_[ind])
        
    print("Saving to disk.\n")

    # for idx, train_dict in enumerate(train_data):
    #     with open(train_path + str(idx) + '.npz', 'wb') as f:
    #         np.savez_compressed(f, data=train_dict)
    # for idx, test_dict in enumerate(test_data):
    #     with open(test_path + str(idx) + '.npz', 'wb') as f:
    #         np.savez_compressed(f, data=test_dict)

    for idx, train_dict in enumerate(train_data):
        # 修改X格式：从 (n_samples, seq_len) 改为 (n_samples, 2)，每个样本包含 [序列, 长度]
        x_data = train_dict['x']
        n_samples = len(x_data)
        seq_len = len(x_data[0]) if n_samples > 0 else 80

        # 创建新的X格式：每个样本是 [序列(list), 长度(int)]
        x_formatted = np.array([[list(x_data[i]), seq_len] for i in range(n_samples)], dtype=object)

        formatted_dict = {'x': x_formatted, 'y': train_dict['y']}
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=formatted_dict)

    for idx, test_dict in enumerate(test_data):
        # 修改X格式：从 (n_samples, seq_len) 改为 (n_samples, 2)，每个样本包含 [序列, 长度]
        x_data = test_dict['x']
        n_samples = len(x_data)
        seq_len = len(x_data[0]) if n_samples > 0 else 80

        # 创建新的X格式：每个样本是 [序列(list), 长度(int)]
        x_formatted = np.array([[list(x_data[i]), seq_len] for i in range(n_samples)], dtype=object)

        formatted_dict = {'x': x_formatted, 'y': test_dict['y']}
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=formatted_dict)

    print("Finish generating dataset.\n")

if __name__ == "__main__":
    generate_dataset(dir_path)