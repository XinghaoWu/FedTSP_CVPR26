# PFLlib: Personalized Federated Learning Algorithm Library
# Based on the interface of generate_DomainNet.py

import time
import numpy as np
import os
import random
import torchvision.transforms as transforms
from utils.dataset_utils import split_data, save_file
from os import path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import glob

# PACS 数据集的 7 个类别，固定顺序以保证跨域 Label 对齐
CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def read_pacs_data(dataset_path, domain_name, split="all"):
    """
    PACS 数据集通常比较小 (约10k图片)，且通常以文件夹形式直接提供。
    这里直接遍历文件夹读取路径。
    Structure: dataset_path/domain_name/class_name/image.jpg
    """
    data_paths = []
    data_labels = []
    
    domain_dir = path.join(dataset_path, "kfold", domain_name) 
    # 注意：有的 PACS 下载解压后直接是 domain_name，有的是在 kfold 文件夹下
    # 如果您的解压结构不同，请修改上面的路径，例如: domain_dir = path.join(dataset_path, domain_name)
    if not os.path.exists(domain_dir):
         domain_dir = path.join(dataset_path, domain_name)

    # 遍历该 Domain 下的所有类别
    for cls_name in CLASSES:
        cls_folder = path.join(domain_dir, cls_name)
        if not os.path.exists(cls_folder):
            continue
            
        # 获取该类别下所有图片
        img_names = os.listdir(cls_folder)
        img_names.sort() # 排序保证确定性
        
        for img_name in img_names:
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = path.join(cls_folder, img_name)
                label = CLASS_TO_IDX[cls_name]
                data_paths.append(img_path)
                data_labels.append(label)

    # 简单模拟 Train/Test split (为了保持接口一致)
    # 既然主函数最后会合并它们再重新 split，这里简单的按 8:2 切分即可
    # 或者根据 split 参数返回对应的切片
    total_len = len(data_paths)
    indices = list(range(total_len))
    # 为了保证复现性，这里使用局部随机
    r = random.Random(42)
    r.shuffle(indices)
    
    split_point = int(total_len * 0.8)
    
    if split == "train":
        selected_indices = indices[:split_point]
    elif split == "test":
        selected_indices = indices[split_point:]
    else:
        selected_indices = indices

    return [data_paths[i] for i in selected_indices], [data_labels[i] for i in selected_indices]


class PACSDataset(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name):
        super(PACSDataset, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.domain_name = domain_name

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index]
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)


def get_pacs_dloader(dataset_path, domain_name):
    # 读取数据
    train_data_paths, train_data_labels = read_pacs_data(dataset_path, domain_name, split="train")
    test_data_paths, test_data_labels = read_pacs_data(dataset_path, domain_name, split="test")
    
    # 保持与 DomainNet 相同的 Transform 逻辑，统一 Resize 到 64x64
    # 如果您做 ResNet 等实验，通常 PACS 原始大小是 224x224，这里为了适配接口缩小了
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_dataset = PACSDataset(train_data_paths, train_data_labels, transforms_train, domain_name)
    train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
    
    test_dataset = PACSDataset(test_data_paths, test_data_labels, transforms_test, domain_name)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    return train_loader, test_loader


random.seed(1)
np.random.seed(1)
data_path = "rawdata/PACS"

# Allocate data to users
def generate_dataset():
    # PACS 的 4 个 Domain
    domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    num_clients = len(domains)
    dir_path = f"PACS_{num_clients}/"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    root = data_path
    
    # PACS 下载链接 (常用的学术镜像)
    # 注意：PACS 官方通常是 Google Drive 链接，wget 很难直接下载。
    # 这里使用一个常用的公开直接链接，如果失效，请手动下载 PACS.zip 并解压到 rawdata/PACS
    pacs_url = "http://cwl.uni-jena.de/data/DATASETS/pacs/PACS_hw.zip" 
    
    # Get PACS data
    if not os.path.exists(root):
        os.makedirs(root)
        print("Downloading PACS dataset... (This might take a while)")
        os.system(f'wget {pacs_url} -P rawdata/')
        # 解压
        print("Unzipping PACS dataset...")
        os.system(f'unzip rawdata/PACS_hw.zip -d rawdata/')
        # 重命名文件夹以匹配路径 logic (如果是 PACS_hw.zip 通常解压出来叫 kfold 或 PACS)
        if os.path.exists("rawdata/kfold") and not os.path.exists(root):
             os.rename("rawdata/kfold", root) # 将 kfold 重命名为 PACS
        elif os.path.exists("rawdata/PACS_hw") and not os.path.exists(root):
             os.rename("rawdata/PACS_hw", root)
        
        # 清理 zip
        # os.system('rm rawdata/PACS_hw.zip')

    X, y = [], []
    for d in domains:
        print(f"Processing Domain: {d}")
        train_loader, test_loader = get_pacs_dloader(root, d)

        # 检查是否成功加载数据
        if len(train_loader.dataset) == 0:
            print(f"Warning: No data found for domain {d} at {root}. Check directory structure.")
            continue

        for _, tt in enumerate(train_loader):
            train_data, train_label = tt
        for _, tt in enumerate(test_loader):
            test_data, test_label = tt

        dataset_image = []
        dataset_label = []

        dataset_image.extend(train_data.cpu().detach().numpy())
        dataset_image.extend(test_data.cpu().detach().numpy())
        dataset_label.extend(train_label.cpu().detach().numpy())
        dataset_label.extend(test_label.cpu().detach().numpy())

        X.append(np.array(dataset_image))
        y.append(np.array(dataset_label))

    labelss = []
    for yy in y:
        labelss.append(len(set(yy)))
    print(f'Number of labels per client: {labelss}')
    print(f'Number of clients: {num_clients}')

    statistic = [[] for _ in range(num_clients)]
    for client in range(num_clients):
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))

    # 这里调用 utils.dataset_utils 中的 split_data
    # 该函数通常会将 X, y (也就是每个 Client 的全量数据) 按照一定比例切分为 Train/Test
    # 这里的 Train/Test 是指 FL 训练中的本地训练集和测试集
    train_data, test_data = split_data(X, y)
    
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, 7, 
        statistic, None, None, None)
    
    print("PACS dataset generation complete.")


if __name__ == "__main__":
    generate_dataset()