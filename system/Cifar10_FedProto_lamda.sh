#!/bin/bash

# 从命令行解析参数
noniid=$1
alpha=$2
num_clients=$3
model_family=$4
batch_size=${5:-100}
seed=${6:-0}
device_id=${7:-0}

echo "参数列表:"
echo "alpha: ${alpha}"
echo "num_clients: ${num_clients}"
echo "model_family: ${model_family}"
echo "batch_size: ${batch_size}"
echo "seed: ${seed}"
echo "device_id: ${device_id}"

if [ -z "$noniid" ] || [ -z "$alpha" ] || [ -z "$num_clients" ] || [ -z "$model_family" ]; then
  echo "请提供 noniid alpha, num_clients, model_family, lamda 和 seed 参数，例如："
  echo "./script.sh dir 0.1 20 HtFE2 7 100 0 0"
  exit 1
fi

# 动态生成 dataset 参数
dataset="Cifar10_${noniid}_${alpha}_balance_${num_clients}"

for lamda in 1 2 3 4 5;
do
    python main.py --dataset=${dataset} \
        --num_classes=10 \
        --model_family=${model_family} \
        --local_learning_rate=0.01 \
        --global_rounds=300 \
        --algorithm=FedProto \
        --local_epochs=5 \
        --batch_size=${batch_size} \
        --num_clients=${num_clients} \
        --lamda=${lamda} \
        --seed=${seed} \
        --device_id=${device_id} \
        --save_model=0
done

