#!/bin/bash

# 从命令行解析参数
dataset=$1
num_classes=$2
noniid=$3
alpha=$4
num_clients=$5
model_family=$6
batch_size=${7:-100}
device_id=${8:-0}
global_rounds=${9:-300}
switch_alpha=${10:-1}
switch_round=${11:-(-1)}

echo "参数列表:"
echo "dataset: ${dataset}"
echo "num_classes: ${num_classes}"
echo "noniid: ${noniid}"
echo "alpha: ${alpha}"
echo "num_clients: ${num_clients}"
echo "model_family: ${model_family}"
echo "batch_size: ${batch_size}"
echo "device_id: ${device_id}"
echo "global_rounds: ${global_rounds}"

if [ -z "$dataset" ] || [ -z "$noniid" ] || [ -z "$alpha" ] || [ -z "$num_clients" ] || [ -z "$model_family" ]; then
  echo "请提供 dataset noniid alpha, num_clients, model_family 等参数"
  exit 1
fi

# 动态生成 dataset 参数
origin_dataset="${dataset}_${noniid}_${alpha}_balance_${num_clients}"
switch_dataset="${dataset}_${noniid}_${switch_alpha}_balance_${num_clients}"

for seed in 0;
do
    python main.py --dataset=${origin_dataset} \
        --num_classes=${num_classes} \
        --model_family=${model_family} \
        --local_learning_rate=0.01 \
        --global_rounds=${global_rounds} \
        --algorithm=FedTGP_CVPR26 \
        --local_epochs=5 \
        --batch_size=${batch_size} \
        --num_clients=${num_clients} \
        --lamda=1 \
        --seed=${seed} \
        --device_id=${device_id} \
        --save_model=0 \
        --switch_dataset=${switch_dataset} \
        --switch_round=${switch_round}
done

