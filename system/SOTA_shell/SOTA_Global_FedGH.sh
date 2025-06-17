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
visualization_mode=${10:-test}
visualization_dataset_type=${11:-test}
test_data_mode=${12:-global}

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
dataset="${dataset}_${noniid}_${alpha}_balance_${num_clients}"

for seed in 0 1 2;
do
    python visualization.py --dataset=${dataset} \
        --num_classes=${num_classes} \
        --model_family=${model_family} \
        --local_learning_rate=0.01 \
        --global_rounds=${global_rounds} \
        --algorithm=FedGH \
        --local_epochs=5 \
        --batch_size=${batch_size} \
        --num_clients=${num_clients} \
        --server_learning_rate=0.01 \
        --seed=${seed} \
        --device_id=${device_id} \
        --save_model=0 \
        --visualization_mode=${visualization_mode} \
        --visualization_dataset_type=${visualization_dataset_type} \
        --test_data_mode=${test_data_mode}
done

