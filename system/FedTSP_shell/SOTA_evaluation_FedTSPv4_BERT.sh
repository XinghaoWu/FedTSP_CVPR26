#!/bin/bash

# 从命令行解析参数
# 从命令行解析参数
dataset=$1
num_classes=$2
noniid=$3
alpha=$4
num_clients=$5
model_family=$6
lamda=$7
prompt_epoch=$8
len_prompt=$9
batch_size=${10:-100}
device_id=${11:-0}
global_rounds=${12:-60}
visualization_mode=${13:-test}
visualization_dataset_type=${14:-test}
test_data_mode=${15:-global}

echo "参数列表:"
echo "dataset: ${dataset}"
echo "num_classes: ${num_classes}"
echo "noniid: ${noniid}"
echo "alpha: ${alpha}"
echo "num_clients: ${num_clients}"
echo "model_family: ${model_family}"
echo "lamda: ${lamda}"
echo "prompt_epoch: ${prompt_epoch}"
echo "len_prompt: ${len_prompt}"
echo "batch_size: ${batch_size}"
echo "device_id: ${device_id}"
echo "global_rounds: ${global_rounds}"

if [ -z "$noniid" ] || [ -z "$alpha" ] || [ -z "$num_clients" ] || [ -z "$model_family" ] || [ -z "$lamda" ]; then
  echo "请提供 noniid alpha, num_clients, model_family, lamda 和 seed 参数，例如："
  echo "./script.sh dir 0.1 20 HtFE2 7 100 0 0"
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
        --algorithm=FedTSPv4 \
        --local_epochs=5 \
        --batch_size=${batch_size} \
        --num_clients=${num_clients} \
        --prompt_lr=0.01 \
        --prompt_epoch=${prompt_epoch} \
        --lamda=${lamda} \
        --len_prompt=${len_prompt} \
        --vision_proto=0 \
        --EMA_alpha=0 \
        --server_model=bert \
        --feature_dim=768 \
        --seed=${seed} \
        --device_id=${device_id} \
        --manual_prompt=0 \
        --negative_class=0 \
        --save_model=0 \
        --visualization_mode=${visualization_mode} \
        --visualization_dataset_type=${visualization_dataset_type} \
        --test_data_mode=${test_data_mode}
done