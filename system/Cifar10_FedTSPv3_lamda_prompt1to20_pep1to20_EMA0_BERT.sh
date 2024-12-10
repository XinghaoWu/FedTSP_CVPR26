#!/bin/bash

# 从命令行解析参数
noniid=$1
alpha=$2
num_clients=$3
model_family=$4
lamda=$5
batch_size=${6:-100}
seed=${7:-0}
device_id=${8:-0}

echo "参数列表:"
echo "alpha: ${alpha}"
echo "num_clients: ${num_clients}"
echo "model_family: ${model_family}"
echo "lamda: ${lamda}"
echo "batch_size: ${batch_size}"
echo "seed: ${seed}"
echo "device_id: ${device_id}"

if [ -z "$noniid" ] || [ -z "$alpha" ] || [ -z "$num_clients" ] || [ -z "$model_family" ] || [ -z "$lamda" ]; then
  echo "请提供 noniid alpha, num_clients, model_family, lamda 和 seed 参数，例如："
  echo "./script.sh dir 0.1 20 HtFE2 7 100 0 0"
  exit 1
fi

# 动态生成 dataset 参数
dataset="Cifar10_${noniid}_${alpha}_balance_${num_clients}"

python main.py --dataset=${dataset} \
          --num_classes=10 \
          --model_family=${model_family} \
          --local_learning_rate=0.01 \
          --global_rounds=100 \
          --algorithm=FedTSPv3 \
          --local_epochs=5 \
          --batch_size=${batch_size} \
          --num_clients=${num_clients} \
          --prompt_lr=0.01 \
          --prompt_epoch=0 \
          --lamda=${lamda} \
          --len_prompt=0 \
          --vision_proto=0 \
          --EMA_alpha=0 \
          --server_model=bert \
          --feature_dim=768 \
          --prompt_random_init \
          --seed=${seed} \
          --device_id=${device_id} \
          --save_model=0 &

for lamda in ${lamda};
do
  for prompt_epoch in 1 5 10 15 20;
  do
    for len_prompt in 1 5 10 15 20;
    do
      for EMA_alpha in 0;
      do
        python main.py --dataset=${dataset} \
          --num_classes=10 \
          --model_family=${model_family} \
          --local_learning_rate=0.01 \
          --global_rounds=100 \
          --algorithm=FedTSPv3 \
          --local_epochs=5 \
          --batch_size=${batch_size} \
          --num_clients=${num_clients} \
          --prompt_lr=0.01 \
          --prompt_epoch=${prompt_epoch} \
          --lamda=${lamda} \
          --len_prompt=${len_prompt} \
          --vision_proto=0 \
          --EMA_alpha=${EMA_alpha} \
          --server_model=bert \
          --feature_dim=768 \
          --prompt_random_init \
          --seed=${seed} \
          --device_id=${device_id} \
          --save_model=0 &

        python main.py --dataset=${dataset} \
          --num_classes=10 \
          --model_family=${model_family} \
          --local_learning_rate=0.01 \
          --global_rounds=100 \
          --algorithm=FedTSPv3 \
          --local_epochs=5 \
          --batch_size=${batch_size} \
          --num_clients=${num_clients} \
          --prompt_lr=0.01 \
          --prompt_epoch=${prompt_epoch} \
          --lamda=${lamda} \
          --len_prompt=${len_prompt} \
          --vision_proto=0 \
          --EMA_alpha=${EMA_alpha} \
          --server_model=bert \
          --feature_dim=768 \
          --seed=${seed} \
          --device_id=${device_id} \
          --save_model=0 &

        wait
      done
    done
  done
done