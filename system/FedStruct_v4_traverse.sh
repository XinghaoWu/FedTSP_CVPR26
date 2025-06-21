#!/bin/bash

# 固定参数
dataset=Cifar10
num_classes=10
noniid=dir
num_clients=20
model_family=HtFE9
batch_size=100
global_rounds=100

# 卡配置
num_gpus=4
max_tasks_per_gpu=6
total_parallel_tasks=$((num_gpus * max_tasks_per_gpu))

# 超参数列表
lamda_list=(0 0.01 0.1 1)
gamma_list=(0 0.01 0.1 1)

# 任务索引
task_idx=0
pids=()

# 遍历参数组合
for alpha in 0.5 1.0; do
    for rotation in 0 1; do
        for lamda in "${lamda_list[@]}"; do
            for gamma in "${gamma_list[@]}"; do

            # 自动分配 GPU ID
            device_id=$((task_idx % total_parallel_tasks / max_tasks_per_gpu))

            echo "正在运行: alpha=$alpha, rotation=$rotation, lamda=$lamda, gamma=$gamma, device_id=$device_id"

            bash SOTA_FedStruct_v4.sh "$dataset" "$num_classes" "$noniid" "$alpha" "$num_clients" "$model_family" "$batch_size" "$device_id" "$global_rounds" "$rotation" "$lamda" "$gamma" &

            pids+=($!)
            ((task_idx++))

            # 控制并发：每 batch 最多 total_parallel_tasks 个任务
            if (( ${#pids[@]} >= total_parallel_tasks )); then
                wait "${pids[@]}"
                pids=()
            fi
            done
        done
    done
done

# 等待最后一批任务完成
wait "${pids[@]}"
echo "所有任务已完成 ✅"
