#!/bin/bash
# 固定参数
dataset=Cifar10
num_classes=10
noniid=dir
dataset_clients=20
num_clients=4
model_family=HtM4
batch_size=100
global_rounds=100
version=5
# 卡配置
num_gpus=2
max_tasks_per_gpu=5
total_parallel_tasks=$((num_gpus * max_tasks_per_gpu))
# 超参数列表
lamda_list=(0 0.1 1 10 100)
model_familys=("FedAvgCNN" "mobilenet_v2" "resnet18" "vit_b_16")
# gamma_list=(1 5 10 15 20)
# lamda_list=(0)
# gamma_list=(5 10 2 15)
# 任务索引
task_idx=0
pids=()
# 遍历参数组合
for alpha in 10000.0; do
for lamda in "${lamda_list[@]}"; do
for model_family in "${model_familys[@]}"; do
# 自动分配 GPU ID
# device_id=$((task_idx % total_parallel_tasks / max_tasks_per_gpu))
device_id=$((task_idx % num_gpus))
echo "正在运行: alpha=$alpha, model_family=${model_family}, lamda=$lamda, device_id=$device_id"
Dataset="${dataset}_${noniid}_${alpha}_balance_${dataset_clients}"
python main.py --dataset=${Dataset} --num_classes=${num_classes} --model_family=${model_family} --local_learning_rate=0.01 --global_rounds=${global_rounds} --algorithm=FedProtoCos --local_epochs=5 --batch_size=${batch_size} --num_clients=${num_clients} --lamda=${lamda} --seed=0 --device_id=${device_id} --save_model=1 &
pids+=($!)
task_idx=$((task_idx + 1))
# 控制并发：每 batch 最多 total_parallel_tasks 个任务
if (( ${#pids[@]} >= total_parallel_tasks )); then
wait "${pids[@]}"
pids=()
fi
done
done
done
# 等待最后一批任务完成
wait "${pids[@]}"
echo "所有任务已完成 ✅"