#!/bin/bash

# 命令行参数解析
# 使用方法: ./motivation_metric_experiments.sh [ALGORITHM] [DEVICE_ID]
# 例如: ./motivation_metric_experiments.sh FedProtoCKA 0
#      ./motivation_metric_experiments.sh FedProtoCos 1
if [ -n "$1" ]; then
    ALGORITHM="$1"
else
    ALGORITHM="FedProtoCos"  # 默认值
fi

if [ -n "$2" ]; then
    DEVICE_ID="$2"
else
    DEVICE_ID=0  # 默认值
fi

# 实验配置参数（除lambda外的固定参数）
DATASET="Cifar10_dir_1.0_balance_20"
NUM_CLASSES=10
MODEL_FAMILY="HtFE9"
LOCAL_LR=0.01
GLOBAL_ROUNDS=200
LOCAL_EPOCH=5
BATCH_SIZE=100
NUM_CLIENTS=9
SEED=0
TAG="detach"
VISUALIZATION_MODE="testing_feature_difference"

# Lambda值数组（可以根据需要修改）
LAMBDA_VALUES=(0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0 5.0 10.0 20.0)

echo "========================================="
echo "Experiment Configuration:"
echo "ALGORITHM: ${ALGORITHM}"
echo "DEVICE_ID: ${DEVICE_ID}"
echo "DATASET: ${DATASET}"
echo "MODEL_FAMILY: ${MODEL_FAMILY}"
echo "Lambda values to test: ${#LAMBDA_VALUES[@]} values"
echo "========================================="
echo ""

# 遍历所有lambda值
for LAMBDA in "${LAMBDA_VALUES[@]}"
do
    echo "========================================="
    echo "Running experiment with lambda=${LAMBDA}"
    echo "========================================="

    python visualization.py \
        --dataset=${DATASET} \
        --num_classes=${NUM_CLASSES} \
        --model_family=${MODEL_FAMILY} \
        --local_learning_rate=${LOCAL_LR} \
        --global_rounds=${GLOBAL_ROUNDS} \
        --algorithm=${ALGORITHM} \
        --local_epoch=${LOCAL_EPOCH} \
        --batch_size=${BATCH_SIZE} \
        --num_clients=${NUM_CLIENTS} \
        --lamda=${LAMBDA} \
        --seed=${SEED} \
        --device_id=${DEVICE_ID} \
        --tag=${TAG} \
        --visualization_mode=${VISUALIZATION_MODE}

    # 检查上一个命令是否成功
    if [ $? -eq 0 ]; then
        echo "Successfully completed experiment with lambda=${LAMBDA}"
    else
        echo "Error: Experiment with lambda=${LAMBDA} failed"
        # 可选：遇到错误时退出
        # exit 1
    fi

    echo ""
done

echo "========================================="
echo "All experiments completed!"
echo "Results saved in ../result/"
echo "========================================="
