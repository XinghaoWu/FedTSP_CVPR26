# 从命令行解析参数
alpha=$1
num_clients=$2
model_family=$3
lamda=$4
batch_size=${5:-100}
seed=${6:-0}
device_id=${7:-0}

echo "参数列表:"
echo "alpha: ${alpha}"
echo "num_clients: ${num_clients}"
echo "model_family: ${model_family}"
echo "lamda: ${lamda}"
echo "batch_size: ${batch_size}"
echo "seed: ${seed}"
echo "device_id: ${device_id}"