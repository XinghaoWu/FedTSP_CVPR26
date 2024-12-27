@echo off

REM 从命令行解析参数
set dataset=%1
set num_classes=%2
set noniid=%3
set alpha=%4
set num_clients=%5
set model_family=%6
set batch_size=%7
set device_id=%8
set global_rounds=%9

if "%batch_size%"=="" (
    set batch_size=100
)

if "%device_id%"=="" (
    set device_id=0
)

if "%global_rounds%"=="" (
    set global_rounds=300
)

echo 参数列表:
echo dataset: %dataset%
echo num_classes: %num_classes%
echo noniid: %noniid%
echo alpha: %alpha%
echo num_clients: %num_clients%
echo model_family: %model_family%
echo batch_size: %batch_size%
echo device_id: %device_id%
echo global_rounds: %global_rounds%

if "%dataset%"=="" (
    echo 请提供 dataset noniid alpha, num_clients, model_family 等参数
    exit /b 1
)

REM 动态生成 dataset 参数
set dataset=%dataset%_%noniid%_%alpha%_balance_%num_clients%

for %%S in (0 1 2) do (
    python main.py --dataset=%dataset% ^
        --num_classes=%num_classes% ^
        --model_family=%model_family% ^
        --local_learning_rate=0.01 ^
        --global_rounds=%global_rounds% ^
        --algorithm=FedKD ^
        --local_epochs=5 ^
        --batch_size=%batch_size% ^
        --num_clients=%num_clients% ^
        --mentee_learning_rate=0.01 ^
        --T_start=0.95 ^
        --T_end=0.95 ^
        --seed=%%S ^
        --device_id=%device_id% ^
        --save_model=1
)
