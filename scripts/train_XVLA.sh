source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=2 
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file configs/accelerate/zero2.yaml \
    --num_processes 8 \
    train.py \
    --config_path configs/XVLA.yaml \
    2>&1 | tee train_XVLA.log
