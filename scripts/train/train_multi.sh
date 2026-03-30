/root/apps/miniconda3/envs/univam/bin/accelerate launch \
    --config_file configs/accelerate/zero2.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --num_machines $WORLD_SIZE \
    --machine_rank $RANK \
    train.py \
    --config_path configs/XVLA.yaml \
    2>&1 | tee train_XVLA_node${RANK}.log
