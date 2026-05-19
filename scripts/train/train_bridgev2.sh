CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file configs/accelerate/zero2.yaml \
    --num_processes 4 \
    train.py \
    --config_path configs/bridgev2.yaml \
    2>&1 | tee train_bridgev2.log
