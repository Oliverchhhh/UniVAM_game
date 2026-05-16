CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file configs/accelerate/zero3.yaml \
    --num_processes 2 \
    train.py \
    --config_path configs/libero.yaml \
    2>&1 | tee train_libero.log
