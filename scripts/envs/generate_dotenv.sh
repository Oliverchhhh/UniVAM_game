#/bin/bash

cat << EOF > .env
# sdpa | flash_attention_2
ATTN_MODE=sdpa
PRETRAINED_MODEL_PATH=/path/to/your/pretrained_model_path
DATASETS_PATH=/path/to/your/datasets_path
CHECK_TENSOR=0

# eval part
EVAL_JSON_PATH=./jsons/debug.json
RESUME_PATH=./ckpt/model/
EOF
