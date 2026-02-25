#/bin/bash

cat << EOF > .env
ATTN_MODE=sdpa
PRETRAINED_MODEL_PATH=/path/to/your/pretrained_model_path
DATASETS_PATH=/path/to/your/datasets_path
CHECK_TENSOR=0
EOF
