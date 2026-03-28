export HF_ENDPOINT=https://hf-mirror.com

DATASET_ID="Facebear/XVLA-Soft-Fold"
DATASETS_PATH="./"

THREADS=10

echo "开始下载数据集: $DATASET_ID"
echo "保存路径: $DATASETS_PATH/${DATASET_ID##*/}"

until ./hfd.sh "$DATASET_ID" --dataset --tool aria2c -x $THREADS; do
    echo ""
    echo "⚠️ [异常中断]"
    sleep 5
done

