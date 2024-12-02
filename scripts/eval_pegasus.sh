set -e

DEVICE_ID=$1
DATA="./dataset/"
OUTPUT_DIR="./out/"

CUDA_VISIBLE_DEVICES=$DEVICE_ID python eval.py \
    --data_dir $DATA/xsum/ \
    --output_dir $OUTPUT_DIR/xsum/pegasus/ \


CUDA_VISIBLE_DEVICES=$DEVICE_ID python eval.py \
    --data_dir $DATA/cnndm/ \
    --output_dir $OUTPUT_DIR/cnndm/pegasus/ \
    --split_sent \
    --num_test_samples 5000 \
