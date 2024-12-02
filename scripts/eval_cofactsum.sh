set -e

DEVICE_ID=$1
DATA="./dataset/"
OUTPUT_DIR="./out/"

cd ..

CUDA_VISIBLE_DEVICES=$DEVICE_ID python eval.py \
    --data_dir $DATA/xsum/ \
    --output_dir $OUTPUT_DIR/xsum/cofactsum/ \


CUDA_VISIBLE_DEVICES=$DEVICE_ID python eval.py \
    --data_dir $DATA/cnndm/ \
    --output_dir $OUTPUT_DIR/cnndm/cofactsum/ \
    --split_sent \
    --num_test_samples 5000 \
