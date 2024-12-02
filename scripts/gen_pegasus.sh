set -e

DEVICE_ID=$7
DATA="./dataset/"
OUTPUT_DIR="./out/"

CUDA_VISIBLE_DEVICES=$DEVICE_ID python generate.py \
    --data_dir $DATA/xsum/ \
    --model_path google/pegasus-xsum \
    --output_dir $OUTPUT_DIR/pegasus_large/xsum/ \
    --batch_size 1 \
    --num_beams 8 \
    --max_input_length 512 \


CUDA_VISIBLE_DEVICES=$DEVICE_ID python generate.py \
    --data_dir $DATA/cnndm/ \
    --model_path google/pegasus-cnn_dailymail \
    --output_dir $OUTPUT_DIR/pegasus_large/cnndm/ \
    --batch_size 1 \
    --num_beams 8 \
    --max_input_length 512 \
    --split_sent \
    --num_test_samples 5000 \
