set -e

DEVICE_ID=$1
DATA="./dataset/"
OUTPUT_DIR="./out/"

cd ..

CUDA_VISIBLE_DEVICES=$DEVICE_ID python gen_counterfact.py \
    --data_dir $DATA/xsum/ \
    --model_path $OUTPUT_DIR/pegasus_large/xsum_consistency_prj_train/checkpoint-50000 \
    --topk_attn_model_path $OUTPUT_DIR/pegasus_large/xsum_topk_attn/checkpoint-50000 \
    --output_dir $OUTPUT_DIR/pegasus_large/xsum_final/ \
    --batch_size 1 \
    --num_beams 12 \
    --max_input_length 512 \
    --mask_ratio 0.5 \
    --debias_ratio 0.15 \
    --debias_ratio_2 0.15 \


CUDA_VISIBLE_DEVICES=$DEVICE_ID python gen_counterfact.py \
    --data_dir $DATA/cnndm/ \
    --model_path $OUTPUT_DIR/pegasus_large/cnndm_consistency_prj_train/checkpoint-50000 \
    --topk_attn_model_path $OUTPUT_DIR/pegasus_large/cnndm_topk_attn/checkpoint-50000 \
    --output_dir $OUTPUT_DIR/pegasus_large/cnndm_final/ \
    --batch_size 1 \
    --num_beams 20 \
    --max_input_length 512 \
    --mask_ratio 0.1 \
    --debias_ratio 0.05 \
    --debias_ratio_2 0.01 \
    --split_sent \
    --num_test_samples 5000 \
