set -e

DEVICE_ID=$1
DATA="./dataset/"
PRED_DIR="./pred/"
OUTPUT_DIR="./out/"

CUDA_VISIBLE_DEVICES=$DEVICE_ID python eval.py \
    --data_dir $DATA/xsum_with_ent/ \
    --pred_dir $PRED_DIR/xsum/86712/generated_txt_0_xsum_beam=1_512_256/ \
    --output_dir $OUTPUT_DIR/xsum/spancopy/86712/ \
    --spancopy \


CUDA_VISIBLE_DEVICES=$DEVICE_ID python eval.py \
    --data_dir $DATA/cnndm_with_ent/ \
    --pred_dir $PRED_DIR/cnndm/89723/generated_txt_0_cnndm_beam=1_1024_256/ \
    --output_dir $OUTPUT_DIR/cnndm/spancopy/89723/ \
    --split_sent \
    --num_test_samples 5000 \
    --spancopy \
