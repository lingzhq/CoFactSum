set -e

DEVICE_ID=$1
NUM_DEVICE=1
PORT_NUM=29520
DATA="./dataset/"
OUTPUT_DIR="./out/"

cd ..

CUDA_VISIBLE_DEVICES=$DEVICE_ID python -m torch.distributed.launch --nproc_per_node=$NUM_DEVICE --master_port $PORT_NUM train_topk_attn.py \
    --model_path google/pegasus-xsum \
    --data_dir $DATA/xsum \
    --output_dir $OUTPUT_DIR/pegasus_large/xsum_topk_attn/ \
    --evaluation_strategy epoch \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --max_steps 50000 \
    --max_input_length 512 \
    --max_target_length 64 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --mask_ratio 0.5 \
    --save_total_limit 3 \
    --adafactor \
    --sharded_ddp simple \
    --report_to tensorboard \


CUDA_VISIBLE_DEVICES=$DEVICE_ID python -m torch.distributed.launch --nproc_per_node=$NUM_DEVICE --master_port $PORT_NUM train_topk_attn.py \
    --model_path google/pegasus-cnn_dailymail \
    --data_dir $DATA/cnndm \
    --output_dir $OUTPUT_DIR/pegasus_large/cnndm_topk_attn/ \
    --evaluation_strategy epoch \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --max_steps 50000 \
    --max_input_length 1024 \
    --max_target_length 128 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --mask_ratio 0.1 \
    --save_total_limit 3 \
    --adafactor \
    --sharded_ddp simple \
    --report_to tensorboard \
