set -e

DEVICE_ID=$1
NUM_DEVICE=1
PORT_NUM=29521
DATA="./dataset/"
OUTPUT_DIR="./out/"

cd ..

CUDA_VISIBLE_DEVICES=$DEVICE_ID python -m torch.distributed.launch --nproc_per_node=$NUM_DEVICE --master_port $PORT_NUM train_consistency_prj.py \
    --model_path google/pegasus-xsum \
    --data_dir $DATA/xsum \
    --negative_data_dir $DATA/xsum_negative \
    --output_dir $OUTPUT_DIR/pegasus_large/xsum_consistency_prj/ \
    --mask_type negative_syslowcon \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --metric_for_best_model acc \
    --early_stopping_patience 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --max_steps 50000 \
    --max_input_length 512 \
    --max_target_length 64 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --mask_ratio 0.5 \
    --save_total_limit 3 \
    --sharded_ddp simple \
    --report_to tensorboard \
    --load_best_model_at_end True \


CUDA_VISIBLE_DEVICES=$DEVICE_ID python -m torch.distributed.launch --nproc_per_node=$NUM_DEVICE --master_port $PORT_NUM train_consistency_prj.py \
    --model_path google/pegasus-cnn_dailymail \
    --data_dir $DATA/cnndm \
    --negative_data_dir $DATA/cnndm_negative \
    --output_dir $OUTPUT_DIR/pegasus_large/cnndm_consistency_prj/ \
    --mask_type negative_syslowcon \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --metric_for_best_model acc \
    --early_stopping_patience 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --max_steps 50000 \
    --max_input_length 1024 \
    --max_target_length 128 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --mask_ratio 0.1 \
    --save_total_limit 3 \
    --sharded_ddp simple \
    --report_to tensorboard \
    --load_best_model_at_end True \
