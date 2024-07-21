#!/bin/bash
OUTPUT_DIR=./output/solar_inst/
mkdir -p $OUTPUT_DIR

TRAIN_DATA=./data/train.json
EVAL_DATA=./data/valid.json
MODEL_ID=upstage/SOLAR-10.7B-Instruct-v1.0 
BATCH_SIZE=8
NUM_GPUS=2
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
NUM_EPOCHS=15
WANDB_RUN_NAME=mobis_solar_inst

torchrun --nproc_per_node=$NUM_GPUS --nnodes 1 --rdzv_backend c10d --rdzv_endpoint localhost:0 train.py \
--output_dir $OUTPUT_DIR \
--model_id $MODEL_ID \
--train_data $TRAIN_DATA \
--eval_data $EVAL_DATA \
--num_train_epochs $NUM_EPOCHS \
--batch_size $BATCH_SIZE \
--per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
--per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
--wandb_run_name $WANDB_RUN_NAME
