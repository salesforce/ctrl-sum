#! /bin/bash
#
# train_seqlabel.sh
# Copyright (C) 2020-07-14 Junxian <He>
#
# Distributed under terms of the MIT license.
#


DATE=`date +%Y%m%d`
data_bin="arxiv"
model="bert-large-cased"
# model="roberta-large"
TPU=0
max_steps=50000
batch=8
update_freq=2
dropout=0.1
# lr=3e-5
lr=5e-5
num_cores=8

while getopts ":t:p:n:d:c:r:" arg; do
    case $arg in
        t) TPU="$OPTARG"
        ;;
        p) LOADDIR="$OPTARG"
        ;;
        n) num_cores="$OPTARG"
        ;;
        d) data_bin="$OPTARG"
        ;;
        c) cont="$OPTARG"
        ;;
        r) redis="$OPTARG"
        ;;
        \?) echo "Invalid option -$OPTARG" >&2
        ;;
    esac
done

if [ "$data_bin" = "cnn_dm_bart_new" ];
then
  max_steps=20000
  lr=3e-5
elif [ "$data_bin" = "arxiv" ];
then
  max_steps=50000
  lr=5e-5
else
  max_steps=20000
  lr=5e-5
fi

if [[ -v redis ]];
then
  redis_cmd="--build_redis"
else
  redis_cmd=""
fi


if [[ -v LOADDIR ]];
then
  if [[ -v cont ]];
  then
    SAVE=checkpoint/seqlabel/${data_bin}/${DATE}/${data_bin}.${model}.dropout${dropout}.uf${update_freq}.tpu${TPU}_continue
    rm -r ${SAVE}; mkdir -p ${SAVE}
    extra="--overwrite_output_dir --tokenizer_name ${model} --do_train --do_eval --prediction_loss_only"
    model=${LOADDIR}
    stdout="stdout.log"
  else
    SAVE=${LOADDIR}
    extra="--tokenizer_name ${model} --do_predict --prediction_loss_only"
    stdout="eval.log"
    model=${LOADDIR}
  fi
else
  SAVE=checkpoint/seqlabel/${data_bin}/${DATE}/${data_bin}.${model}.dropout${dropout}.uf${update_freq}.tpu${TPU}
  rm -r ${SAVE}; mkdir -p ${SAVE}
  extra="--overwrite_output_dir --do_train --do_eval --prediction_loss_only ${redis_cmd}"
  stdout="stdout.log"
fi

mkdir -p ${SAVE}/wandb

# wandb variables
export WANDB_NAME="${data_bin}.${DATE}.tpu${TPU}"
export WANDB_DIR="${SAVE}/wandb"

export TPU_IP_ADDRESS=${TPU}
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

python control_summary/scripts/xla_spawn.py --num_cores ${num_cores} \
  control_summary/token-classification/main_tpu.py \
  --data_dir control_summary/datasets/${data_bin}/ \
  --labels control_summary/datasets/${data_bin}/labels.txt \
  --model_name_or_path ${model} \
  --output_dir ${SAVE} \
  --num_train_epochs 3 \
  --max_steps ${max_steps} \
  --max_seq_length 512 \
  --per_device_train_batch_size ${batch} \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps ${update_freq} \
  --save_steps 1000 \
  --eval_steps 1000 \
  --threshold 0.1 \
  --dropout ${dropout} \
  --warmup_steps 500 \
  --weight_decay 0.01 \
  --learning_rate ${lr} \
  --evaluate_during_training \
  --logging_steps 100 \
  --save_total_limit 10 \
  --logging_dir ${SAVE} \
  --seed 1 \
  ${extra} \
  | tee -a ${SAVE}/${stdout}
  # --fp16 \
  # --eval_split dev \
  # --overwrite_cache \
