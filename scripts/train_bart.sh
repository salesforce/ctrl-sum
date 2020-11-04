#! /bin/bash
#
# train.sh
# Copyright (C) 2020-06-04 Junxian <He>
#
# Distributed under terms of the MIT license.
#

DATE=`date +%Y%m%d`
data_bin="cnndm"
dropout=0.1
label_smoothing=0.1
train_steps=30000
warmup_updates=500
lr=3e-05
src='oraclewordsource'
tgt='target'
update_freq=8
max_tokens=1024
save_interval_updates=2000
keep_interval_updates=1
log_interval=200

criterion='label_smoothed_cross_entropy'

GPU=0
checkpoint="checkpoint_best.pt"

while getopts ":g:p:d:l:b:s:t:" arg; do
    case $arg in
        g) GPU="$OPTARG"
        ;;
        p) LOADDIR="$OPTARG"
        ;;
        d) data_bin="$OPTARG"
        ;;
        l) checkpoint="$OPTARG"
        ;;
        b) bartpath="$OPTARG"
        ;;
        s) src="$OPTARG"
        ;;
        t) tgt="$OPTARG"
        ;;
        \?) echo "Invalid option -$OPTARG" >&2
        ;;
    esac
done

if [ "$data_bin" = "cnndm" ];
then
    train_steps=20000
elif [ "$data_bin" = "arxiv" ];
then
    train_steps=50000
elif [ "$data_bin" = "big_patent" ];
then
    train_steps=300000
    warmup_updates=3000
    lr=5e-3
    save_interval_updates=15000
else
    train_steps=30000
fi


if [[ -v LOADDIR ]];
then
    add_load_string=""
    cstring="_continue"
else
    add_load_string="--reset-optimizer --reset-dataloader --reset-meters"
    cstring=""
fi


GPUSTR=$(printf "$GPU" | tr , _)

SAVE=checkpoint/${data_bin}/${DATE}/${data_bin}.${criterion}.${src}-${tgt}.lsm${label_smoothing}.drop${dropout}.uf${update_freq}.gpu${GPUSTR}${cstring}
TENSORBOARD=${SAVE}/tensorboard

rm -r ${SAVE}; mkdir -p ${SAVE} ${TENSORBOARD}

if [[ -v LOADDIR ]];
then
    echo "load from ${LOADDIR}/${checkpoint}"
    cp ${LOADDIR}/${checkpoint} ${SAVE}/checkpoint_load.pt
    restore_file=${SAVE}/checkpoint_load.pt
else
    echo "load BART large"
    restore_file=${bartpath}
fi

echo "start training"

CUDA_VISIBLE_DEVICES=${GPU} fairseq-train data-bin/${data_bin} \
    --restore-file ${restore_file} \
    --max-tokens ${max_tokens} \
    --task translation \
    --source-lang ${src} --target-lang ${tgt} \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr ${lr} --total-num-update ${train_steps} --warmup-updates ${warmup_updates} \
    --max-update ${train_steps} \
    --update-freq ${update_freq} \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --log-format simple --log-interval ${log_interval} \
    --best-checkpoint-metric ppl \
    --save-dir ${SAVE} \
    --save-interval-updates ${save_interval_updates} --tensorboard-logdir ${TENSORBOARD}\
    --validate-interval 1000 --keep-interval-updates ${keep_interval_updates} --save-interval 1000 --no-epoch-checkpoints \
    ${add_load_string} \
    | tee -a ${SAVE}/stdout.log

