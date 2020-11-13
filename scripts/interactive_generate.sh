#! /bin/bash

# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

data=cnn_dm_bart_doc2word
src=source512
tgt=extwords
postfix="best"
strategy="beam"

while getopts ":g:p:f:s:" arg; do
  case $arg in
    g) GPU="$OPTARG"
    ;;
    p) LOADDIR="$OPTARG"
    ;;
    f) postfix="$OPTARG"
    ;;
    s) strategy="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

if [ "$strategy" = "beam" ];
then
    # gen_str="--beam 4 --lenpen 2.0 --max-len-b 140 --no-repeat-ngram-size 3"
    gen_str="--beam 5"
elif [ "$strategy" = "sampling" ];
then
    gen_str="--sampling --sampling-topp 0.95 --beam 1"
else
    gen_str="--beam 1"
fi

cat control_summary/datasets/cnn_dm_bart/test.tok.mosebpe40000.source512 | \
  CUDA_VISIBLE_DEVICES=${GPU} python interactive.py control_summary/data-bin/${data} \
    --path ${LOADDIR}/checkpoint_${postfix}.pt -s ${src} -t ${tgt} \
    --task translation \
    ${gen_str} --fp16 --batch-size 100 --buffer-size 100 --remove-bpe > ${LOADDIR}/gen_interactive_${strategy}_${postfix}.out

grep ^H ${LOADDIR}/gen_interactive_${strategy}_${postfix}.out | cut -f3- > ${LOADDIR}/gen_interactive_${strategy}_${postfix}.out.sys
# grep ^T ${LOADDIR}/gen_${postfix}.out | cut -f2- > ${LOADDIR}/gen_${postfix}.out.ref

