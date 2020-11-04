#! /bin/bash
#
# gpt2_encode.sh
# Copyright (C) 2020-11-03 Junxian <He>
#
# Distributed under terms of the MIT license.
#

dataset=$1

sourcelang=${2:-oraclewordsource}
targetlang=${3:-target}

wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json' -P datasets/${dataset}
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe' -P datasets/${dataset}
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt' -P datasets/${dataset}

for SPLIT in train val
do
  for LANG in ${sourcelang} ${targetlang}
  do
    python scripts/multiprocessing_bpe_encoder.py \
    --encoder-json datasets/${dataset}/encoder.json \
    --vocab-bpe datasets/${dataset}/vocab.bpe \
    --inputs "datasets/${dataset}/$SPLIT.$LANG" \
    --outputs "datasets/${dataset}/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done
