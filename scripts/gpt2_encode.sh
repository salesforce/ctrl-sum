#! /bin/bash

# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

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
