#! /bin/bash
#
# preprocess_fairseq.sh
# Copyright (C) 2020-06-04 Junxian <He>
#
# Distributed under terms of the MIT license.
#


dataset=$1
datadir=datasets/${dataset}

sourcelang=${2:-oraclewordsource}
targetlang=${3:-target}

fairseq-preprocess --source-lang ${sourcelang} --target-lang ${targetlang} \
    --trainpref ${datadir}/train.bpe --validpref ${datadir}/val.bpe \
    --destdir data-bin/${dataset} --workers 20 \
    --srcdict ${datadir}/dict.txt --tgtdict ${datadir}/dict.txt

