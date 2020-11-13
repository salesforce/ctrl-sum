#! /bin/bash

# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

dataset=$1
datadir=datasets/${dataset}

sourcelang=${2:-oraclewordsource}
targetlang=${3:-target}

fairseq-preprocess --source-lang ${sourcelang} --target-lang ${targetlang} \
    --trainpref ${datadir}/train.bpe --validpref ${datadir}/val.bpe \
    --destdir data-bin/${dataset} --workers 20 \
    --srcdict ${datadir}/dict.txt --tgtdict ${datadir}/dict.txt

