#! /bin/bash
#
# test_bart_from_tagger.sh
# Copyright (C) 2020-09-22 Junxian <He>
#
# Distributed under terms of the MIT license.
#



gpu=0
src=test.extwordssource
tgt=test.targettrunc
exp=.
data=cnn_dm_bart_new
outfix=default
split="test"

while getopts ":g:e:p:d:t:m:r:n:" arg; do
    case $arg in
        g) gpu="$OPTARG"
        ;;
        e) exp="$OPTARG"
        ;;
        p) pred="$OPTARG"
        ;;
        d) data="$OPTARG"
        ;;
        t) tgt="$OPTARG"
        ;;
        m) maxword="$OPTARG"
        ;;
        r) ts="$OPTARG"
        ;;
        n) sumlen="$OPTARG"
        ;;
        \?) echo "Invalid option -$OPTARG" >&2
        ;;
    esac
done

outfix_tune="sumlen${sumlen}.mw${maxword}.tune"
cd control_summary/datasets/${data}
python preprocess.py --mode post_tag --split ${split} --summary-size ${sumlen} --maximum-word ${maxword} --threshold ${ts} --outfix ${outfix_tune} --tag-pred ${pred}
python preprocess.py --mode paste --split ${split} --src sourcetrunc --paste-key ts${ts}.${outfix_tune}.predwords
python preprocess.py --mode add_leading_space --split ${split} --src ts${ts}.${outfix_tune}.predwordssourcetrunc --tgt targettrunc

src="${split}.ts${ts}.${outfix_tune}.predwordssourcetrunclead"
cd ../../../

bash control_summary/scripts/test_bart.sh -g ${gpu} -s ${src} -t ${tgt} -p ${exp} -d ${data}

