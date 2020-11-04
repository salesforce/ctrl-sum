#! /bin/bash
#
# test_bart.sh
# Copyright (C) 2020-07-12 Junxian <He>
#
# Distributed under terms of the MIT license.
#
gpu=0
src=test.oraclewordnssource
tgt=test.target
exp=.
data=cnndm
outfix=default
lenpen=1
minlen=1
extra=""

export CLASSPATH=stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar

while getopts ":g:s:p:d:t:o:l:m:e:" arg; do
    case $arg in
        g) gpu="$OPTARG"
        ;;
        s) src="$OPTARG"
        ;;
        p) exp="$OPTARG"
        ;;
        d) data="$OPTARG"
        ;;
        t) tgt="$OPTARG"
        ;;
        o) outfix="$OPTARG"
        ;;
        l) lenpen="$OPTARG"
        ;;
        m) minlen="$OPTARG"
        ;;
        e) extra="$OPTARG"
        ;;
        \?) echo "Invalid option -$OPTARG" >&2
        ;;
    esac
done

IFS=','
read -a gpuarr <<< "$gpu"

nsplit=${#gpuarr[@]}
# echo "nsplit ${nsplit}"

split -n l/${nsplit} -d datasets/${data}/${src} datasets/${data}/${src}.


for ((i=0;i<nsplit;i++))
do
    # echo "i $i"
    gpu_s=${gpuarr[$i]}
    # echo "gpu ${gpu_s}"
    printf -v j "%02d" $i
    # echo "j: $j"
    CUDA_VISIBLE_DEVICES=${gpu_s} python scripts/generate_bart.py --exp ${exp} --src ${src}.$j --dataset ${data} --outfix ${outfix} --lenpen ${lenpen} --min-len ${minlen} ${extra} &
done

# wait for the decoding to finish
wait

> ${exp}/${src}.${outfix}.hypo
for ((i=0;i<nsplit;i++))
do
    printf -v j "%02d" $i
    cat ${exp}/${src}.$j.${outfix}.hypo >> ${exp}/${src}.${outfix}.hypo
    rm ${exp}/${src}.$j.${outfix}.hypo
    rm datasets/${data}/${src}.$j
done

cat ${exp}/${src}.${outfix}.hypo | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${exp}/${src}.${outfix}.hypo.tokenized
cat datasets/${data}/${tgt} | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${exp}/${tgt}.tokenized
files2rouge ${exp}/${tgt}.tokenized ${exp}/${src}.${outfix}.hypo.tokenized  > ${exp}/${src}.${outfix}.rouge

# compute BERTScore
# cp datasets/${data}/${tgt} ${exp}/
# CUDA_VISIBLE_DEVICES=${gpu} bert-score -r ${exp}/${tgt} -c ${exp}/${src}.${outfix}.hypo --lang en --rescale_with_baseline > ${exp}/${src}.${outfix}.bertscore

