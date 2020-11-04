#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-09-22 Junxian <He>
#
# Distributed under terms of the MIT license.

"""
Evaluate length control
"""

import argparse
import numpy as np

from scipy.stats import pearsonr
from control_summary.datasets.cnn_dm_bart_new.preprocess import (
       cluster_length_to_bin,
       length_to_string,
)

def eval_mean(lenfile, ref, sys, num_bin=5, mode='std', iterate=False):
    with open(lenfile) as fin:
        length_list = [len(line.strip().split()) for line in fin]

    len_bin = cluster_length_to_bin(length_list, num_bin)

    print(f'lenbin: {len_bin}')

    sum_ = 0
    cnt = 0
    with open(ref) as fref, \
         open(sys) as fsys:
        for i, (lr, ls) in enumerate(zip(fref, fsys)):
            bucket_s = int(length_to_string(len(ls.strip().split()), len_bin))
            if not iterate:
                bucket_r = int(length_to_string(len(lr.strip().split()), len_bin))
            else:
                bucket_r = i % num_bin
            if mode == 'std':
                inc = (bucket_r - bucket_s) ** 2
            elif mode == 'am':
                inc = np.absolute(bucket_r - bucket_s)
            else:
                raise ValueError
            sum_ += inc
            cnt += 1

    if mode == 'std':
        mean = np.sqrt(sum_ / cnt)
    elif mode == 'am':
        mean = sum_ / cnt
    print(f'mean: {mean}')
    return mean

def eval_token_std(ref, sys):
    sum_ = 0
    cnt = 0
    with open(ref) as fref, \
         open(sys) as fsys:
        for lr, ls in zip(fref, fsys):
            bucket_r = len(lr.strip().split())
            bucket_s = len(ls.strip().split())
            sum_ += (bucket_r - bucket_s) ** 2
            cnt += 1

    std = np.sqrt(sum_ / cnt)
    print(f'token std: {std}')
    return std

def eval_var(ref, sys):
    sum_ = 0
    cnt = 0
    with open(ref) as fref, \
         open(sys) as fsys:
        for lr, ls in zip(fref, fsys):
            bucket_r = len(lr.strip().split())
            bucket_s = len(ls.strip().split())
            sum_ += (bucket_r - bucket_s) ** 2
            cnt += 1

    var = 0.001 * sum_ / cnt
    print(f'var: {var}')
    return var

def eval_pcc(lenfile, sys, num_bin=5):
    with open(lenfile) as fin:
        length_list = [len(line.strip().split()) for line in fin]

    len_bin = cluster_length_to_bin(length_list, num_bin)
    print(f'lenbin: {len_bin}')

    length_code = []
    actual_len = []
    with open(sys) as fsys:
        for i, line in enumerate(fsys):
            length_code.append(i % num_bin)
            actual_len.append(int(length_to_string(len(line.strip().split()), len_bin)))
            # actual_len.append(len(line.strip().split()))

    correlation = pearsonr(length_code, actual_len)
    print(f'PCC: {correlation[0]}, p-value: {correlation[1]}')

parser = argparse.ArgumentParser(description='various preprocessing for summarization task')
parser.add_argument('--mode', type=str, choices=['std', 'token_std', 'var', 'am', 'pcc'], default='std')
parser.add_argument('--lenfile', type=str,
        help='the target file used to split length into length buckets')
parser.add_argument('--sys', type=str, help='system output, untokenized')
parser.add_argument('--ref', type=str, help='reference, untokenized')
parser.add_argument('--iterate', action='store_true', default=False,
        help='iterated version')

args = parser.parse_args()

if args.mode == 'std' or args.mode == 'am':
    # qudratic mean
    eval_mean(args.lenfile, args.ref, args.sys, mode=args.mode, iterate=args.iterate)

if args.mode == 'token_std':
    eval_token_std(args.ref, args.sys)

if args.mode == 'var':
    eval_var(args.ref, args.sys)

if args.mode == 'pcc':
    if not args.iterate:
        raise ValueError
    eval_pcc(args.lenfile, args.sys)

