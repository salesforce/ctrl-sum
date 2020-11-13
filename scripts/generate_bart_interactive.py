#! /usr/bin/env python

# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import os
import torch
import argparse
import sys
from fairseq.models.bart import BARTModel

parser = argparse.ArgumentParser(description='generation with BART')
parser.add_argument('--exp', type=str, help='experiment path')
parser.add_argument('--src', type=str, default='test.extwordssourcetrunclead', help='experiment path')
parser.add_argument('--dataset', type=str, default='cnn_dm_bart_new', help='dataset name')
parser.add_argument('--outfix', type=str, default='default', help='outfix')
parser.add_argument('--lenpen', type=float, default=1.0, help='length penalty')
parser.add_argument('--unconstrained', action='store_true', default=False, help='unconstrained summarization model')
parser.add_argument('--max-len-b', type=int, default=140, help='length penalty')
parser.add_argument('--rc', action='store_true', default=False, help='reading comprehension test')

args = parser.parse_args()

params_dict = {
        'cnndm': {'max_len_b': 140},
        'arxiv': {'max_len_b': 256},
        'big_patent': {'max_len_b': 256},
    }

params = params_dict[args.dataset]

if args.rc:
    params_rc = {
        'exclude_prefix_max_len': 10
    }
else:
    params_rc = None
    # args.lenpen = 0.1

try:
    bart = BARTModel.from_pretrained(
        args.exp,
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path='data-bin/{}'.format(args.dataset)
    )
except:
    bart = BARTModel.from_pretrained(
        args.exp,
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=args.exp
    )

# bart = torch.hub.load('pytorch/fairseq', 'bart.large.cnn')

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 1
with open('datasets/{}/{}'.format(args.dataset, args.src)) as source:
    input_src = source.readlines()


while True:
    id_ = input('Type example id :')
    if id_ == 'exit':
        break

    if args.unconstrained:
        source = input_src[int(id_)].strip()
        source = ' {}'.format(source)
        slines = [source]
        prefix_tokens = None
    else:
        source = input_src[int(id_)].rstrip()
        keywords = source.split(' => ')[0]
        content = ' => '.join(source.split(' => ')[1:])
        print('automatic keywords :{}'.format(keywords))

        #remove sentence separator
        # keywords = ' '.join(keywords.split(' | '))
        slines = [' => '.join([keywords, content])]

        prefix_tokens = None

    with torch.no_grad():
        hypotheses_batch = bart.sample(slines, beam=4, prefix_tokens=prefix_tokens, lenpen=args.lenpen, max_len_b=params['max_len_b'], min_len=1, no_repeat_ngram_size=3, extra_gen_cls_kwargs=params_rc)

    for hypothesis in hypotheses_batch:
        print(f'SUMMARY: {hypothesis}\n')

    while True:
        if args.unconstrained:
            prefix = input("input prefix (type 'exit' to go back to example selection):")
            if prefix == 'exit':
                break
            prefix = ' {}'.format(prefix.strip())
        else:
            keywords = input("input keywords (type 'exit' to go back to example selection):")
            if keywords == 'exit':
                break
            prefix = input("input prefix (type 'exit' to go back to example selection):")
            if prefix.strip() == '':
                prefix = None
            elif prefix == 'same':
                prefix = keywords

            keywords = ' {}'.format(keywords.strip())
            if prefix is not None:
                prefix = ' {}'.format(prefix.strip())
            slines = [' => '.join([keywords, content])]

        if prefix is None:
            prefix_tokens = None
        else:
            prefix_tokens = [prefix]

        with torch.no_grad():
            try:
                hypotheses_batch = bart.sample(slines, beam=4, prefix_tokens=prefix_tokens, lenpen=args.lenpen, max_len_b=params['max_len_b'], min_len=1, no_repeat_ngram_size=3, extra_gen_cls_kwargs=params_rc)
            except:
                hypotheses_batch = bart.sample(slines, beam=4, prefix_tokens=prefix_tokens, lenpen=args.lenpen, max_len_b=params['max_len_b'], min_len=1, no_repeat_ngram_size=0, extra_gen_cls_kwargs=params_rc)

        for hypothesis in hypotheses_batch:
            print(f'SUMMARY: {hypothesis}\n')

