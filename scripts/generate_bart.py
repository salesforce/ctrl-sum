#! /usr/bin/env python

# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import os
import torch
import argparse
from fairseq.models.bart import BARTModel

parser = argparse.ArgumentParser(description='generation with BART')
parser.add_argument('--exp', type=str, help='experiment path')
parser.add_argument('--src', type=str, default='test.extwordssource512', help='experiment path')
parser.add_argument('--dataset', type=str, default='cnn_dm_bart_new', help='dataset name')
parser.add_argument('--outfix', type=str, default='default', help='outfix')
parser.add_argument('--lenpen', type=float, default=1.0, help='length penalty')
parser.add_argument('--keywords-as-prefix', action='store_true', default=False, help='use keywords as prefix as well')
parser.add_argument('--prefix-only', action='store_true', default=False,
        help='use keywords as prefix but not keywords, this is for unconstrained sum model')
parser.add_argument('--remove-prefix-from-output', action='store_true', default=False,
        help='remove prefix from output, this is only valid when prefix is turned on')
parser.add_argument('--rc', action='store_true', default=False, help='reading comprehension test')
parser.add_argument('--min-len', type=int, default=1, help='length penalty')
parser.add_argument('--max-len', type=int, default=None, help='length penalty')

args = parser.parse_args()

params_dict = {
        'cnndm': {'max_len_b': 140, 'beam': 4, 'no_repeat': 3},
        'arxiv': {'max_len_b': 256, 'beam': 4, 'no_repeat': 3},
        'big_patent': {'max_len_b': 256, 'beam': 4, 'no_repeat': 3},
    }

params = params_dict.get(args.dataset, params_dict['cnndm'])

if args.max_len is not None:
    params['max_len_b'] = args.max_len

if args.rc:
    params_rc = {
        'exclude_prefix_max_len': 10
    }
    args.lenpen = 0.01
    # params['beam'] = 4
    # params['no_repeat'] = 0
else:
    params_rc = None

try:
    bart = BARTModel.from_pretrained(
        args.exp,
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path='data-bin/{}'.format(args.dataset),
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
bsz = 16
prefix_tokens = None
if args.rc:
     bsz = 1
with open('datasets/{}/{}'.format(args.dataset, args.src)) as source, open(os.path.join(args.exp, '{}.{}.hypo'.format(args.src, args.outfix)), 'w') as fout:
    sline = source.readline().rstrip()
    if args.keywords_as_prefix or args.prefix_only:
        prefix = sline.split(' => ')[0]
        prefix = ' {}'.format(prefix.strip())
        prefix_tokens = [prefix]

    if args.prefix_only:
        sline = ' => '.join(sline.split(' => ')[1:])
        sline = ' {}'.format(sline.strip())

    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                try:
                    hypotheses_batch = bart.sample(slines, beam=params['beam'], prefix_tokens=prefix_tokens, lenpen=args.lenpen, max_len_b=params['max_len_b'], min_len=args.min_len, no_repeat_ngram_size=params['no_repeat'], extra_gen_cls_kwargs=params_rc)
                except:
                    hypotheses_batch = bart.sample(slines, beam=params['beam'], prefix_tokens=prefix_tokens, lenpen=args.lenpen, max_len_b=params['max_len_b'], min_len=args.min_len, no_repeat_ngram_size=0, extra_gen_cls_kwargs=params_rc)

            for i, hypothesis in enumerate(hypotheses_batch):
                # print(hypothesis.strip())
                if args.remove_prefix_from_output:
                    hypothesis = hypothesis[len(prefix_tokens[i])-5:]
                fout.write(hypothesis.strip() + '\n')
                fout.flush()
            slines = []

            if args.keywords_as_prefix or args.prefix_only:
                prefix_tokens = []

        if args.keywords_as_prefix or args.prefix_only:
            prefix = sline.split(' => ')[0]
            prefix = ' {}'.format(prefix.strip())
            prefix_tokens.append(prefix)

        if args.prefix_only:
            sline = ' => '.join(sline.split(' => ')[1:])
            sline = ' {}'.format(sline.strip())

        slines.append(sline.rstrip())
        count += 1
    if slines != []:
        try:
            hypotheses_batch = bart.sample(slines, beam=params['beam'], prefix_tokens=prefix_tokens, lenpen=args.lenpen, max_len_b=params['max_len_b'], min_len=args.min_len, no_repeat_ngram_size=params['no_repeat'], extra_gen_cls_kwargs=params_rc)
        except:
            hypotheses_batch = bart.sample(slines, beam=params['beam'], prefix_tokens=prefix_tokens, lenpen=args.lenpen, max_len_b=params['max_len_b'], min_len=args.min_len, no_repeat_ngram_size=0, extra_gen_cls_kwargs=params_rc)
        for i, hypothesis in enumerate(hypotheses_batch):
            if args.remove_prefix_from_output:
                hypothesis = hypothesis[len(prefix_tokens[i])-5:]
            fout.write(hypothesis.strip() + '\n')
            fout.flush()
