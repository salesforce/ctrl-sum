
# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import argparse


parser = argparse.ArgumentParser(description='repeat over different length to test length control')
parser.add_argument('--src', type=str, help='src')
parser.add_argument('--tgt', type=str, help='tgt')
parser.add_argument('--num', type=int, default=5, help='bucket number')

args = parser.parse_args()
src_out = args.src + 'repeat{}'.format(args.num)
tgt_out = args.tgt + 'repeat{}'.format(args.num)

len_list = ['<len_{}>'.format(x) for x in range(args.num)]

with open(args.src) as fin_src, \
     open(args.tgt) as fin_tgt, \
     open(src_out, 'w') as fout_src, \
     open(tgt_out, 'w') as fout_tgt:
     for line_src, line_tgt in zip(fin_src, fin_tgt):
        fout_src.write(line_src)
        fout_tgt.write(line_tgt)

        true_len = line_src.split()[0]
        src_content = ' '.join(line_src.split()[1:])
        for len_tok in len_list:
            if len_tok != true_len:
                fout_src.write('{} {}\n'.format(len_tok, src_content))
                fout_tgt.write(line_tgt)
