
# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import argparse

def cluster_length_to_bin(len_list, num):
    """compute bin bound
    """

    avg_bin_len = len(len_list) // num
    sort_list = sorted(len_list)

    bin_bound = []

    for i in range(num):
        if i == (num - 1):
            bin_bound.append((sort_list[i * avg_bin_len], 1e5))
        else:
            bin_bound.append((sort_list[i * avg_bin_len], sort_list[(i+1) * avg_bin_len]))

    return bin_bound

def length_to_string(len, bin_bound):
    flag = False
    for i, bucket in enumerate(bin_bound):
        if len >= bucket[0] and len < bucket[1]:
            id_ = i
            flag = True
            break

    if not flag:
        raise ValueError("didn't find a bucket for length {}".format(len))

    return "<len_{}>".format(id_)


parser = argparse.ArgumentParser(description='prepend length control code to the source file')
parser.add_argument('--src', type=str, help='src input')
parser.add_argument('--tgt', type=str, help='tgt input')

args = parser.parse_args()
num_buckets = 5

split_list = ['train', 'valid', 'test']


with open('train.{}'.format(args.tgt_postfix)) as ftgt:
    length_list = [len(line.rstrip().split()) for line in ftgt]

bin_bound = cluster_length_to_bin(length_list, num_buckets)
print('bin_bound {}'.format(bin_bound))


# for split_name in split_list:
#     fout = open('{}.{}ctrl'.format(split_name, args.src_postfix), 'w')

#     with open('{}.{}'.format(split_name, args.src_postfix)) as fsrc, \
#          open('{}.{}'.format(split_name, args.tgt_postfix)) as ftgt:
#         for line_src, line_tgt in zip(fsrc, ftgt):
#             length = len(line_tgt.rstrip().split())
#             new_line_src = "{} {}".format(length_to_string(length, bin_bound), line_src.rstrip())
#             fout.write('{}\n'.format(new_line_src))

#     fout.close()
