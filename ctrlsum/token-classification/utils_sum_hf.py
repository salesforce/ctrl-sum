# coding=utf-8

# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

""" This code is based on
https://github.com/huggingface/transformers/blob/master/examples/token-classification/utils_ner.py,
but we modify it to use the hugginface/datasets library as the backend
to deal with large datasets
"""


import logging
import os
import sys
import shutil
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union, Dict

from filelock import FileLock

from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available


logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]]


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class Split(Enum):
    train = "train"
    dev = "val"
    test = "test"


import torch
from torch import nn
from torch.utils.data.dataset import Dataset

from datasets import load_dataset

# this is ugly, but to use multiprocessing in datasets.map, seems
# I need to define the map function at the top level

cls_token_at_end = None
cls_token = None
cls_token_segment_id = None
sep_token = None
sep_token_extra = None
pad_on_left = pad_token = pad_token_segment_id = pad_token_label_id = None
sequence_a_segment_id = None
mask_padding_with_zero = None
tokenizer = None
max_seq_length = None

def convert_examples_to_features(example):
    """ Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers

    tokens = []
    label_ids = []
    for word, label in zip(example['tokens'], example['labels']):
        word_tokens = tokenizer.tokenize(word)

        # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
        if len(word_tokens) > 0:
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label] + [pad_token_label_id] * (len(word_tokens) - 1))

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = tokenizer.num_special_tokens_to_add()
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        label_ids = label_ids[: (max_seq_length - special_tokens_count)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens += [sep_token]
    label_ids += [pad_token_label_id]
    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if cls_token_at_end:
        tokens += [cls_token]
        label_ids += [pad_token_label_id]
        segment_ids += [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        label_ids = ([pad_token_label_id] * padding_length) + label_ids
    else:
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length


    if "token_type_ids" not in tokenizer.model_input_names:
        segment_ids = None

    # features.append(
    #     InputFeatures(
    #         input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label_ids=label_ids
    #     )
    # )

    return {'input_ids': input_ids,
            'attention_mask': input_mask,
            'label_ids': label_ids,
            'token_type_ids': segment_ids,
            }

def create_hf_dataset(data_dir: str,
                      local_tokenizer: PreTrainedTokenizer,
                      labels: List[str],
                      model_type: str,
                      local_max_seq_length: Optional[int] = None,
                      overwrite_cache = False,
                      num_workers=16,
                      split=None,
                      ):
    """create a pytorch dataset using huggingface datasets library
    Returns:
        dataset: instance of torch.utils.data.dataset.Datasets
    """
    global cls_token_at_end
    global cls_token
    global cls_token_segment_id
    global sep_token
    global sep_token_extra
    global pad_on_left, pad_token, pad_token_segment_id, pad_token_label_id
    global sequence_a_segment_id
    global mask_padding_with_zero
    global tokenizer
    global max_seq_length

    max_seq_length = local_max_seq_length
    tokenizer = local_tokenizer
    cls_token_at_end=bool(model_type in ["xlnet"])
    # xlnet has a cls token at the end
    cls_token=tokenizer.cls_token
    cls_token_segment_id=2 if model_type in ["xlnet"] else 0
    sep_token=tokenizer.sep_token
    sep_token_extra=False
    # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
    pad_on_left=bool(tokenizer.padding_side == "left")
    pad_token=tokenizer.pad_token_id
    pad_token_segment_id=tokenizer.pad_token_type_id
    pad_token_label_id=nn.CrossEntropyLoss().ignore_index

    sequence_a_segment_id=0
    mask_padding_with_zero=True

    cache_dir = os.path.join(data_dir, 'hf_cache')
    if overwrite_cache:
        shutil.rmtree(cache_dir)
        os.makedirs(os.path.join(data_dir, 'hf_cache'))

    dataset = load_dataset('json',
                            data_files={
                                'train': os.path.join(data_dir, 'train.seqlabel.jsonl'),
                                'validation': os.path.join(data_dir, 'val.seqlabel.jsonl'),
                                'test': os.path.join(data_dir, 'test.seqlabel.jsonl'),
                            } if split is None else os.path.join(data_dir, f'{split}.seqlabel.jsonl'),
                            cache_dir=os.path.join(data_dir, 'hf_cache'),
                            )

    split_list = ['train', 'validation', 'test']

    if split is None:
        dataset = dataset.map(convert_examples_to_features,
                              num_proc=num_workers,
                              cache_file_names={x: os.path.join(cache_dir, f'cache_mapped_{x}.arrow')
                                  for x in split_list
                                  },
                              )
    else:
        # 'train' is the default split name
        dataset = dataset['train']
        dataset = dataset.map(convert_examples_to_features,
                              num_proc=num_workers,
                              cache_file_name=os.path.join(cache_dir, f'cache_mapped_{split}.arrow'),
                              )


    dataset.set_format(columns=['input_ids',
        'token_type_ids', 'attention_mask', 'label_ids'])

    return dataset


def get_labels(path: str) -> List[str]:
    return ["0", "1"]
