#! /usr/bin/env python

# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax, BeamableMM, GradMultiply, LearnedPositionalEmbedding,
    LinearizedConvolution,
)


# archiecture used in Fan 17
# https://arxiv.org/pdf/1711.05217.pdf
@register_model_architecture('fconv', 'fconv_fan')
def fconv_fan(args):
    args.dropout = getattr(args, 'dropout', 0.2)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 340)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(512, 3)] * 8')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 340)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(512, 3)] * 8')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 340)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    args.share_input_output_embed = getattr(args, 'share_input_output_embed', True)

