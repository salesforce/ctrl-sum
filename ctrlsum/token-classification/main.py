# coding=utf-8

# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

""" This code is based on
https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import load_dataset
from utils_sum_hf import create_hf_dataset, Split, get_labels


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    threshold: float = field(
        default=0.2, metadata={"help": "prob threshold to select out keywords"}
    )
    # prediction_loss_only: bool = field(
    #     default=False, metadata={"help": "only evaluate loss"}
    # )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    read_data: bool = field(
        default=False, metadata={"help": "read data to cache on one device"}
    )
    eval_split: str = field(
        default='test',
        metadata={"help": "run prediction on the specified split"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    labels = ["0", "1"]
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    hf_dataset = create_hf_dataset(
                    data_dir=data_args.data_dir,
                    local_tokenizer=tokenizer,
                    labels=labels,
                    model_type=config.model_type,
                    local_max_seq_length=data_args.max_seq_length,
                    overwrite_cache=data_args.overwrite_cache,
                )


    if training_args.do_train:
        hf_dataset = create_hf_dataset(
                        data_dir=data_args.data_dir,
                        local_tokenizer=tokenizer,
                        labels=labels,
                        model_type=config.model_type,
                        local_max_seq_length=data_args.max_seq_length,
                        overwrite_cache=data_args.overwrite_cache,
                    )

        if data_args.read_data:
            return
    # Get datasets
    train_dataset = hf_dataset['train'] if training_args.do_train else None
    # train_dataset = hf_dataset['validation'] if training_args.do_eval else None
    eval_dataset = hf_dataset['validation'] if training_args.do_eval else None

    # return

    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        label2id={label: i for i, label in enumerate(labels)}
        preds = np.argmax(predictions, axis=2)
        predictions = scipy.special.softmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]
        preds_prob_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])
                    preds_prob_list[i].append(predictions[i][j][label2id['1']])

        return preds_list, out_label_list, preds_prob_list

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list, _ = align_predictions(p.predictions, p.label_ids)
        out_label_list = sum(out_label_list, [])
        preds_list = sum(preds_list, [])
        return {
            "precision": precision_score(out_label_list, preds_list, pos_label='1', average='binary'),
            "recall": recall_score(out_label_list, preds_list, pos_label='1', average='binary'),
            "f1": f1_score(out_label_list, preds_list, pos_label='1', average='binary'),
        }

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=None,
        # prediction_loss_only=data_args.prediction_loss_only if training_args.do_train else False,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            results.update(result)

    # Predict
    if training_args.do_predict:
        # split = 'test' if data_args.eval_split == 'test' else 'valeval'
        split = data_args.eval_split
        hf_dataset = create_hf_dataset(
                        data_dir=data_args.data_dir,
                        local_tokenizer=tokenizer,
                        labels=labels,
                        model_type=config.model_type,
                        local_max_seq_length=data_args.max_seq_length,
                        overwrite_cache=data_args.overwrite_cache,
                        split=split,
                    )
        test_dataset = hf_dataset

        predictions, label_ids, metrics = trainer.predict(test_dataset)
        preds_list, _, preds_prob_list = align_predictions(predictions, label_ids)

        output_test_results_file = os.path.join(training_args.output_dir, f"{data_args.eval_split}_results.txt")
        if trainer.is_world_master():
            with open(output_test_results_file, "w") as writer:
                for key, value in metrics.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))


        output_test_predictions_file = os.path.join(training_args.output_dir, f"{data_args.eval_split}_predictions.txt")
        if trainer.is_world_master():
            # Save predictions
            # split = Split.test if data_args.eval_split == 'test' else Split.dev
            # test_examples = read_examples_from_file(data_args.data_dir, split)
            test_examples = load_dataset('json',
                                          data_files=os.path.join(data_args.data_dir, f'{split}.seqlabel.jsonl'),
                                          cache_dir=os.path.join(data_args.data_dir, 'hf_cache'))

            if 'train' in test_examples:
                test_examples = test_examples['train']
            with open(output_test_predictions_file, "w") as writer:
                assert len(test_examples) == len(preds_prob_list)
                for line_s, line_t in zip(test_examples, preds_prob_list):
                    # threshold = min(max(line_t), data_args.threshold)
                    for tok_s, pred in zip(line_s['tokens'], line_t):
                        writer.write('{}:{:.3f} '.format(tok_s, pred))
                    writer.write('\n')
                # example_id = 0
                # for line in f:
                #     if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                #         writer.write(line)
                #         if not preds_list[example_id]:
                #             example_id += 1
                #     elif preds_list[example_id]:
                #         output_line = line.split()[0] + " " + preds_list[example_id].pop(0) + "\n"
                #         writer.write(output_line)
                #     else:
                #         logger.warning(
                #             "Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0]
                #         )

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
