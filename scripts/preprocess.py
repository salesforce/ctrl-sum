import os
import argparse
import random
import pickle
import json
import spacy
import re
import sys
import subprocess
import numpy as np
import pandas as pd

import stanza

from typing import List, Dict, Any, Optional, Tuple
from collections import (
    defaultdict,
    OrderedDict,
    namedtuple,
    Counter,
)

from multiprocessing import Pool

from spacy.lang.en import English
from spacy.tokens import Token, Doc
from spacy.tokenizer import Tokenizer

def tokenize(src_file: str,
             tgt_file: str,
             tokenizer: Tokenizer,
             split: str,
             annotator: Optional[str] = None,
             batch_size=100,
             max_position=1024,
             max_tgt_position=256,
             save_to_file=True,
             datadir: Optional[str] = None) -> Dict:
    """perform tokenization and sentence segmentation,
    return results or save results to a pickle file
    (if save_to_file is True).

    Args:
        src_file: the summarization source file
        tgt_file: the summarization target file
        tokenizer: the spacy tokenizer
        split: split name
        annotator: some annotation names to disambiguate saved files
            (only used when save_to_file is True)
        max_position: the maximum length of the source file (before tokenization),
            the source will be truncated automatically
        max_tgt_position: the maximum length of the target file (before tokenization),
            the target will be truncated automatically
        save_to_file: whether to save the tokenization results into a pickle file
        datadir: the dataset directory (only used when save_to_file is True)

    Returns:
        a dictionary that contains the tokenized spacy.tokens.Doc objects
            for every source and target example
    """
    def tokenize_batch(batched, data):
        src_docs = tokenizer.pipe([x[1] for x in batched], batch_size=batch_size)
        tgt_docs = tokenizer.pipe([x[2] for x in batched], batch_size=batch_size)
        id_list = [x[0] for x in batched]
        for id_, src_doc, tgt_doc in zip(id_list, src_docs, tgt_docs):
            assert id_ not in data
            data[id_] = {'id': id_,
                       'src_doc': src_doc.to_bytes() if save_to_file else src_doc,
                       'tgt_doc': tgt_doc.to_bytes() if save_to_file else tgt_doc,
                       }
    def truncate(x, max_len):
        x_s = x.rstrip().split()
        max_len = min(len(x_s), max_len)
        return ' '.join(x_s[:max_len])

    data = {}
    with open(src_file) as fsrc, open(tgt_file) as ftgt:
        batched = []
        for i, (src_l, tgt_l) in enumerate(zip(fsrc, ftgt)):
            batched.append((i, truncate(src_l, max_position), truncate(tgt_l, max_tgt_position)))
            if (i + 1) % batch_size == 0:
                tokenize_batch(batched, data)
                batched = []

            if i % 1000 == 0:
                print("processed {} lines".format(i))

    if len(batched) > 0:
        tokenize_batch(batched, data)
        batched = []

    if save_to_file:
        with(open(f'{datadir}/{split}.{annotator}.pickle', 'wb')) as outfile:
            pickle.dump(data, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    return data

def get_tokens(text, tokens):
    """get list of tokenized text
    """

    # remove the linebreak symbol
    return [text[tok['start']:tok['end']] for tok in tokens[:-1]]


def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
      n: which n-grams to calculate
      text: An array of tokens
    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def _greedy_selection(doc_sent_list: List[List[str]],
                     abstract_sent_list: List[List[str]],
                     summary_size: int) -> List[int]:
    """select sentences from the source to maximum its ROUGE scores with the oracle summries.
    Borrowed from BertSum: https://github.com/nlpyang/BertSum/blob/9aa6ab84fa/src/prepro/data_builder.py.
    we select candidate sentences to maximum ROUGE-Recall scores.

    Args:
        doc_sent_list: the source list of sentences
        abstract_sent_list: the target list of sentences
        summary_size: the number of maximum source sentences that can be selected

    Returns:
        the sorted id of selected sentences
    """
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    # Following is from BertSum to maximum rouge score w.r.t. the whole summary
    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))

            # use recall score
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['r']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['r']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            # return selected
            return sorted(selected)
            # maybe should be sorted(selected)
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def oracle_sent(split,
                annotator: Optional[str] = None,
                summary_size=3,
                data: Optional[Dict] = None,
                save_to_file=True,
                datadir=None) -> Dict:
    """The sentence selection step in the pipeline. This step is after the tokenization step
    This function greedily selects source sentences to maximize the ROUGE scores with the
    summaries.

    Args:
        split: the split name
        annotator: the annotation name
        summary_size: the maximum number of selected source sentences
        data: the tokenized data dict from last step in the pipeline
        save_to_file: whether to save the tokenization results into a pickle file
        datadir: the dataset directory (only used when save_to_file is True)

    Returns:
        a dictionary that contains all the preprocessing results until this stepss
    """
    nlp = English()

    # if test_extract:
    #     fout_src = open('{}.extsents'.format(split) ,'w')
    #     fout_tgt = open('{}.extsents.gold'.format(split) ,'w')
    from_file = False
    if data is None:
        with open('{}.tok.pickle'.format(split), 'rb') as fin:
            data = pickle.load(fin)
        from_file = True

    new_data = {}
    print('finish loading pickle data from {}'.format(split))

    for i in range(len(data)):
        k, v = i, data[i]
        if i % 1000 == 0:
            print("processed {} examples".format(i))
        src_doc = Doc(nlp.vocab).from_bytes(v['src_doc']) if from_file else v['src_doc']
        tgt_doc = Doc(nlp.vocab).from_bytes(v['tgt_doc']) if from_file else v['tgt_doc']
        doc_sent_list = [[token.text for token in sent] for sent in src_doc.sents]
        abs_sent_list = [[token.text for token in sent] for sent in tgt_doc.sents]
        # selected = greedy_selection(doc_sent_list, abs_sent_list, summary_size)
        # summary size equal to the gold
        selected = _greedy_selection(doc_sent_list, abs_sent_list, max(summary_size, len(abs_sent_list)))
        data[k].update({'oracle_sents': selected})

        # debug purpose
        # if test_extract:
        #     ext = sum([doc_sent_list[s] for s in selected], [])
        #     abs = sum(abs_sent_list, [])
        #     fout_src.write('{}\n'.format(' '.join(ext)))
        #     fout_tgt.write('{}\n'.format(' '.join(abs)))
    if save_to_file:
        with(open(f'{datadir}/{split}.{annotator}.pickle', 'wb')) as outfile:
            pickle.dump(data, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    return data

def write_ext_sent(split, datadir):
    """load the preprocessing results from picle file and write the
    selected sentences into text files
    """

    prefix = f'{datadir}/{split}'
    nlp = English()
    with open(f'{prefix}.oracle_ext.pickle', 'rb') as fin, \
         open(f'{prefix}.extsents', 'w') as fout_src, \
         open(f'{prefix}.extsents.gold', 'w') as fout_tgt:
         data = pickle.load(fin)
         print('finish loading pickle data from {}'.format(split))

         for i in range(len(data)):
            k, v = i, data[i]
            if i % 1000 == 0:
                print("processed {} examples".format(i))

            src_doc = Doc(nlp.vocab).from_bytes(v['src_doc'])
            tgt_doc = Doc(nlp.vocab).from_bytes(v['tgt_doc'])
            doc_sent_list = [[token.text for token in sent] for sent in src_doc.sents]
            ext = sum([doc_sent_list[s] for s in v['oracle_sents']], [])
            abs = [tok.text for tok in tgt_doc]
            fout_src.write('{}\n'.format(' '.join(ext)))
            fout_tgt.write('{}\n'.format(' '.join(abs)))

def _extract_word(p_text: List[str],
                  e_text: List[str],
                  p_token: List[spacy.tokens.Token],
                  e_token: List[spacy.tokens.Token]) -> List[Tuple[int, int]]:
    """obtain oracle keywords given source and target sentences
    Args:
        p_text: the source word list
        e_text: the target word list
        p_token: the spacy.Token list corresponding to p_text
        e_token: the spacy.Token list corresponding to e_text

    Returns:
        a list of tuples where the first item is the keyword id in the source document,
            while the second item is the keyword id in the target document
    """



    res = []
    # modified based on
    # https://github.com/sebastianGehrmann/bottom-up-summary/blob/master/preprocess_copy.py
    # tsplit = t.split()
    def getsubidx(x, y):
        if len(y) == 0:
            return None

        l1, l2 = len(x), len(y)
        for i in range(l1):
            if x[i:i+l2] == y:
                return i

        return None
    startix = 0
    endix = 1
    matches = []
    src_set = set()
    tgt_set = set()
    while endix <= len(p_text):
        # last check is to make sure that phrases at end can be copied
        tgt_idx = getsubidx(e_text, p_text[startix: endix])
        if tgt_idx is not None and endix <= len(p_text):
            endix += 1
        else:
            # only phrases, not words
            # uncomment the -1 if you only want phrases > len 1
            if startix >= endix-1:
                endix += 1
            else:
                # restrict to not select single stop word separately
                if not (endix-1 == startix + 1 and p_token[startix].is_stop):
                    for offset, loc in enumerate(range(startix, endix-1)):
                        if loc not in src_set and (prev_idx + offset) not in tgt_set \
                                and not p_token[loc].is_stop:
                            res.append((p_token[loc].i, e_token[prev_idx + offset].i))
                            src_set.update([loc])
                            tgt_set.update([prev_idx + offset])
                #endix += 1
            startix = endix - 1

        prev_idx = tgt_idx

    # deal with the corner case matching to the end
    if endix-1 > startix and not (endix-1 == startix + 1 and p_token[startix].is_stop):
        for offset, loc in enumerate(range(startix, endix-1)):
            if loc not in src_set and (prev_idx + offset) not in tgt_set \
                    and not p_token[loc].is_stop:
                res.append((p_token[loc].i, e_token[prev_idx + offset].i))
                src_set.update([loc])
                tgt_set.update([prev_idx + offset])

    return res

def oracle_keyword(split,
                   annotator=None,
                   data=None,
                   save_to_file=True,
                   datadir=None,
                   ):
    """the keyword extraction step in the preprocessing pipeline. This step is
    after the oracle_sent step

    Args:
        split: the split name
        annotator: the annotation name
        data: the tokenized data dict from last step in the pipeline
        save_to_file: whether to save the tokenization results into a pickle file
        datadir: the dataset directory (only used when save_to_file is True)

    Returns:
        a dictionary that contains all the preprocessing results until this steps
    """
    nlp = English()
    # vocab = defaultdict(lambda: len(vocab))
    from_file = False
    if data is None:
        with open(f'{datadir}/{split}.oracle_ext.pickle', 'rb') as fin:
             data = pickle.load(fin)
             print('finish loading pickle data from {}'.format(split))
        from_file = True

    cnt = 0
    for i in range(len(data)):
        k, v = i, data[i]
        if i % 1000 == 0:
            print("processed {} examples".format(i))

        src_doc = Doc(nlp.vocab).from_bytes(v['src_doc']) if from_file else v['src_doc']
        tgt_doc = Doc(nlp.vocab).from_bytes(v['tgt_doc']) if from_file else v['tgt_doc']
        doc_sent_list = [[token for token in sent] for sent in src_doc.sents]
        ext_sents = sum([doc_sent_list[s] for s in v['oracle_sents']], [])
        ext_sents = [tok for tok in ext_sents if not tok.is_punct]
        tgt_tok = [tok for tok in tgt_doc if not tok.is_punct]
        abs = [tok.text for tok in tgt_tok]
        # print(i)
        ext_sents_text = [x.text for x in ext_sents]
        # ext_sents_id = [vocab[x] for x in ext_sents_text]
        # abs_id = [vocab[x] for x in abs]
        selected = _extract_word(ext_sents_text, abs, ext_sents, tgt_tok)
        for (src_loc, tgt_loc) in selected:
            assert src_doc[src_loc].text == tgt_doc[tgt_loc].text

        data[k].update({'oracle_tok': selected})

    if save_to_file:
        with(open(f'{datadir}/{split}.{annotator}.pickle', 'wb')) as outfile:
            pickle.dump(data, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    return data

def write_ext_word(split,
                   datadir,
                   sent_separator: str,
                   sampling=True,
                   data: Optional[Dict] = None,
                   suffix='',
                   ):
    """load the preprocessing results from picle file and write the
    extracted keywords into text files. The keywords are separated
    with a special token in terms of their corresponding source sentences,
    and follow their orders in the source

    Args:
        split: the split name
        sent_separator: the special token to separate the keywords that
            appear in different source sentences
        sampling: perform keyword dropout if True
        data: the preprocessed data in the current step
        suffix: the saved file suffix to disambiguate
        datadir: the dataset directory
    """

    prefix = f'{datadir}/{split}'
    nlp = English()
    outpath = f'{prefix}.oracleword{suffix}' if sampling else f'{prefix}.oraclewordns{suffix}'

    from_file = False
    if data is None:
        with open(f'{prefix}.oracle_word.pickle', 'rb') as fin:
            data = pickle.load(fin)
        print('finish loading pickle data from {}'.format(split))
        from_file = True

    with open(outpath, 'w') as fout_src:
        for i in range(len(data)):
            k, v = i, data[i]
            if i % 1000 == 0:
                print("processed {} examples".format(i))

            if sent_separator is not None:
                src_doc = Doc(nlp.vocab).from_bytes(v['src_doc']) if from_file else v['src_doc']
            tgt_doc = Doc(nlp.vocab).from_bytes(v['tgt_doc']) if from_file else v['tgt_doc']
            abs = [tok.text for tok in tgt_doc]

            tok_order_tgt = sorted(v['oracle_tok'], key=lambda x: x[1])
            if len(tok_order_tgt) > 0:
                cur_char = tgt_doc[tok_order_tgt[0][1]].sent.start_char

            ext_before = [tgt_doc[s[1]] for s in tok_order_tgt]

            tok_separate_sent = []
            new_sent = []
            for tok in tok_order_tgt:
                if tgt_doc[tok[1]].sent.start_char != cur_char:
                    tok_separate_sent.append(new_sent)
                    new_sent = [tok]
                    cur_char = tgt_doc[tok[1]].sent.start_char
                else:
                    new_sent.append(tok)

            if new_sent != []:
                tok_separate_sent.append(new_sent)

            # randomly sample a subset of keywords from each target sentence
            if sampling:
                if len(tok_separate_sent) > 0:
                    new_sample_tok = [random.sample(sent, random.sample(range(1, len(sent)+1), 1)[0]) for sent in tok_separate_sent]
                else:
                    new_sample_tok = []
                new_sample_tok = sum(new_sample_tok, [])
                ext_after = [tgt_doc[s[1]] for s in new_sample_tok]
                oracle_tok = sorted(new_sample_tok, key=lambda x: x[0])
            else:
                # keywords follow the order of them in the src
                oracle_tok = sorted(v['oracle_tok'], key=lambda x: x[0])

            if sent_separator is None:
                ext = [tgt_doc[s[1]].text for s in oracle_tok]
                fout_src.write('{}\n'.format(' '.join(ext)))
            else:
                ext = [src_doc[s[0]] for s in oracle_tok]
                if len(ext) > 0:
                    cur_start_char = ext[0].sent.start_char

                content = []
                for tok in ext:
                    assert tok.sent.start_char >= cur_start_char
                    if tok.sent.start_char != cur_start_char:
                        content.append(sent_separator)
                        cur_start_char = tok.sent.start_char
                    content.append(tok.text)
                fout_src.write('{}\n'.format(' '.join(content)))

    return outpath


def paste(split, src, datadir, key='oracleword', remove_separator=False):
    """paste the keywords file and summarization source file together:
    the summarization source document is prepended with the keywords, with
    a separate token '=>'

    Args:
        split: the split name
        src: the source suffix
        datadir: the dataset directory
        key: the keywords suffix
        remove_separator: remove the sentence separator ('|') during paste
    """

    prefix = f'{datadir}/{split}'
    with open(f'{prefix}.{key}', 'r') as fkey, \
         open(f'{prefix}.{src}', 'r') as fsrc, \
         open(f'{prefix}.{key}{src}', 'w') as fout:
             for line_k, line_s in zip(fkey, fsrc):
                 if remove_separator:
                     line_k = ' '.join(line_k.rstrip('| \n').split(' | '))
                 fout.write('{} => {}\n'.format(line_k.rstrip('| \n'), line_s.rstrip()))

def prepare_tag(split,
                src,
                datadir,
                eval=False,
                max_len=512,
                stride=300,
                data=None,
                suffix='',
                offset=0,
                jsonl=True):
    """prepare formated data for the keywords sequence labeling task. This function would
    performs a sliding window method which separates one long document into several spans
    so that the model is able to deal with long documents

    Args:
        split: the split name
        src: the source suffix
        datadir: the dataset directory
        eval: whether this is an evaluation dataset. During training we only include spans
            with keywords but we must include every span at test time
        max_len: the maximum length of each span
        stride: the stride size of the sliding window
        data: the preprocessed data dict from the pipeline
        suffix: saved file suffix to disambiguate
        offset: the id offset for example id. This is for parallel processing purpose
        jsonl: save file into a json file where each line is an examples
    """

    # label the whole document
    def _check_is_max_context(doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token.
        from: https://github.com/huggingface/transformers/blob/77abd1e79f/templates/adding_a_new_example_script/utils_xxx.py

        """

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index

    nlp = English()

    prefix = f'{datadir}/{split}'
    from_file = False
    if data is None:
        with open(f'{prefix}.oracle_word.pickle', 'rb') as fin:
            data = pickle.load(fin)
        from_file = True

    json_suffix = '.jsonl' if jsonl else ''
    if split == 'val' and eval:
        outpath = f'{datadir}/valeval.seqlabel{json_suffix}{suffix}'
    else:
        outpath = f'{prefix}.seqlabel{json_suffix}{suffix}'
    with open(outpath, 'w') as fout:
        for example_index in range(len(data)):
            k, v = example_index, data[example_index]
            if example_index % 1000 == 0:
                print("processed {} examples".format(example_index))

            src_doc = Doc(nlp.vocab).from_bytes(v['src_doc']) if from_file else v['src_doc']

            # sliding window approach to deal with long input
            _DocSpan = namedtuple("DocSpan", ["start", "length"])  # pylint: disable=invali
            doc_spans = []
            start_offset = 0

            label_list = [0] * len(src_doc)
            for s in v['oracle_tok']:
                label_list[s[0]] = 1

            while start_offset < len(src_doc):
                length = len(src_doc) - start_offset
                if length > max_len:
                    length = max_len
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(src_doc):
                    break
                start_offset += min(length, stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                # only span with keywords is used for training
                if split != 'test' and not eval and sum(label_list[doc_span.start:doc_span.start + length]) == 0:
                    continue
                if jsonl:
                    json_record = {'id': example_index + offset, 'tokens': [], 'labels': [], 'max_context': []}
                for i in range(doc_span.length):
                    pos = doc_span.start + i
                    is_max_context = _check_is_max_context(doc_spans, doc_span_index, pos)
                    if jsonl:
                        json_record['tokens'].append(src_doc[pos].text)
                        json_record['labels'].append(label_list[pos])
                        json_record['max_context'].append(int(is_max_context))
                    else:
                        fout.write('{} {} {:d} {:d}\n'.format(src_doc[pos].text, label_list[pos],
                            is_max_context, example_index + offset))

                if jsonl:
                    fout.write(json.dumps(json_record, ensure_ascii=False))
                fout.write('\n')

        return outpath


def process_tagger_prediction(split, datadir, tag_pred: str, threshold: float, summary_len=10, minimum_word=1,
        maximum_word=25, outfix='default', extsent=False, weight_sent=False,
        sent_separator=True):
    """post-process the prediction from tagging model and select keywords for later decoding

    Args:
        split: the split name
        datadir: the dataset directory
        tag_pred: the prediction file from the tagger
        threshold: the keyword selection confidence threshold
        summary_len: the maximum number of sentences that can be selected
            from which to pick the keywords
        minimum_word: the minimum number of keywords
        maximum_word: the maximum number of keywords
        outfix: the output file suffix to disambiguate
        extsent: whether to write out intermediate extracted sentences
        weight_sent: weight each word with the average confidence of the sentence where
            the word occurs
        sent_separator: whether to use a sentence separator token for the keywords
    """
    data_pred = {}
    cur_example = 0
    local_index = Counter()
    orig_data = {}
    prefix = f'{datadir}/{split}'
    with open(f'{prefix}.seqlabel.jsonl') as finput, \
         open(tag_pred) as fpred:
        for cnt, (line_s, line_p) in enumerate(zip(finput, fpred)):
            if cnt % 1000 == 0:
                print(f"read {cnt} examples")
                # if cnt == 1000:
                #     break
            src_dict = json.loads(line_s.rstrip())
            pred = line_p.strip()
            pred = pred.split()
            pred = [(':'.join(x.split(':')[:-1]), float(x.split(':')[-1])) for x in pred]
            input_ = []
            for i in range(len(src_dict['tokens'])):
                text, label, valid, example_id = \
                        src_dict['tokens'][i], src_dict['labels'][i], src_dict['max_context'][i], src_dict['id']
                valid, example_id = int(valid), int(example_id)
                if valid:
                    local_index[example_id] += 1
                    if example_id not in orig_data:
                        orig_data[example_id] = []

                    orig_data[example_id].append(text)

                if i >= len(pred):
                    continue

                assert text == pred[i][0]

                # only use the features with maximal context
                if valid:
                    if example_id not in data_pred:
                        data_pred[example_id] = {}

                    data_pred[example_id][local_index[example_id]-1] = pred[i]


    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")

    suffix = f'ts{threshold}.mw{maximum_word},sumlen{summary_len}.{outfix}.predword'
    with open(f'{prefix}.{suffix}', 'w') as fout:
        for i in sorted(data_pred.keys()):
            tok_prob = data_pred[i]
            words = orig_data[i]
            # k, v = i, data[i]
            spaces = [True] * (len(words) - 1) + [False]
            src_doc = Doc(nlp.vocab, words=words, spaces=spaces)
            src_doc = sentencizer(src_doc)
            if i % 1000 == 0:
                print("processed {} examples".format(i))
            # src_doc = Doc(nlp.vocab).from_bytes(v['src_doc'])

            doc_score = []
            sent_list = list(src_doc.sents)
            sent2score = {}
            for sent in sent_list:
                sent_score = []
                for tok in sent:
                    if tok.i in tok_prob:
                        assert tok.text == tok_prob[tok.i][0]
                        sent_score.append(tok_prob[tok.i][1])
                if len(sent_score) > 0:
                    score = np.mean(sent_score)
                else:
                    score = 0
                doc_score.append(score)
                sent2score[sent.start_char] = score


            order = np.argsort(-np.array(doc_score))
            sent_num = min(len(order), summary_len)
            order = order[:sent_num]

            if extsent:
                tokens = [tok.text for sent_id in sorted(order) for tok in sent_list[sent_id]]
                fout.write('{}\n'.format(' '.join(tokens)))
                continue

            cand_list = []
            select_list = []
            for k, sent_id in enumerate(order):
                for tok in sent_list[sent_id]:
                    if tok.i in tok_prob:
                        cand_list.append((tok, tok_prob[tok.i][1], tok_prob[tok.i][1] * sent2score[tok.sent.start_char]))
                        # if tok_prob[tok.i][1] > threshold:
                        #     # select_list.append((tok, tok_prob[tok.i][1], tok_prob[tok.i][1] * sent2score[tok.sent.start_char]))
                        #     select_list.append((tok, tok_prob[tok.i][1], tok_prob[tok.i][1]))

            compare_key = 2 if weight_sent else 1
            select_list = sorted(cand_list, key=lambda x: -x[compare_key])[:min(len(cand_list), maximum_word)]
            ts = min(threshold, select_list[minimum_word-1][compare_key])
            new_list = []
            record_set = set()
            for x in select_list:
                if x[compare_key] >= ts and x[0].text not in record_set:
                    new_list.append(x)
                    record_set.update([x[0].text])

            select_list = new_list
            assert len(select_list) > 0
            # write keywords in order
            select_list = sorted(select_list, key=lambda x: x[0].i)
            cur_start_char = select_list[0][0].sent.start_char
            for tok in select_list:
                tok = tok[0]
                if tok.sent.start_char != cur_start_char:
                    if sent_separator:
                        fout.write('| ')
                    cur_start_char = tok.sent.start_char
                fout.write('{} '.format(tok.text))

            fout.write('\n')

    return suffix


def entity_tag(split, src, datadir, filter_user=True):
    """tag entities in the source document and generate a keyword file
    {split}.entitywords{[filter]} that uses the entities as keywords

    Args:
        split: the split name
        src: the source document suffix
        datadir: the dataset directory
        filter_user: if True, only select entites given in the `user_ent_type`
            variable. This is to better simulate user perferences since many
            entity types are unlikely to be specified by users
    """
    # spacy.prefer_gpu()
    # nlp = spacy.load("en_core_web_sm", disable=['tagger', 'parser'])
    user_ent_type = set(['EVENT', 'FAC', 'GPE', 'LAW', 'NORP', 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART'])
    batch_size = 64
    nlp = stanza.Pipeline(lang='en', processors='tokenize,ner', ner_batch_size=batch_size, tokenize_no_ssplit=True)
    batches = []
    no_entity = 0
    prefix = f'{datadir}/{split}'
    postfix = 'filter' if filter_user else ''
    with open(f'{prefix}.{src}') as fin, \
            open(f'{prefix}.entitywords{postfix}', 'w') as fout:
        for i, line in enumerate(fin):
            batches.append(line.strip())

            if (i+1) % batch_size == 0:
                for doc in nlp('\n\n'.join(batches)).sentences:
                    flag = False
                    for ent in doc.ents:
                        if not filter_user or ent.type in user_ent_type:
                            flag = True
                            fout.write(ent.text + ' ')
                    # for tok in doc:
                    #     if tok.ent_type_ != '':
                    #         flag = True
                    #         fout.write(tok.text + ' ')
                    fout.write('\n')
                    if not flag:
                        no_entity += 1

                batches = []

            if i % 1000 == 0:
                print('processing {} lines'.format(i))


        if batches != []:
            for doc in nlp('\n\n'.join(batches)).sentences:
                flag = False
                for ent in doc.ents:
                    if not filter_user or ent.type in user_ent_type:
                        flag = True
                        fout.write(ent.text + ' ')
                # for tok in doc:
                #     if tok.ent_type_ != '':
                #         flag = True
                #         fout.write(tok.text + ' ')
                fout.write('\n')
                if not flag:
                    no_entity += 1

    print('{} examples without entity detected'.format(no_entity))

def entity_random(split, src, datadir, nsample=100, human_study=False):
    """randomly sample `nsample` documents and repeatedly select every entity as keyword.
    If `human_study` is False, there are totally 4 files to be generated by running this function:
    `{split}.{nsample}samplelead3.{entity|src}` represents the corresponding
    entity-source files for lead3 entities, and  `{split}.{nsample}samplefull.{entity|src}`
    are for entities in the full article. If `human_study` is True, for each document select one
    important entity and one unimportant entity, there are 2 files to be generated:
    `{split}.entityhumanstudy{src}` and `{split}.entityhumanstudy{tgt}`

     Args:
        split: the split name
        datadir: the dataset directory
        nsample: the number of source samples to draw
        human_study: if true, for each document select one important entity
            and one unimportant entity
    """
    # spacy.prefer_gpu()
    # nlp = spacy.load("en_core_web_sm", disable=['tagger', 'parser'])
    user_ent_type = set(['EVENT', 'FAC', 'GPE', 'LAW', 'NORP', 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART'])
    batch_size = 64

    # to avoid extracting entities from truncated part later, since we are doing
    # subword encoding later which will increase the length
    # to compare with Fan et al. 18, they truncate to 400 tokens to evaluate success rate
    max_len = 500
    nlp = stanza.Pipeline(lang='en', processors='tokenize,ner', ner_batch_size=batch_size)
    no_entity = 0

    prefix = f'{datadir}/{split}'
    with open(f'{prefix}.{src}') as fin:
        data_src = fin.readlines()
    indexes = random.sample(range(len(data_src)), nsample)
    data = [data_src[x] for x in indexes]
    data = [x.strip() for x in data]
    data = [' '.join(x.split()[:min(max_len, len(x.split()))]) for x in data]

    if not human_study:
        with open(f'{prefix}.{nsample}samplelead3.entity', 'w') as fout_e_lead, \
             open(f'{prefix}.{nsample}samplefull.entity', 'w') as fout_e_full, \
             open(f'{prefix}.{nsample}samplelead3.{src}', 'w') as fout_s_lead, \
             open(f'{prefix}.{nsample}samplefull.{src}', 'w') as fout_s_full:
            for ind, x in zip(indexes, data):
                doc = nlp(x)
                record = set() # avoid duplication
                for i, sent in enumerate(doc.sentences):
                    for ent in sent.ents:
                        ent_text = ent.text.strip("'")
                        ent_text = ent.text.strip('"')
                        if ent.type in user_ent_type and ent_text not in record:
                            record.update([ent_text])
                            if i <= 2:
                                fout_e_lead.write(f'{ent_text}\n')
                                fout_s_lead.write(f'{data_src[ind].strip()}\n')

                            fout_e_full.write(f'{ent_text}\n')
                            fout_s_full.write(f'{data_src[ind].strip()}\n')
    else:
        smaller_user_ent_type = set(['EVENT', 'LAW', 'NORP', 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART'])
        with open('{}.{}'.format(split, tgt)) as fin:
            data_tgt = fin.readlines()

        # double the required size, and after generation selects nsample successful
        # control exampels for human study
        indexes = random.sample(range(len(data_src)), 2 * nsample)
        data_tgt = [data_tgt[x] for x in indexes]
        data_tgt = [x.strip() for x in data_tgt]

        data_src = [data_src[x] for x in indexes]
        data_src = [x.strip() for x in data_src]
        data_src = [' '.join(x.split()[:min(max_len, len(x.split()))]) for x in data_src]

        with open(f'{prefix}.entityhumanstudy{src}', 'w') as fout_src, \
                open(f'{prefix}.entityhumanstudy{tgt}', 'w') as fout_tgt:
            for doc_src, doc_tgt in zip(nlp('\n\n'.join(data_src)).sentences,
                    nlp('\n\n'.join(data_tgt)).sentences):
                if len(doc_tgt.ents) == 0 or len(doc_src.ents) < 2:
                    continue
                # randomly select one important entity
                visit_list = [True] * len(doc_tgt.ents)
                important_ent = None
                while True:
                    sample_index = random.sample(range(len(visit_list)), 1)[0]
                    visit_list[sample_index] = False
                    if doc_tgt.ents[sample_index].type in smaller_user_ent_type and \
                            doc_tgt.ents[sample_index].text in doc_src.text:
                        important_ent = doc_tgt.ents[sample_index].text
                        break

                    if sum(visit_list) == 0:
                        break

                if important_ent is None:
                    continue

                # unimportant entity not in target and lead-3
                visit_list = [True] * len(doc_src.ents)
                unimportant_ent = None
                tgt_ent_set = set([ent.text for ent in doc_tgt.ents])
                doc_src_tokens = [token.text for token in doc_src.tokens]

                # first three
                lead_3 = ' '.join(doc_src_tokens).split('.')
                if len(lead_3) <= 3:
                    continue
                lead_3 = lead_3[:3]
                lead_3 = ' '.join(' '.join(lead_3).split())

                while True:
                    sample_index = random.sample(range(len(visit_list)), 1)[0]
                    visit_list[sample_index] = False
                    if doc_src.ents[sample_index].type in smaller_user_ent_type and \
                            doc_src.ents[sample_index].text not in doc_tgt.text and \
                            doc_src.ents[sample_index].text not in lead_3:
                        unimportant_ent = doc_src.ents[sample_index].text
                        break

                    if sum(visit_list) == 0:
                        break

                if unimportant_ent is not None:
                    fout_src.write(' {} => {}\n'.format(important_ent, doc_src.text.strip()))
                    fout_tgt.write('{}\n'.format(doc_tgt.text.strip()))

                    fout_src.write(' {} => {}\n'.format(unimportant_ent, doc_src.text.strip()))
                    fout_tgt.write('{}\n'.format(doc_tgt.text.strip()))

def _cluster_length_to_bin(len_list: List[int], num: int) -> List[Tuple[int, int]]:
    """cluster a list of lengths into `num` buckets

    Args:
        len_list: the list of target summary lengths
        num: the number of buckets

    Returns:
        a list of tuples of which each element is the
        [lower bound, upper bound) of the bucket
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

def _length_to_string(length: int, bin_bound: List[Tuple[int, int]]) -> str:
    """convert the length into the id of bucket, return the string of the id

    Args:
        length: the length to be converted
        bin_bound: the bucket denoted by a list of tuples
    """
    flag = False
    for i, bucket in enumerate(bin_bound):
        if length >= bucket[0] and length < bucket[1]:
            id_ = i
            flag = True
            break

    if not flag:
        raise ValueError("didn't find a bucket for length {}".format(length))

    return str(id_)

def prepend_oracle_len(split, datadir, src, tgt,
                len_bin: Optional[List] = None,
                num_bin=5, iterate=False):
    """prepend corresponding oracle length labels to the source, and generate files
    `{split}.len{src}lead`

    Args:
        split: the split name
        datadir: the dataset directory
        src: the source suffix
        tgt: the target suffix
        len_bin: the length bucket. If absent, this function would re-compute
            the length buckets by itself
        num_bin: number of length buckets
        iterate: if True, iterate over all length buckets for every source
    """

    prefix = f'{datadir}/{split}'
    with open(f'{prefix}.{tgt}') as fin:
        length_list = [len(line.strip().split()) for line in fin]

    if len_bin is None:
        len_bin = _cluster_length_to_bin(length_list, num_bin)

    if not iterate:
        with open(f'{prefix}.{src}') as fin, \
            open(f'{prefix}.len{src}lead', 'w') as fout:
            for i, line in enumerate(fin):
                # prepend space for gpt2 encoding
                fout.write(' {} {}\n'.format(_length_to_string(length_list[i], len_bin), line.strip()))
    else:
        with open(f'{prefix}.{src}'.format(split, src)) as fin, \
            open(f'{split}.{tgt}') as ftgt, \
            open(f'{prefix}.iterate.len{src}lead', 'w') as fout, \
            open(f'{split}.iterate.{tgt}', 'w') as fout_tgt:
            for i, (ls, lt) in enumerate(zip(fin, ftgt)):
                for j in range(len(len_bin)):
                    fout.write(f' {j} {ls.strip()}\n')
                    fout_tgt.write(f'{lt.strip()}\n')

    return len_bin

def add_leading_space(path):
    """add a leading space to the file, this is for GPT2 bpe encoding
    """
    with open(path) as fin, \
        open('{}lead'.format(path), 'w') as fout:
        for line in fin:
            fout.write(' {}\n'.format(line.strip()))

def auto_truncate(in_path, out_path, max_len):
    """truncate the input file and write to `out_path`
    """
    with open(in_path) as fin,\
         open(out_path, 'w') as fout:
        for x in fin:
            x_s = x.rstrip().split()
            max_len_s = min(len(x_s), max_len)
            fout.write(' '.join(x_s[:max_len_s]) + '\n')

def add_prefix(split, src, datadir, prefix: str):
    """add a prefix string to every example in the source file.
    Separated with the '=>' token
    """

    datapath = f'{datadir}/{split}'

    with open(f'{datapath}.{src}') as fsrc, \
         open(f'{datapath}.prefix{src}lead', 'w') as fout_src:

        data = fsrc.readlines()
        data = [x.strip() for x in data]

        for line in data:
            fout_src.write(f' {prefix} => {line}\n')

def human_study_csv_entity(datadir, src, tgt, nsample):
    """process model output for the entity control experiment, and
    generate csv files for human study of entity control.
    This function will yield a `human_study_entity.csv` file in the
    `datadir` directory

    Args:
        datadir: the dataset directory
        src: the input src file
        tgt: the model generation file
        nsample: only use first `nsample` examples
    """
    data = []
    data_tmp = {}
    with open(src) as fsrc, \
        open(tgt) as ftgt:
        for i, (lsrc, ltgt) in enumerate(zip(fsrc, ftgt)):
            if i % 2 == 0:
                cur = 'important'
                data_tmp = {}
            else:
                cur = 'unimportant'

            lsrc = lsrc.strip()
            ltgt = ltgt.strip()
            entity = lsrc.split(' => ')[0]
            source = ' => '.join(lsrc.split(' => ')[1:])
            if entity in ltgt and entity.strip() != ltgt.strip():
                data_tmp[cur] = {'id': i, 'entity': entity,
                    'article': source, 'summary': ltgt}

            if i % 2 != 0 and len(data_tmp) == 2:
                assert data_tmp['important']['article'] == data_tmp['unimportant']['article']
                data.append({'id': data_tmp['important']['id'],
                    'article': data_tmp['important']['article'],
                    'important_keywords': data_tmp['important']['entity'],
                    'important_summary': data_tmp['important']['summary'],
                    'unimportant_keywords': data_tmp['unimportant']['entity'],
                    'unimportant_summary': data_tmp['unimportant']['summary']})


    if len(data) < nsample:
        raise ValueError('valid examples are not enough')

    data = data[:nsample]
    data_frame = {key: [] for key in data[0].keys()}
    for d in data:
        for k, v in d.items():
            data_frame[k].append(v)

    order = ['id', 'article', 'important_keywords', 'important_summary',
            'unimportant_keywords', 'unimportant_summary']
    data_frame = OrderedDict(sorted(data_frame.items(), key=lambda t: order.index(t[0])))
    df = pd.DataFrame(data=data_frame)
    df.to_csv('human_study_entity.csv')

def human_study_purpose(reference, nsample=1000):
    """process model generations and generate jsonl files
    for human study of purpose
    Args:
        reference: the reference summary file
    """
    data = []
    nlp = stanza.Pipeline('en', processors='tokenize')

    with open(src) as fsrc:
        raw_data = fsrc.readlines()

    sampled_id = random.sample(range(len(raw_data)), nsample)

    for i in sampled_id:
        text = raw_data[i].strip()
        doc = nlp(text)
        data.append({'id': i,
            'summary': [x.text for x in doc.sentences]})

    with open('human_study_purpose.jsonl', 'w') as fout:
        for record in data:
            fout.write(json.dumps(record, ensure_ascii=False))
            fout.write('\n')


def pipeline(split, args, suffix='', offset=0):
    """the entire preprocessing pipeline consisting of tokenization,
    selecting oracle key sentences, selecting oracle keywords, and preparing
    sequence labeling dataset to tag keywords. This function should be able
    generate all the required data files to train and evaluate the summarization model

    Args:
        split: the split name
        suffix: output file suffix for disambiguation purpose
        offset: mainly for parallel processing purpose
    """
    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)

    # an issue for spacy default tokenization, see http://www.longest.io/2018/01/27/spacy-custom-tokenization.html
    def custom_tokenizer(nlp):
        prefix_re = spacy.util.compile_prefix_regex(nlp.Defaults.prefixes)
        suffix_re = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)
        custom_infixes = ['\.\.\.+', '(?<=[0-9])-(?=[0-9])', '[!&:,()]']
        infix_re = spacy.util.compile_infix_regex(tuple(list(nlp.Defaults.infixes) + custom_infixes))

        tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab,
                                              nlp.Defaults.tokenizer_exceptions,
                                              prefix_search=prefix_re.search,
                                              suffix_search=suffix_re.search,
                                              infix_finditer=infix_re.finditer,
                                              token_match=None)

        return tokenizer

    if args.pretokenize:
        nlp.tokenizer = Tokenizer(nlp.vocab)
    else:
        nlp.tokenizer = custom_tokenizer(nlp)

    print(f"----- tokenize split '{split}' -------")
    data = tokenize(f'{args.datadir}/{split}.{args.src}{suffix}',
        f'{args.datadir}/{split}.{args.tgt}{suffix}', nlp, split,
        max_position=args.max_position,
        max_tgt_position=args.max_tgt_position,
        save_to_file=False,
        datadir=None)

    # print(data[0]['tgt_doc'])

    print(f"----- greedy-selection of sentences split '{split}' --------")
    data = oracle_sent(split, summary_size=args.summary_size, data=data, save_to_file=False)

    print(f"----- extract keywords split '{split}' --------")
    data = oracle_keyword(split, data=data, save_to_file=False)

    # v = data[0]
    # print([v['src_doc'][x[0]] for x in v['oracle_tok']])
    fname_list = []
    print(f"----- write keywords split '{split}' --------")
    fname = write_ext_word(split, args.datadir, args.sent_separator,
            sampling=True, data=data, suffix=suffix)

    fname_list.append(fname)

    fname = write_ext_word(split, args.datadir, args.sent_separator,
            sampling=False, data=data, suffix=suffix)
    fname_list.append(fname)

    print(f"----- prepare tags for seq labeling split '{split}' --------")
    fname = prepare_tag(split, args.src, args.datadir, data=data, suffix=suffix, offset=offset)
    fname_list.append(fname)

    if split == 'val':
        fname = prepare_tag(split, args.src, args.datadir, eval=True,
                suffix=suffix, data=data, offset=offset)
        fname_list.append(fname)

    return fname_list

def calc_len_bin(datadir, split, tgt, num_bin=5, sample=False):
    """Given a target file, compute the length buckets and number
    of keywords correponding to different buckets. If `sample` is True,
    consider counting the number of keywords after keyword dropout

    Returns:
        tuple:
            - a dict mapping bucekt id to number of keywords
            - a list of length buckets
    """
    with open(f'{datadir}/{split}.{tgt}') as fin:
        length_list = [len(line.strip().split()) for line in fin]
    nlp = English()

    num_key_list = []
    input_file = f'{datadir}/{split}.extwordsns' if not sample \
        else f'{datadir}/{split}.extwords'

    with open(input_file) as fin:
        for line in fin:
            l = [x for x in line.strip().split() if x != '|']
            num_key_list.append(len(l))

    assert len(num_key_list) == len(length_list)

    len_bin = _cluster_length_to_bin(length_list, num_bin)

    lenbin2num = {key: [] for key in len_bin}
    for lw, lr in zip(num_key_list, length_list):
        id_ = int(_length_to_string(lr, len_bin))
        lenbin2num[len_bin[id_]].append(lw)

    for key in lenbin2num:
        lenbin2num[key] = round(np.mean(lenbin2num[key]))

    return lenbin2num, len_bin

def get_keyword_len(datadir,
                    split,
                    tgt,
                    tag_pred,
                    lenbin2num,
                    len_bin,
                    iterate=False,
                    src=None):
    """obtain the keywords of each example for length control purpose
    by processing the tagger model predictions. This function needs the
    oracle length of reference summaries to simulate user preference. This
    function will generate a keywords file `{split}.lengthcontrol{suffix}.predwords`

    Args:
        datadir: the dataset directory
        split: the split name
        tgt: the summarization suffix
        tag_pred: the prediction file path generated tagger model
        lenbin2num: the dict mapping bucket to number of keywords
        len_bin: the length buckets
        iterate: If true, then this function iterates over all the lengths instead of
            just the oracle length.
        src: only required when iterate is True
    """
    data_pred = {}
    cur_example = 0
    local_index = Counter()
    orig_data = {}
    suffix = '.iterate' if iterate else ''
    with open(f'{datadir}/{split}.seqlabel.jsonl') as finput, \
         open(tag_pred) as fpred:
        for cnt, (line_s, line_p) in enumerate(zip(finput, fpred)):
            if cnt % 1000 == 0:
                print(f"read {cnt} examples")
            src_dict = json.loads(line_s.rstrip())
            pred = line_p.strip()
            pred = pred.split()
            pred = [(':'.join(x.split(':')[:-1]), float(x.split(':')[-1])) for x in pred]
            input_ = []
            for i in range(len(src_dict['tokens'])):
                text, label, valid, example_id = \
                        src_dict['tokens'][i], src_dict['labels'][i], src_dict['max_context'][i], src_dict['id']
                valid, example_id = int(valid), int(example_id)
                if valid:
                    local_index[example_id] += 1
                    if example_id not in orig_data:
                        orig_data[example_id] = []

                    orig_data[example_id].append(text)

                if i >= len(pred):
                    continue

                assert text == pred[i][0]

                # only use the features with maximal context
                if valid:
                    if example_id not in data_pred:
                        data_pred[example_id] = {}

                    data_pred[example_id][local_index[example_id]-1] = pred[i]

    with open('{}.{}'.format(split, tgt)) as fin:
        length_list = [len(line.strip().split()) for line in fin]

    assert len(length_list) == len(data_pred)

    if iterate:
        with open(f'{datadir}/{split}.{tgt}') as fin, \
             open(f'{datadir}/{split}.iterate.{tgt}', 'w') as fout:
            for line in fin:
                for _ in len_bin:
                    fout.write(f'{line.strip()}\n')

        with open(f'{datadir}/{split}.{src}') as fin, \
             open(f'{datadir}/{split}.iterate.{src}', 'w') as fout:
            for line in fin:
                for _ in len_bin:
                    fout.write(f'{line.strip()}\n')

    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")

    with open(f'{datadir}/{split}.lengthcontrol{suffix}.predwords', 'w') as fout:
        # for i in range(len(data_pred)):
        for k, i in enumerate(sorted(data_pred.keys())):
            tok_prob = data_pred[i]
            words = orig_data[i]
            spaces = [True] * (len(words) - 1) + [False]
            src_doc = Doc(nlp.vocab, words=words, spaces=spaces)
            src_doc = sentencizer(src_doc)
            if k % 1000 == 0:
                print("processed {} examples".format(k))

            cand_list = []
            sent_list = list(src_doc.sents)
            for sent in sent_list:
                for tok in sent:
                    if tok.i in tok_prob:
                        cand_list.append((tok, tok_prob[tok.i][1]))

            if not iterate:
                tgt_len = length_list[k]
                len_bin_bound = len_bin[int(_length_to_string(tgt_len, len_bin))]
                nk = lenbin2num[len_bin_bound]
                nk_list = [nk]
            else:
                nk_list = [lenbin2num[len_bin_bound] for len_bin_bound in len_bin]

            for nk in nk_list:
                select_list = sorted(cand_list, key=lambda x: -x[1])[:min(len(cand_list), nk)]
                assert len(select_list) > 0

                # write keywords in order
                select_list = sorted(select_list, key=lambda x: x[0].i)
                cur_start_char = select_list[0][0].sent.start_char
                for tok in select_list:
                    tok = tok[0]
                    if tok.sent.start_char != cur_start_char:
                        fout.write('| ')
                        cur_start_char = tok.sent.start_char
                    fout.write('{} '.format(tok.text))

                fout.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='various preprocessing for summarization task')
    parser.add_argument('dataset', type=str, help='the dataset name')
    parser.add_argument('--mode', type=str, choices=['truncate', 'process_tagger_prediction',
        'prepend_oracle_len', 'human_study_entity', 'human_study_purpose', 'get_keyword_len',
        'pipeline'],
        help='preprocessing mode. Please see the comments doc of each function for details')


    parser.add_argument('--max-position', type=int, default=1024,
        help='maximum source length')
    parser.add_argument('--max-tgt-position', type=int, default=256,
        help='maximum target length')
    parser.add_argument('--src', type=str, default='source',
        help='source file suffix')
    parser.add_argument('--tgt', type=str, default='target',
        help='target file suffix')
    parser.add_argument('--outfix', type=str, default='default',
        help='output file suffix to disambiguate')
    parser.add_argument('--split', type=str, default=None,
        help='the specific split to preprocess. If None then processing all splits')
    parser.add_argument('--pretokenize', action='store_true', default=False,
        help='whether the input data is already tokenized')
    parser.add_argument('--tag-pred', type=str, default=None,
        help='prediction file from tagger')

    # keyword selection hyperparams
    parser.add_argument('--threshold', type=float, default=0.1,
        help='the confidence threshold to select keywords from tagger output')
    parser.add_argument('--maximum-word', type=int, default=25,
        help='maximum number of keywords for tagger')
    parser.add_argument('--summary-size', type=int, default=3,
        help='maximum number of firstly extracted sentences before keyword extraction')


    parser.add_argument('--sent-separator', type=str, default='|',
        help='if specified, include a sentence separator in the keywords')
    parser.add_argument('--num-workers', type=int, default=20,
            help='number of processes in the pipeline mode, 1 to disable')

    args = parser.parse_args()

    datadir = f'datasets/{args.dataset}'
    args.datadir = datadir

    mode_to_id = {'truncate': 'trunc', 'tokenize': 'tok', 'greedy_selection':
    'oracle_ext', 'align': 'oracle_word'}
    indicator = mode_to_id.get(args.mode, None)

    print("start mode {}".format(args.mode))

    if args.mode == 'truncate':
        split_list = ['val', 'test', 'train'] if args.split is None else [args.split]
        for split in split_list:
            auto_truncate('{}.{}'.format(split, args.src), '{}.{}trunc'.format(split, args.src), args.max_position)
            if not args.disable_tgt:
                auto_truncate('{}.{}'.format(split, args.tgt), '{}.{}trunc'.format(split, args.tgt), args.max_tgt_position)
    elif args.mode == 'process_tagger_prediction':
        split_list = [args.split]
        for split in split_list:
            suffix = process_tagger_prediction(split, datadir,  args.tag_pred, args.threshold,
                summary_len=args.summary_size, maximum_word=args.maximum_word,
                outfix=args.outfix)
            paste(split, args.src, datadir, key=suffix)
            add_leading_space(f'{datadir}/{split}.{suffix}{args.src}')
            os.rename(f'{datadir}/{split}.{suffix}{args.src}lead', f'{datadir}/{split}.{suffix}{args.src}')

    elif args.mode == 'prepend_oracle_len':
        len_bin = prepend_len('train', datadir, args.src, args.tgt, num_bin=5)
        print(f'len_bin: {len_bin}')
        split_list = ['test', 'val'] if args.split is None else [args.split]
        for split in split_list:
            prepend_len(split, datadir, args.src, args.tgt, len_bin=len_bin, iterate=args.length_iterate)
    elif args.mode == 'get_keyword_len':
        # use keywords approach to perform length control
        sample = False
        lenbin2num, len_bin = calc_len_bin(datadir, 'train', args.tgt, num_bin=5, sample=sample)
        print(f'lenbin2num: {lenbin2num}')
        print(f'len_bin: {len_bin}')

        get_keyword_len(args.split, args.tgt, args.tag_pred,
                lenbin2num=lenbin2num, len_bin=len_bin,
                iterate=args.length_iterate, src=args.src)
    elif args.mode == 'human_study_entity':
        human_study_csv_entity(datadir, args.src, args.tgt, nsample=100)
    elif args.mode == 'human_study_purpose':
        human_study_purpose(args.src)
    elif args.mode == 'pipeline':
        split_list = ['val', 'test', 'train'] if args.split is None else args.split.split(',')
        for split in split_list:
            if args.num_workers > 1:
                proc = subprocess.run(['wc', '-l', f'{datadir}/{split}.{args.tgt}'], capture_output=True)
                num_example = int(proc.stdout.decode('utf-8').split()[0])
                bsz = num_example // args.num_workers + 1
                subprocess.run(['split', '-l', f'{bsz}', '-d',
                    f'{datadir}/{split}.{args.src}', f'{datadir}/{split}.{args.src}'])
                subprocess.run(['split', '-l', f'{bsz}', '-d',
                    f'{datadir}/{split}.{args.tgt}', f'{datadir}/{split}.{args.tgt}'])

                with Pool(args.num_workers) as p:
                    file_list = p.starmap(pipeline, [[split, args, f'{x:02d}', x * bsz]
                        for x in range(args.num_workers)])
                    p.close()
                    p.join()

                file_list = file_list[0]
                for f in file_list:
                    with open(f[:-2], 'w'):
                        pass
                    for i in range(args.num_workers):
                        subprocess.run(f'cat {f[:-2]}{i:02d} >> {f[:-2]}', shell=True)
                        subprocess.run(['rm', f'{f[:-2]}{i:02d}'])

                for i in range(args.num_workers):
                    subprocess.run(['rm', f'{datadir}/{split}.{args.src}{i:02d}'])
                    subprocess.run(['rm', f'{datadir}/{split}.{args.tgt}{i:02d}'])

            else:
                pipeline(split, args)

            paste(split, args.src, datadir, key='oracleword')

            auto_truncate(f'{datadir}/{split}.oracleword{args.src}', f'{datadir}/tmp.src', args.max_position)
            auto_truncate(f'{datadir}/{split}.{args.tgt}', f'{datadir}/tmp.tgt', args.max_tgt_position)

            os.rename(f'{datadir}/tmp.src', f'{datadir}/{split}.oracleword{args.src}')
            os.rename(f'{datadir}/tmp.tgt', f'{datadir}/{split}.{args.tgt}')

            add_leading_space(f'{datadir}/{split}.oracleword{args.src}')
            os.rename(f'{datadir}/{split}.oracleword{args.src}lead', f'{datadir}/{split}.oracleword{args.src}')

            if split == 'test':
                paste(split, args.src, datadir, key='oraclewordns')
                auto_truncate(f'{datadir}/{split}.oraclewordns{args.src}', f'{datadir}/tmp.src', args.max_position)
                os.rename(f'{datadir}/tmp.src', f'{datadir}/{split}.oraclewordns{args.src}')
                add_leading_space(f'{datadir}/{split}.oraclewordns{args.src}')
                os.rename(f'{datadir}/{split}.oraclewordns{args.src}lead', f'{datadir}/{split}.oraclewordns{args.src}')

