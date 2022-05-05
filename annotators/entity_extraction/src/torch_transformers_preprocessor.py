# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import random
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
import torch
from typing import Tuple, List, Optional, Union, Dict, Set

import numpy as np
from nltk.corpus import stopwords
from transformers import AutoTokenizer
from transformers.data.processors.utils import InputFeatures

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import zero_pad
from deeppavlov.core.models.component import Component
from deeppavlov.models.preprocessors.mask import Mask

log = getLogger(__name__)


@register('torch_transformers_ner_preprocessor')
class TorchTransformersNerPreprocessor(Component):
    """
    Takes tokens and splits them into bert subtokens, encodes subtokens with their indices.
    Creates a mask of subtokens (one for the first subtoken, zero for the others).

    If tags are provided, calculates tags for subtokens.

    Args:
        vocab_file: path to vocabulary
        do_lower_case: set True if lowercasing is needed
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        max_subword_length: replace token to <unk> if it's length is larger than this
            (defaults to None, which is equal to +infinity)
        token_masking_prob: probability of masking token while training
        provide_subword_tags: output tags for subwords or for words
        subword_mask_mode: subword to select inside word tokens, can be "first" or "last"
            (default="first")

    Attributes:
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        max_subword_length: rmax lenght of a bert subtoken
        tokenizer: instance of Bert FullTokenizer
    """

    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = False,
                 max_seq_length: int = 512,
                 max_subword_length: int = None,
                 token_masking_prob: float = 0.0,
                 provide_subword_tags: bool = False,
                 subword_mask_mode: str = "first",
                 return_offsets: bool = False,
                 **kwargs):
        self._re_tokenizer = re.compile(r"[\w']+|[^\w ]")
        self.provide_subword_tags = provide_subword_tags
        self.mode = kwargs.get('mode')
        self.max_seq_length = max_seq_length
        self.max_subword_length = max_subword_length
        self.subword_mask_mode = subword_mask_mode
        if Path(vocab_file).is_file():
            vocab_file = str(expand_path(vocab_file))
            self.tokenizer = AutoTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)
        self.token_masking_prob = token_masking_prob
        self.return_offsets = return_offsets

    def __call__(self,
                 tokens: Union[List[List[str]], List[str]],
                 tags: List[List[str]] = None,
                 **kwargs):
        tokens_offsets_batch = [[] for _ in tokens]
        if isinstance(tokens[0], str):
            tokens_batch = []
            tokens_offsets_batch = []
            for s in tokens:
                tokens_list = []
                tokens_offsets_list = []
                for elem in re.finditer(self._re_tokenizer, s):
                    tokens_list.append(elem[0])
                    tokens_offsets_list.append((elem.start(), elem.end()))
                tokens_batch.append(tokens_list)
                tokens_offsets_batch.append(tokens_offsets_list)
            tokens = tokens_batch
        subword_tokens, subword_tok_ids, startofword_markers, subword_tags = [], [], [], []
        for i in range(len(tokens)):
            toks = tokens[i]
            ys = ['O'] * len(toks) if tags is None else tags[i]
            assert len(toks) == len(ys), \
                f"toks({len(toks)}) should have the same length as ys({len(ys)})"
            sw_toks, sw_marker, sw_ys = \
                self._ner_bert_tokenize(toks,
                                        ys,
                                        self.tokenizer,
                                        self.max_subword_length,
                                        mode=self.mode,
                                        subword_mask_mode=self.subword_mask_mode,
                                        token_masking_prob=self.token_masking_prob)
            if self.max_seq_length is not None:
                if len(sw_toks) > self.max_seq_length:
                    raise RuntimeError(f"input sequence after bert tokenization"
                                       f" shouldn't exceed {self.max_seq_length} tokens.")
            subword_tokens.append(sw_toks)
            subword_tok_ids.append(self.tokenizer.convert_tokens_to_ids(sw_toks))
            startofword_markers.append(sw_marker)
            subword_tags.append(sw_ys)
            assert len(sw_marker) == len(sw_toks) == len(subword_tok_ids[-1]) == len(sw_ys), \
                f"length of sow_marker({len(sw_marker)}), tokens({len(sw_toks)})," \
                f" token ids({len(subword_tok_ids[-1])}) and ys({len(ys)})" \
                f" for tokens = `{toks}` should match"

        subword_tok_ids = zero_pad(subword_tok_ids, dtype=int, padding=0)
        startofword_markers = zero_pad(startofword_markers, dtype=int, padding=0)
        attention_mask = Mask()(subword_tokens)

        if tags is not None:
            if self.provide_subword_tags:
                return tokens, subword_tokens, subword_tok_ids, \
                       attention_mask, startofword_markers, subword_tags
            else:
                nonmasked_tags = [[t for t in ts if t != 'X'] for ts in tags]
                for swts, swids, swms, ts in zip(subword_tokens,
                                                 subword_tok_ids,
                                                 startofword_markers,
                                                 nonmasked_tags):
                    if (len(swids) != len(swms)) or (len(ts) != sum(swms)):
                        log.warning('Not matching lengths of the tokenization!')
                        log.warning(f'Tokens len: {len(swts)}\n Tokens: {swts}')
                        log.warning(f'Markers len: {len(swms)}, sum: {sum(swms)}')
                        log.warning(f'Masks: {swms}')
                        log.warning(f'Tags len: {len(ts)}\n Tags: {ts}')
                return tokens, subword_tokens, subword_tok_ids, \
                       attention_mask, startofword_markers, nonmasked_tags
        if self.return_offsets:
            return tokens, subword_tokens, subword_tok_ids, startofword_markers, attention_mask, tokens_offsets_batch
        else:
            return tokens, subword_tokens, subword_tok_ids, startofword_markers, attention_mask

    @staticmethod
    def _ner_bert_tokenize(tokens: List[str],
                           tags: List[str],
                           tokenizer: AutoTokenizer,
                           max_subword_len: int = None,
                           mode: str = None,
                           subword_mask_mode: str = "first",
                           token_masking_prob: float = None) -> Tuple[List[str], List[int], List[str]]:
        do_masking = (mode == 'train') and (token_masking_prob is not None)
        do_cutting = (max_subword_len is not None)
        tokens_subword = ['[CLS]']
        startofword_markers = [0]
        tags_subword = ['X']
        for token, tag in zip(tokens, tags):
            token_marker = int(tag != 'X')
            subwords = tokenizer.tokenize(token)
            if not subwords or (do_cutting and (len(subwords) > max_subword_len)):
                tokens_subword.append('[UNK]')
                startofword_markers.append(token_marker)
                tags_subword.append(tag)
            else:
                if do_masking and (random.random() < token_masking_prob):
                    tokens_subword.extend(['[MASK]'] * len(subwords))
                else:
                    tokens_subword.extend(subwords)
                if subword_mask_mode == "last":
                    startofword_markers.extend([0] * (len(subwords) - 1) + [token_marker])
                else:
                    startofword_markers.extend([token_marker] + [0] * (len(subwords) - 1))
                tags_subword.extend([tag] + ['X'] * (len(subwords) - 1))

        tokens_subword.append('[SEP]')
        startofword_markers.append(0)
        tags_subword.append('X')
        return tokens_subword, startofword_markers, tags_subword


@register('split_markups')
class SplitMarkups:
    def __init__(self, **kwargs):
        pass
    def __call__(self, y_batch):
        y_types_batch, y_spans_batch = [], []
        for y_list in y_batch:
            y_types_list, y_spans_list = [], []
            for i in range(len(y_list)):
                if y_list[i].startswith("B-"):
                    label = y_list[i].split("-")[1]
                    y_types_list.append(label)
                    y_spans_list.append("B-ENT")
                elif y_list[i].startswith("I-"):
                    label = y_list[i].split("-")[1]
                    y_types_list.append(label)
                    y_spans_list.append("I-ENT")
                else:
                    y_types_list.append("O")
                    y_spans_list.append("O")
            y_types_batch.append(y_types_list)
            y_spans_batch.append(y_spans_list)
        return y_types_batch, y_spans_batch


@register('merge_markups')
class MergeMarkups:
    def __init__(self, tags_file: str, use_o_tag: bool = False, long_ent_thres: float = 0.4,
                       ent_thres: float = 0.4, top_n: int = 1, include_misc: bool = True, lang: str = "en", **kwargs):
        tags_file = str(expand_path(tags_file))
        self.tags_list = []
        with open(tags_file, 'r') as fl:
            lines = fl.readlines()
            for line in lines:
                tag, score = line.strip().split()
                if tag != "O":
                    self.tags_list.append(tag)
        self.use_o_tag = use_o_tag
        self.ent_thres = ent_thres
        self.long_ent_thres = long_ent_thres
        self.top_n = top_n
        self.include_misc = include_misc
        self.lang = lang
        if self.lang == "en":
            self.stopwords = set(stopwords.words("english"))
        else:
            self.stopwords = set(stopwords.words("russian"))
    
    def __call__(self, tokens_batch, y_types_batch, y_spans_batch):
        y_batch, entities_batch, entity_positions_batch, entity_tags_batch, entity_probas_batch = [], [], [], [], []
        for tokens_list, y_types_list, y_spans_list in zip(tokens_batch, y_types_batch, y_spans_batch):
            y_types_list = y_types_list.tolist()
            y_list = []
            tags_with_probas_list = []
            label = ""
            conf = 0.0
            entities_list, entity_positions_list, entity_tags_list, entity_probas_list = [], [], [], []
            for i in range(len(y_types_list)):
                if y_spans_list[i].startswith("B-") or (y_spans_list[i].startswith("I-") and \
                        (i == 0 or (i > 0 and y_spans_list[i - 1] == "O"))):
                    if "MISC" not in y_spans_list[i] or ("MISC" in y_spans_list[i] and self.include_misc):
                        tags_with_probas = {tag: 0.0 for tag in self.tags_list}
                        num_words = 0
                        if self.use_o_tag:
                            for k in range(1, len(y_types_list[i])):
                                tags_with_probas[self.tags_list[k - 1]] += y_types_list[i][k]
                        else:
                            for k in range(len(y_types_list[i])):
                                tags_with_probas[self.tags_list[k]] += y_types_list[i][k]
                        num_words += 1
                        for j in range(i + 1, len(y_types_list)):
                            if y_spans_list[j].startswith("I-"):
                                if self.use_o_tag:
                                    for k in range(1, len(y_types_list[j])):
                                        tags_with_probas[self.tags_list[k - 1]] += y_types_list[j][k]
                                else:
                                    for k in range(len(y_types_list[j])):
                                        tags_with_probas[self.tags_list[k]] += y_types_list[j][k]
                                num_words += 1
                            else:
                                break
                        tags_with_probas = list(tags_with_probas.items())
                        tags_with_probas = [(tag, round(proba_sum / num_words, 3)) for tag, proba_sum in tags_with_probas]
                        tags_with_probas = sorted(tags_with_probas, key=lambda x: x[1], reverse=True)
                        tags_with_probas_list.append(tags_with_probas)
                        label = tags_with_probas[0][0]
                        conf = tags_with_probas[0][1]
                        if conf > self.long_ent_thres or (num_words <= 2 and conf > self.ent_thres):
                            y_list.append(f"B-{label}")
                            new_entity = " ".join(tokens_list[i:i + num_words])
                            if new_entity not in self.stopwords:
                                entities_list.append(new_entity)
                                entity_positions_list.append(list(range(i, i + num_words)))
                                if self.top_n == 1:
                                    entity_tags_list.append(tags_with_probas[0][0])
                                    entity_probas_list.append(tags_with_probas[0][1])
                                else:
                                    entity_tags_list.append([elem[0] for elem in tags_with_probas[:self.top_n]])
                                    entity_probas_list.append([elem[1] for elem in tags_with_probas[:self.top_n]])
                        else:
                            y_list.append("O")
                    else:
                        y_list.append("O")
                elif y_spans_list[i].startswith("I-"):
                    if "MISC" not in y_spans_list[i] or ("MISC" in y_spans_list[i] and self.include_misc):
                        if conf > self.long_ent_thres or (num_words <= 2 and conf > self.ent_thres):
                            y_list.append(f"I-{label}")
                        else:
                            y_list.append("O")
                    else:
                        y_list.append("O")
                else:
                    y_list.append("O")
                    label = ""
                    conf = 0.0
            y_batch.append(y_list)
            entities_batch.append(entities_list)
            entity_positions_batch.append(entity_positions_list)
            entity_tags_batch.append(entity_tags_list)
            entity_probas_batch.append(entity_probas_list)
        return y_batch, entities_batch, entity_positions_batch, entity_tags_batch, entity_probas_batch
