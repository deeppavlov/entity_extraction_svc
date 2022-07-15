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

from typing import List, Tuple, Union, Dict
from collections import defaultdict

import numpy as np

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('question_sign_checker')
class QuestionSignChecker(Component):
    """This class adds question sign if it is absent or replaces dot with question sign"""

    def __init__(self, **kwargs):
        pass

    def __call__(self, questions: List[str]) -> List[str]:
        questions_sanitized = []
        for question in questions:
            if not question.endswith('?'):
                if question.endswith('.'):
                    question = question[:-1] + '?'
                else:
                    question += '?'
            questions_sanitized.append(question)
        return questions_sanitized


@register('entity_detection_parser')
class EntityDetectionParser(Component):
    """This class parses probabilities of tokens to be a token from the entity substring."""

    def __init__(self, o_tag: str, tags_file: str, entity_tags: List[str] = None, not_used_tags: List[str] = None,
                 ignore_points: bool = False, return_entities_with_tags: bool = False, thres_proba: float = 0.8,
                 **kwargs):
        """
        Args:
            entity_tags: tags for entities
            type_tag: tag for types
            o_tag: tag for tokens which are neither entities nor types
            tags_file: filename with NER tags
            ignore_points: whether to consider points as separate symbols
            return_entities_with_tags: whether to return a dict of tags (keys) and list of entity substrings (values)
                or simply a list of entity substrings
            thres_proba: if the probability of the tag is less than thres_proba, we assign the tag as 'O'
        """
        self.entity_tags = entity_tags
        self.o_tag = o_tag
        self.ignore_points = ignore_points
        self.return_entities_with_tags = return_entities_with_tags
        self.thres_proba = thres_proba
        self.tag_ind_dict = {}
        self.tags_init = []
        with open(str(expand_path(tags_file))) as fl:
            tags = [line.split('\t')[0] for line in fl.readlines()]
            self.tags_init = tags
            if self.entity_tags is None:
                self.entity_tags = list({tag.split('-')[1] for tag in tags if len(tag.split('-')) > 1}.difference({self.o_tag}))
            self.entity_tags = [elem for elem in self.entity_tags if elem not in not_used_tags]

            self.entity_prob_ind = {entity_tag: [i for i, tag in enumerate(tags) if entity_tag in tag]
                                    for entity_tag in self.entity_tags}
            self.tags_ind = {tag: i for i, tag in enumerate(tags)}
            self.et_prob_ind = [i for tag, ind in self.entity_prob_ind.items() for i in ind]
            for entity_tag, tag_ind in self.entity_prob_ind.items():
                for ind in tag_ind:
                    self.tag_ind_dict[ind] = entity_tag
            self.tag_ind_dict[0] = self.o_tag

    def __call__(self, question_tokens_batch: List[List[str]], tokens_info_batch: List[List[List[float]]],
                       tokens_probas_batch: np.ndarray) -> \
            Tuple[List[Union[List[str], Dict[str, List[str]]]], List[List[str]],
                  List[Union[List[int], Dict[str, List[List[int]]]]]]:
        """
        Args:
            question_tokens: tokenized questions
            token_probas: list of probabilities of question tokens
        Returns:
            Batch of dicts where keys are tags and values are substrings corresponding to tags
            Batch of substrings which correspond to entity types
            Batch of lists of token indices in the text which correspond to entities
        """
        entities_batch = []
        positions_batch = []
        probas_batch = []
        for tokens, tokens_info, probas in zip(question_tokens_batch, tokens_info_batch, tokens_probas_batch):
            tags, tag_probas = self.tags_from_probas(tokens, probas)
            entities, positions, entities_probas = self.entities_from_tags(tokens, tags, probas)
            entities_batch.append(entities)
            positions_batch.append(positions)
            probas_batch.append(entities_probas)
        return entities_batch, positions_batch, probas_batch

    def tags_from_probas(self, tokens, probas):
        """
        This method makes a list of tags from a list of probas for tags
        Args:
            probas: probabilities for tokens to belong to particular tags
        Returns:
            list of tags for tokens
            list of probabilities of these tags
        """
        tags = []
        tag_probas = []
        for token, proba in zip(tokens, probas):
            tag_num = np.argmax(proba)
            if proba[0] < self.thres_proba:
                proba_list = list(proba)[1:]
                tag_num = np.argmax(proba_list) + 1
            tags.append(self.tags_init[tag_num])
            tag_probas.append(proba[tag_num])

        return tags, tag_probas

    def entities_from_tags(self, tokens, tags, tag_probas):
        """
        This method makes lists of substrings corresponding to entities and entity types
        and a list of indices of tokens which correspond to entities
        Args:
            tokens: list of tokens of the text
            tags: list of tags for tokens
            tag_probas: list of probabilities of tags
        Returns:
            list of entity substrings (or a dict of tags (keys) and entity substrings (values))
            list of substrings for entity types
            list of indices of tokens which correspond to entities (or a dict of tags (keys)
                and list of indices of entity tokens)
        """
        entities_dict = defaultdict(list)
        entity_dict = defaultdict(list)
        entity_positions_dict = defaultdict(list)
        entities_positions_dict = defaultdict(list)
        entities_probas_dict = defaultdict(list)
        entity_probas_dict = defaultdict(list)
        replace_tokens = [(' - ', '-'), ("'s", ''), (' .', ''), ('{', ''), ('}', ''),
                          ('  ', ' '), ('"', "'"), ('(', ''), (')', '')]

        cnt = 0
        for n, (tok, tag, probas) in enumerate(zip(tokens, tags, tag_probas)):
            if tag.split('-')[-1] in self.entity_tags:
                f_tag = tag.split("-")[-1]
                if tag.startswith("B-") and any(entity_dict.values()):
                    for c_tag, entity in entity_dict.items():
                        entity = ' '.join(entity)
                        for old, new in replace_tokens:
                            entity = entity.replace(old, new)
                        if entity:
                            entities_dict[c_tag].append(entity)
                            entities_positions_dict[c_tag].append(entity_positions_dict[c_tag])
                            cur_probas = entity_probas_dict[c_tag]
                            entities_probas_dict[c_tag].append(round(sum(cur_probas)/len(cur_probas), 4))
                        entity_dict[c_tag] = []
                        entity_positions_dict[c_tag] = []
                        entity_probas_dict[c_tag] = []
                
                entity_dict[f_tag].append(tok)
                entity_positions_dict[f_tag].append(cnt)
                entity_probas_dict[f_tag].append(probas[self.tags_ind[tag]])

            elif any(entity_dict.values()):
                for tag, entity in entity_dict.items():
                    c_tag = tag.split("-")[-1]
                    entity = ' '.join(entity)
                    for old, new in replace_tokens:
                        entity = entity.replace(old, new)
                    if entity:
                        entities_dict[c_tag].append(entity)
                        entities_positions_dict[c_tag].append(entity_positions_dict[c_tag])
                        cur_probas = entity_probas_dict[c_tag]
                        entities_probas_dict[c_tag].append(round(sum(cur_probas)/len(cur_probas), 4))
                        
                    entity_dict[c_tag] = []
                    entity_positions_dict[c_tag] = []
                    entity_probas_dict[c_tag] = []
            cnt += 1
        
        if any(entity_dict.values()):
            for tag, entity in entity_dict.items():
                c_tag = tag.split("-")[-1]
                entity = ' '.join(entity)
                for old, new in replace_tokens:
                    entity = entity.replace(old, new)
                if entity:
                    entities_dict[c_tag].append(entity)
                    entities_positions_dict[c_tag].append(entity_positions_dict[c_tag])
                    cur_probas = entity_probas_dict[c_tag]
                    entities_probas_dict[c_tag].append(round(sum(cur_probas)/len(cur_probas), 4))
                    
                entity_dict[c_tag] = []
                entity_positions_dict[c_tag] = []
                entity_probas_dict[c_tag] = []

        entities_list = [entity for tag, entities in entities_dict.items() for entity in entities]
        entities_positions_list = [position for tag, positions in entities_positions_dict.items()
                                   for position in positions]
        entities_probas_list = [proba for tag, probas in entities_probas_dict.items() for proba in probas]
        
        entities_dict = dict(entities_dict)
        entities_positions_dict = dict(entities_positions_dict)
        entities_probas_dict = dict(entities_probas_dict)

        if self.return_entities_with_tags:
            return entities_dict, entities_positions_dict, entities_probas_dict
        else:
            return entities_list, entities_positions_list, entities_probas_list
