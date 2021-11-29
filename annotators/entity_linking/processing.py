import logging
import re
from typing import List, Tuple, Dict, Union
from collections import defaultdict

import numpy as np

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

GENRES_TEMPLATE = re.compile(
    r"(\brock\b|heavy metal|\bjazz\b|\bblues\b|\bpop\b|\brap\b|hip hop\btechno\b" r"|dubstep|classic)"
)
SPORT_TEMPLATE = re.compile(r"(soccer|football|basketball|baseball|tennis|mma|boxing|volleyball|chess|swimming)")

genres_dict = {
    "rock": "Q11399",
    "heavy metal": "Q38848",
    "jazz": "Q8341",
    "blues": "Q9759",
    "pop": "Q37073",
    "rap": "Q6010",
    "hip hop": "Q6010",
    "techno": "Q170611",
    "dubstep": "Q20474",
    "classic": "Q9730",
}

sport_dict = {
    "soccer": "Q2736",
    "football": "Q2736",
    "basketball": "Q5372",
    "baseball": "Q5369",
    "tennis": "Q847",
    "mma": "Q114466",
    "boxing": "Q32112",
    "volleyball": "Q1734",
    "chess": "Q718",
    "swimming": "Q31920",
}


@register("entity_linking_preprocessor")
class EntityLinkingPreprocessor(Component):
    def __init__(self, **kwargs):
        pass

    def __call__(
        self,
        entity_substr,
        template,
        context
    ):
        long_context = short_context = context
        logger.error("Contexts:")
        logger.error(long_context)
        logger.error(short_context)
        logger.error(context)
        # long_context = []
        # short_context = []
        # for entity_substr_list, context_list in zip(entity_substr, context):
        #     last_utt = context_list[-1]
        #     if (
        #             len(last_utt) > 1
        #             and any([entity_substr.lower() == last_utt.lower() for entity_substr in entity_substr_list])
        #             or any([entity_substr.lower() == last_utt[:-1] for entity_substr in entity_substr_list])
        #     ):
        #         context = " ".join(context_list)
        #     else:
        #         context = last_utt
        #     if isinstance(context, list):
        #         context = " ".join(context)
        #     if isinstance(last_utt, list):
        #         short_context = " ".join(last_utt)
        #     else:
        #         short_context = last_utt
        #     long_context.append(context)
        #     short_context.append(short_context)

        entity_types = [[[] for _ in entity_substr_list] for entity_substr_list in entity_substr]
        # entity_info = [[{}] for _ in entity_substr]

        return entity_substr, template, long_context, entity_types, short_context


@register("entity_linking_postprocessor")
class EntityLinkingPostprocessor(Component):
    def __init__(self, **kwargs):
        pass

    def __call__(
        self, entity_ids_batch, conf_batch, tokens_match_conf_batch, entity_substr_batch, short_context_batch
    ):
        entity_info_list = []
        for entity_substr_list, entity_ids_list, conf_list, tokens_match_conf_list, context in zip(
                entity_substr_batch, entity_ids_batch, conf_batch, tokens_match_conf_batch, short_context_batch
        ):
            # entity_info_list = []
            for entity_substr, entity_ids, conf, tokens_match_conf in zip(
                    entity_substr_list, entity_ids_list, conf_list, tokens_match_conf_list
            ):
                entity_info = {}
                entity_info["entity_substr"] = entity_substr
                entity_info["entity_ids"] = entity_ids
                entity_info["confidences"] = [float(elem) for elem in conf]
                entity_info["tokens_match_conf"] = [float(elem) for elem in tokens_match_conf]
                entity_info_list.append(entity_info)
            topic_substr, topic_id = self._extract_topic_skill_entities(context, entity_substr_list, entity_ids_list)
            if topic_substr:
                entity_info = {}
                entity_info["entity_substr"] = topic_substr
                entity_info["entity_ids"] = [topic_id]
                entity_info["confidences"] = [float(1.0)]
                entity_info["tokens_match_conf"] = [float(1.0)]
                entity_info_list.append(entity_info)
            # entity_info_batch.append(entity_info_list)
        return entity_info_list

    def _extract_topic_skill_entities(self, utt, entity_substr_list, entity_ids_list):
        found_substr = ""
        found_id = ""
        found_genres = re.findall(GENRES_TEMPLATE, utt)
        if found_genres:
            genre = found_genres[0]
            genre_id = genres_dict[genre]
            if all([genre not in elem for elem in entity_substr_list]) or all(
                    [genre_id not in entity_ids for entity_ids in entity_ids_list]
            ):
                found_substr = genre
                found_id = genre_id
        found_sport = re.findall(SPORT_TEMPLATE, utt)
        if found_sport:
            sport = found_sport[0]
            sport_id = sport_dict[sport]
            if all([sport not in elem for elem in entity_substr_list]) or all(
                    [sport_id not in entity_ids for entity_ids in entity_ids_list]
            ):
                found_substr = sport
                found_id = sport_id

        return found_substr, found_id
