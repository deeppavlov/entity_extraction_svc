import logging
import re
from typing import List, Tuple, Dict, Union

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

GENRES_TEMPLATE = re.compile(
    r"(\brock\b|heavy metal|\bjazz\b|\bblues\b|\bpop\b|\brap\b|hip hop\btechno\b"
    r"|dubstep|classic)"
)
SPORT_TEMPLATE = re.compile(
    r"(soccer|football|basketball|baseball|tennis|mma|boxing|volleyball|chess|swimming)"
)

GENRES_IDS = {
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

SPORT_IDS = {
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

TEMPLATES = {
    "genres": {"template": GENRES_TEMPLATE, "ids": GENRES_IDS},
    "sport": {"template": SPORT_TEMPLATE, "ids": SPORT_IDS},
}


@register("entity_linking_preprocessor")
class EntityLinkingPreprocessor(Component):
    def __init__(self, **kwargs):
        pass

    def __call__(self, entity_substr, template, context):
        long_context = short_context = context
        entity_types = [
            [[] for _ in entity_substr_list] for entity_substr_list in entity_substr
        ]

        return entity_substr, template, long_context, entity_types, short_context


def _extract_topic_skill_entities(utt, entity_substr_list, entity_ids_list):
    found_substr = ""
    found_id = ""
    for template_name, template_mappings in TEMPLATES.items():
        found_template = re.findall(template_mappings["template"], utt)
        if found_template:
            template_value = found_template[0]
            template_value_id = template_mappings["ids"][template_value]
            is_substr_detected = any(
                template_value in elem for elem in entity_substr_list
            )
            is_id_detected = any(
                template_value_id in entity_ids for entity_ids in entity_ids_list
            )
            if not is_substr_detected or not is_id_detected:
                found_substr = template_value
                found_id = template_value_id

    return found_substr, found_id


@register("entity_linking_postprocessor")
class EntityLinkingPostprocessor(Component):
    def __init__(self, **kwargs):
        pass

    def __call__(
        self,
        entity_ids_batch,
        conf_batch,
        tokens_match_conf_batch,
        entity_substr_batch,
        short_context_batch,
    ):
        entity_info_batch = []
        for (
            entity_substr_list,
            entity_ids_list,
            conf_list,
            tokens_match_conf_list,
            context,
        ) in zip(
            entity_substr_batch,
            entity_ids_batch,
            conf_batch,
            tokens_match_conf_batch,
            short_context_batch,
        ):
            entity_info_list = []
            for entity_substr, entity_ids, conf, tokens_match_conf in zip(
                entity_substr_list, entity_ids_list, conf_list, tokens_match_conf_list
            ):
                entity_info = {
                    "entity_substr": entity_substr,
                    "entity_ids": entity_ids,
                    "confidences": [float(elem) for elem in conf],
                    "tokens_match_conf": [float(elem) for elem in tokens_match_conf],
                }
                entity_info_list.append(entity_info)

            topic_substr, topic_id = _extract_topic_skill_entities(
                context, entity_substr_list, entity_ids_list
            )
            if topic_substr:
                entity_info = {
                    "entity_substr": topic_substr,
                    "entity_ids": [topic_id],
                    "confidences": [float(1.0)],
                    "tokens_match_conf": [float(1.0)],
                }
                entity_info_list.append(entity_info)
            entity_info_batch.append(entity_info_list)

        return entity_info_batch
