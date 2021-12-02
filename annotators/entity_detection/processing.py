import logging
import re
from typing import List, Tuple, Dict, Union

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

nltk_stopwords_file = "nltk_stopwords.txt"
nltk_stopwords = [line.strip() for line in open(nltk_stopwords_file, "r").readlines()]
CHARS_EXCEPT_LETTERS_DIGITS_AND_SPACE = re.compile(r"[^a-zA-Z0-9 \-&*+]")
DOUBLE_SPACES = re.compile(r"\s+")


def _align_entities_tags_offsets(
    entities_batch: List[List[str]],
    tags_batch: List[List[str]],
    offsets_batch: List[List[Tuple[int, int]]],
    context: str,
) -> Dict[Tuple[int, int], Dict[str, Union[str, Tuple[int, int]]]]:
    """Create a dictionary with offsets as keys and dictionaries containing predicted entity info as values.
    This way we can deduplicate predictions form multiple models based on their offsets

    :param entities_batch: predicted entities
    :param tags_batch: predicted tags
    :param offsets_batch: predicted offsets
    :param context: original query
    :return: dict of entities with unique offsets
    """
    aligned_entities = {}
    zipped_entities_batch = zip(entities_batch, tags_batch, offsets_batch)
    for entities_list, tags_list, entities_offsets_list in zipped_entities_batch:
        zipped_entities_list = zip(entities_list, tags_list, entities_offsets_list)
        for entity, tag, offsets in zipped_entities_list:
            if entity not in nltk_stopwords and len(entity) > 2:
                entity = CHARS_EXCEPT_LETTERS_DIGITS_AND_SPACE.sub(" ", entity)
                entity = DOUBLE_SPACES.sub(" ", entity).strip()
                aligned_entities[offsets] = {
                    "entity_substr": entity,
                    "tag": tag,
                    "offsets": offsets,
                    "context": context,
                }

    return aligned_entities


@register("entity_deduplicator")
class EntityDeduplicator(Component):
    def __init__(self, **kwargs):
        pass

    def __call__(
        self,
        text: str,
        entities_alexa: List[List[str]],
        tags_alexa: List[List[str]],
        entities_offsets_alexa: List[List[Tuple[int, int]]],
        entities_lc: List[List[str]],
        tags_lc: List[List[str]],
        entities_offsets_lc: List[List[Tuple[int, int]]],
    ):
        """Merge and deduplicate predictions from alexa and lc models

        :param text: original query
        :param entities_alexa: predicted alexa entities
        :param tags_alexa: predicted alexa tags
        :param entities_offsets_alexa: predicted alexa offsets
        :param entities_lc: predicted lc entities
        :param tags_lc: predicted lc tags
        :param entities_offsets_lc: predicted lc offsets
        :return: batches of substrings, templates and context
        """
        alexa_entities = _align_entities_tags_offsets(
            entities_alexa, tags_alexa, entities_offsets_alexa, text
        )
        lc_entities = _align_entities_tags_offsets(
            entities_lc, tags_lc, entities_offsets_lc, text
        )
        unique_entities = {**alexa_entities, **lc_entities}
        unique_entities_list = list(unique_entities.values())

        entity_substr = [[[ent["entity_substr"] for ent in unique_entities_list]]]
        template = [[""]]
        context = [[unique_entities_list[0]["context"][0]]]

        logger.info(f"detection: {entity_substr}, {template}, {context}")

        return entity_substr, template, context
