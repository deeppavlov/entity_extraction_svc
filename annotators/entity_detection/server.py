import logging
import os
import re
from typing import List, Tuple, Dict, Union

from deeppavlov import build_model
from environs import Env
from fastapi import FastAPI
from pydantic import BaseModel
import sentry_sdk
import uvicorn

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

sentry_sdk.init(os.getenv("SENTRY_DSN"))

app = FastAPI()

env = Env()
env.read_env()

config_name = env.str("CONFIG", "dialog_entity_detection.json")


try:
    entity_detection_alexa = build_model("dialog_entity_detection.json", download=True)
    entity_detection_lcquad = build_model("entity_detection_lcquad.json", download=True)
    entity_detection_alexa(["what is the capital of russia"])
    logger.info("entity detection model is loaded.")
except Exception as e:
    sentry_sdk.capture_exception(e)
    logger.exception(e)
    raise e

nltk_stopwords_file = "nltk_stopwords.txt"
nltk_stopwords = [line.strip() for line in open(nltk_stopwords_file, "r").readlines()]
CHARS_EXCEPT_LETTERS_DIGITS_AND_SPACE = re.compile(r"[^a-zA-Z0-9 \-&*+]")
DOUBLE_SPACES = re.compile(r"\s+")


class EntityDetectionRequest(BaseModel):
    text: str


def _align_entities_tags_offsets(
    entities_batch: List[List[str]],
    tags_batch: List[List[str]],
    offsets_batch: List[List[Tuple[int, int]]],
    context: str,
) -> Dict[Tuple[int, int], Dict[str, Union[str, Tuple[int, int]]]]:
    aligned_entities = {}
    zipped_entities_batch = zip(entities_batch, tags_batch, offsets_batch)
    for entities_list, tags_list, entities_offsets_list in zipped_entities_batch:
        zipped_entities_list = zip(entities_list, tags_list, entities_offsets_list)
        for entity, tag, offsets in zipped_entities_list:
            if entity not in nltk_stopwords and len(entity) > 2:
                entity = CHARS_EXCEPT_LETTERS_DIGITS_AND_SPACE.sub(" ", entity)
                entity = DOUBLE_SPACES.sub(" ", entity).strip()
                aligned_entities[offsets] = {"entity_substr": entity, "tag": tag, "offsets": offsets, "context": context}

    return aligned_entities


def extract_entities(text: str):
    (
        entities_batch,
        tags_batch,
        positions_batch,
        entities_offsets_batch,
        probas_batch
    ) = entity_detection_alexa([text])
    (
        entities_batch_lc,
        tags_batch_lc,
        positions_batch_lc,
        entities_offsets_batch_lc,
        probas_batch_lc,
    ) = entity_detection_lcquad([text])

    logger.info(f"entities:\nalexa\n{entities_batch}\nlc\n{entities_batch_lc}")
    logger.info(f"tags:\nalexa\n{tags_batch}\nlc\n{tags_batch_lc}")
    logger.info(f"positions:\nalexa\n{positions_batch}\nlc\n{positions_batch_lc}")
    logger.info(f"entities_offsets:\nalexa\n{entities_offsets_batch}\nlc\n{entities_offsets_batch_lc}")
    logger.info(f"probas:\nalexa\n{probas_batch}\nlc\n{probas_batch_lc}")

    alexa_entities = _align_entities_tags_offsets(entities_batch, tags_batch, entities_offsets_batch, text)
    lc_entities = _align_entities_tags_offsets(entities_batch_lc, tags_batch_lc, entities_offsets_batch_lc, text)
    unique_entities = {**alexa_entities, **lc_entities}

    logger.info(f"unique_entities: {unique_entities}")

    return list(unique_entities.values())


@app.post("/respond")
async def respond(entity_detection_request: EntityDetectionRequest):
    entities = extract_entities(entity_detection_request.text)
    return entities


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=9103)
