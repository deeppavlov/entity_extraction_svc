import logging
import os
import re
import time
import uvicorn
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile
from filelock import FileLock, Timeout
from pydantic import BaseModel
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware
from deeppavlov import build_model

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

ner_config_name = os.getenv("NER_CONFIG")
el_config_name = os.getenv("EL_CONFIG")

try:
    ner = build_model(ner_config_name, download=True)
    el = build_model(el_config_name, download=True)
    logger.info("model loaded")
except Exception as e:
    logger.exception(e)
    raise e


class Payload(BaseModel):
    texts: List[str]


@app.post("/entity_extraction")
async def entity_extraction(payload: Payload):
    st_time = time.time()
    texts = payload.texts
    entity_info = {}
    try:
        entity_substr, entity_offsets, entity_positions, tags, sentences_offsets, sentences, probas = ner(texts)
        entity_ids, entity_tags, entity_conf, entity_pages = \
            el(entity_substr, tags, sentences, entity_offsets, sentences_offsets, probas)
        entity_info = {"entity_substr": entity_substr, "entity_offsets": entity_offsets, "entity_ids": entity_ids,
                       "entity_tags": tags, "entity_conf": entity_conf, "entity_pages": entity_pages}
    except Exception as e:
        logger.exception(e)
    total_time = time.time() - st_time
    logger.info(f"entity linking exec time = {total_time:.3f}s")
    return entity_info


@app.post("/entity_detection")
async def entity_detection(payload: Payload):
    st_time = time.time()
    texts = payload.texts
    entity_info = {}
    try:
        entity_substr, entity_offsets, entity_positions, tags, sentences_offsets, sentences, probas = ner(texts)
        entity_info = {"entity_substr": entity_substr, "entity_offsets": entity_offsets,
                       "entity_tags": tags, "probas": probas}
    except Exception as e:
        logger.exception(e)
    total_time = time.time() - st_time
    logger.info(f"entity linking exec time = {total_time:.3f}s")
    return entity_info


@app.post("/add_entity")
async def add_entity(payload):
    entity_label = payload.entity_label
    entity_id = payload.entity_id
    num_rels = payload.num_rels
    tag = payload.tag
    page = payload.wiki_page
    if entity_label and entity_id:
        el[0].add_entity(entity_label, entity_id, num_rels, tag, page)


label_rel, type_rel, types_to_tags = "", "", {}

@app.post("/kb_schema")
async def db_schema(payload):
    label_rel = payload.label_rel
    type_rel = payload.type_rel
    types_to_tags = payload.types_to_tags


@app.post("/add_kb")
async def add_kb(fl: Optional[UploadFile] = File(None)):
    if fl:
        kb_data = await fl.read()
        kb_triplets = kb_data.strip("\n").split("\n")
        if label_rel and type_rel and types_to_tags:
            el[0].parse_custom_database(kb_triplets, label_rel, type_rel, types_to_tags)


uvicorn.run(app, host='0.0.0.0', port=9103)
