import json
import logging
import os
import re
import time
import uvicorn
from typing import Any, List, Optional, Dict
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


class TypesAndRels(BaseModel):
    relation_info: Dict[str, Any]


class TripletsList(BaseModel):
    triplets: List[str]


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
async def add_entity(payload: Payload):
    entity_label = payload.entity_label
    entity_id = payload.entity_id
    num_rels = payload.num_rels
    tag = payload.tag
    page = payload.wiki_page
    if entity_label and entity_id:
        el[0].add_entity(entity_label, entity_id, num_rels, tag, page)


label_rel, type_rel, types_to_tags = "", "", {}

@app.post("/kb_schema")
async def db_schema(payload: TypesAndRels):
    logger.info(f"payload {payload}")
    relations = payload.relation_info
    label_rel = relations.get("label_rel", "")
    type_rel = relations.get("type_rel", "")
    types_to_tags = relations.get("types_to_tags", [])
    os.environ["label_rel"] = label_rel
    os.environ["type_rel"] = type_rel
    os.environ["types_to_tags"] = str(types_to_tags)


@app.post("/add_kb")
async def add_kb(payload: TripletsList):
    triplets_list = payload.triplets
    label_rel = os.getenv("label_rel", "")
    type_rel = os.getenv("type_rel", "")
    types_to_tags = json.loads(os.getenv("types_to_tags", "[]"))
    if label_rel:
        el[0].parse_custom_database(triplets_list, label_rel, type_rel, types_to_tags)
        ner[1].ner[6].ent_thres = 0.2
        logger.info(f"------- ner {ner[1].ner[6].ent_thres}")


uvicorn.run(app, host='0.0.0.0', port=9103)
