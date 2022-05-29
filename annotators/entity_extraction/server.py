import datetime
import json
import logging
import os
import re
import subprocess
import time
import uvicorn
from typing import Any, List, Optional, Dict
from fastapi import FastAPI, File, UploadFile
from filelock import FileLock, Timeout
from pydantic import BaseModel
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware
from deeppavlov import build_model
from deeppavlov.core.commands.utils import parse_config
from train import evaluate, ner_config, metrics_filename, LOCKFILE, LOG_PATH

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
include_misc = bool(int(os.getenv("INCLUDE_MISC", "0")))
misc_thres = float(os.getenv("MISC_THRES", "0.88"))

ner_config = parse_config(ner_config_name)
entity_detection_config_name = ner_config['chainer']['pipe'][1]['ner']['config_path']
entity_detection = json.load(open(entity_detection_config_name, 'r'))
entity_detection["chainer"]["pipe"][6]["include_misc"] = include_misc
entity_detection["chainer"]["pipe"][6]["misc_thres"] = misc_thres
json.dump(entity_detection, open(entity_detection_config_name, 'w'), indent=2)

logger.info(f"ner_config {ner_config['chainer']['pipe'][1]['ner']}")

try:
    ner = build_model(ner_config, download=True)
    el = build_model(el_config_name, download=True)
    logger.info("model loaded")
except Exception as e:
    logger.exception(e)
    raise e


def add_stop_signs(texts):
    new_texts = []
    for text in texts:
        if text and isinstance(text, str) and text[-1] not in {".", ",", "?", "!"}:
            text = f"{text}."
        new_texts.append(text)
    return new_texts


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
    texts = add_stop_signs(texts)
    entity_info = {}
    try:
        entity_substr, init_entity_offsets, entity_offsets, entity_positions, tags, sentences_offsets, sentences, \
            probas = ner(texts)
        if el_config_name == "entity_linking_en.json":
            entity_ids, entity_tags, entity_conf, entity_pages = \
                el(entity_substr, tags, sentences, entity_offsets, sentences_offsets, probas)
            entity_info = {"entity_substr": entity_substr, "entity_offsets": entity_offsets, "entity_ids": entity_ids,
                           "entity_tags": entity_tags, "entity_conf": entity_conf, "entity_pages": entity_pages}
        elif el_config_name == "entity_linking_en_full.json":
            entity_ids, entity_tags, entity_conf, entity_pages, image_links, categories, first_pars, dbpedia_types = \
                el(entity_substr, tags, sentences, entity_offsets, sentences_offsets, probas)
            for i in range(len(entity_substr)):
                for j in range(len(entity_substr[i])):
                    if entity_tags[i][j] == []:
                        entity_tags[i][j] = [""]
                    if entity_ids[i][j] == []:
                        entity_ids[i][j] = [""]
                        entity_tags[i][j] = [[]]
                        entity_conf[i][j] = [0.0]
                        entity_pages[i][j] = [""]
                        image_links[i][j] = [""]
                        first_pars[i][j] = [""]
                        categories[i][j] = [[]]
                        dbpedia_types[i][j] = [[]]

            entity_info = {"entity_substr": entity_substr, "entity_offsets": entity_offsets, "entity_ids": entity_ids,
                           "entity_tags": entity_tags, "entity_conf": entity_conf, "entity_pages": entity_pages,
                           "image_links": image_links, "categories": categories, "first_paragraphs": first_pars,
                           "dbpedia_types": dbpedia_types}
    except Exception as e:
        logger.exception(e)
    total_time = time.time() - st_time
    logger.info(f"entity linking exec time = {total_time:.3f}s")
    return entity_info


@app.post("/entity_detection")
async def entity_detection(payload: Payload):
    st_time = time.time()
    texts = payload.texts
    texts = add_stop_signs(texts)
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


@app.post("/train")
async def model_training(fl: Optional[UploadFile] = File(None)):
    data_path = "''"
    logger.info('Trying to start training')
    if fl:
        total_data = json.loads(await fl.read())
        if isinstance(total_data, list):
            train_data = total_data[:int(len(total_data) * 0.9)]
            test_data = total_data[int(len(total_data) * 0.9):]
        elif isinstance(total_data, dict) and "train" in total_data and "test" in total_data:
            train_data = total_data["train"]
            test_data = total_data["test"]
        else:
            raise HTTPException(status_code=400, detail="Train data should be either list with examples or dict with"
                                                        "'train' and 'test' keys")
        logger.info(f"train data {len(train_data)} test data {len(test_data)}")
        data_path = "/tmp/train_filename.json"
        with open(data_path, 'w', encoding="utf8") as out:
            json.dump({"train": train_data, "valid": test_data, "test": test_data},
                      out, indent=2, ensure_ascii=False)
    try:
        with FileLock(LOCKFILE, timeout=1):
            logfile = LOG_PATH / f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'
            subprocess.Popen(['/bin/bash', '-c', f'python train.py {data_path}> {logfile} 2>&1'])
    except Timeout:
        logger.error("Can't start training since process is already running.")
        return {"success": False, "message": "Last training was not finished."}

    return {"success": True, "message": "Training started"}


uvicorn.run(app, host='0.0.0.0', port=9103)
