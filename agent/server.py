import logging
from typing import List

import requests
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EntityExtractionAgentRequest(BaseModel):
    """Agent gateway request"""

    texts: List[str]


class EntityExtractionServiceResponse(BaseModel):
    entity_substr: List[List[str]]
    entity_offsets: List[List[List[int]]]
    entity_ids: List[List[List[str]]]
    entity_tags: List[List[List[str]]]
    entity_conf: List[List[List[List[float]]]]
    entity_pages: List[List[List[str]]]

    def count_entities(self):
        """Counts number of entities in entity-extraction response

        Returns:
            number of entities

        """
        return len(self.entity_offsets[0])

    def count_entity_variants(self, idx):
        """Counts number of entity variants in entity-extraction response

        Args:
            idx: index of entity

        Returns:
            number of entity variants

        """
        return len(self.entity_ids[0][idx])

    def substr(self, idx):
        return self.entity_substr[0][idx]

    def offsets(self, idx):
        return self.entity_offsets[0][idx]

    def id(self, idx, variety_idx):
        return self.entity_ids[0][idx][variety_idx]

    def tag(self, idx, variety_idx):
        varieties = self.entity_tags[0][idx]
        try:
            variety_tag = varieties[variety_idx]
        except IndexError:
            variety_tag = ""

        return variety_tag

    def conf(self, idx, variety_idx):
        return self.entity_conf[0][idx][variety_idx]

    def page(self, idx, variety_idx):
        return self.entity_pages[0][idx][variety_idx]


class ExtractedEntityVariety(BaseModel):
    id: str
    tag: str
    confidence: List[float]
    wiki_page: str


class ExtractedEntity(BaseModel):
    offset: List[int]
    substring: str
    probs: List[ExtractedEntityVariety]


class EntityExtractionAgentResponse(BaseModel):
    """Agent gateway response"""

    entities: List[ExtractedEntity]


def unpack_entity_extraction_service_response(
    entities: EntityExtractionServiceResponse,
):
    readable_entities = []

    for idx in range(entities.count_entities()):
        probs = []

        for variety_idx in range(entities.count_entity_variants(idx)):
            entity_variety = ExtractedEntityVariety(
                id=entities.id(idx, variety_idx),
                tag=entities.tag(idx, variety_idx),
                confidence=entities.conf(idx, variety_idx),
                wiki_page=entities.page(idx, variety_idx),
            )
            probs.append(entity_variety)

        entity = ExtractedEntity(
            offset=entities.offsets(idx),
            substring=entities.substr(idx),
            probs=probs,
        )
        readable_entities.append(entity)

    return EntityExtractionAgentResponse(entities=readable_entities)


@app.post("/")
async def extract(payload: EntityExtractionAgentRequest):
    response = requests.post(
        "http://entity-extraction:9103/entity_extraction", json=payload.dict()
    )
    entities = response.json()
    logger.error(entities)
    entities = EntityExtractionServiceResponse(**entities)
    entities = unpack_entity_extraction_service_response(entities)

    return entities
