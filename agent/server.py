import logging
from datetime import datetime
from typing import List, Union, Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from agent.config import ServerSettings
from agent.constants import (
    WIKIPEDIA_PAGE_URI_PREFIX,
    WIKIPEDIA_FILE_URI_PREFIX,
)


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.DEBUG
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

    text: str


class EntityExtractionServiceRequest(BaseModel):
    texts: List[str]


class EntityExtractionServiceResponse(BaseModel):
    entity_substr: List[List[str]]
    entity_offsets: List[List[List[int]]]
    entity_ids: List[List[List[str]]]
    entity_tags: List[List[List[str]]]
    entity_conf: List[List[List[float]]]
    entity_pages: List[List[List[str]]]
    image_links: List[List[List[str]]]
    categories: List[List[List[List[str]]]]
    first_paragraphs: List[List[List[str]]]
    dbpedia_types: List[List[List[List[str]]]]

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

    def tags(self, idx):
        return self.entity_tags[0][idx]

    def types(self, idx, variety_idx):
        return self.dbpedia_types[0][idx][variety_idx]

    def id(self, idx, variety_idx):
        return self.entity_ids[0][idx][variety_idx]

    def conf(self, idx, variety_idx):
        return self.entity_conf[0][idx][variety_idx]

    def page(self, idx, variety_idx):
        return self.entity_pages[0][idx][variety_idx]

    def page_uri(self, idx, variety_idx):
        page = self.page(idx, variety_idx)
        return f"{WIKIPEDIA_PAGE_URI_PREFIX}/{page}".replace(" ", "_")

    def lod(self, idx, variety_idx):
        uri = self.page_uri(idx, variety_idx)
        return {"wikipedia": uri} if uri else {}

    def image(self, idx, variety_idx):
        return self.image_links[0][idx][variety_idx]

    def image_uri(self, idx, variety_idx):
        image_uri = ""
        image = self.image(idx, variety_idx)
        if image:
            image_uri = f"{WIKIPEDIA_FILE_URI_PREFIX}/{image}".replace(" ", "_")

        return image_uri

    def images(self, idx, variety_idx):
        images_dict = {}
        image_uri = self.image_uri(idx, variety_idx)
        if image_uri:
            images_dict = {"full": image_uri, "thumbnail": f"{image_uri}?width=300"}

        return images_dict

    def category(self, idx, variety_idx):
        return self.categories[0][idx][variety_idx]

    def first_paragraph(self, idx, variety_idx):
        return self.first_paragraphs[0][idx][variety_idx]


class EntityAnnotationImage(BaseModel):
    full: str
    thumbnail: str


class EntityAnnotationLod(BaseModel):
    wikipedia: str


class BaseEntityAnnotation(BaseModel):
    start: int
    end: int
    spot: str
    tags: List[str]

    @property
    def has_wikidata(self):
        return False


class EntityAnnotation(BaseEntityAnnotation):
    confidence: float
    id: str
    title: str
    uri: str
    abstract: str
    label: str
    categories: List[str]
    image: Optional[dict] = {}
    lod: Optional[dict] = {}
    types: List

    @property
    def has_wikidata(self):
        return True


class BaseEntityAnnotationWithExtras(BaseEntityAnnotation):
    extras: List[BaseEntityAnnotation]


class EntityAnnotationWithExtras(EntityAnnotation):
    extras: List[EntityAnnotation]


AnyBaseAnnotation = Union[BaseEntityAnnotation, BaseEntityAnnotationWithExtras]
AnyAnnotation = Union[EntityAnnotation, EntityAnnotationWithExtras]


class EntityExtractionAgentResponse(BaseModel):
    annotations: Optional[List[AnyAnnotation]] = []
    unlisted_annotations: Optional[List[AnyBaseAnnotation]] = []
    lang: str
    timestamp: datetime


def unpack_annotation(
    entities: EntityExtractionServiceResponse,
    entity_idx: int,
    variety_idx: int,
):
    """
    Creates an Annotation data object

    Args:
        entities: entities from entity-extraction service
        entity_idx: entity index
        variety_idx: entity variety index

    Returns:
        annotation object

    """
    start_offset, end_offset = entities.offsets(entity_idx)
    wikidata_id = entities.id(entity_idx, variety_idx)

    if wikidata_id:
        return EntityAnnotation(
            start=start_offset,
            end=end_offset,
            spot=entities.substr(entity_idx),
            confidence=entities.conf(entity_idx, variety_idx),
            id=wikidata_id,
            title=entities.page(entity_idx, variety_idx),
            uri=entities.page_uri(entity_idx, variety_idx),
            abstract=entities.first_paragraph(entity_idx, variety_idx),
            label=entities.page(entity_idx, variety_idx),
            categories=entities.category(entity_idx, variety_idx),
            tags=entities.tags(entity_idx),
            image=entities.images(entity_idx, variety_idx),
            lod=entities.lod(entity_idx, variety_idx),
            types=entities.types(entity_idx, variety_idx)
        )
    else:
        return BaseEntityAnnotation(
            start=start_offset,
            end=end_offset,
            spot=entities.substr(entity_idx),
            tags=entities.tags(entity_idx),
        )


def unpack_entity_extraction_service_response(
    entities: EntityExtractionServiceResponse,
    include_extras: bool = True,
):
    unlisted_annotations = []
    listed_annotations = []

    for idx in range(entities.count_entities()):
        top_annotation = unpack_annotation(entities, idx, 0)

        if include_extras:
            extra_annotations = []

            for variety_idx in range(1, entities.count_entity_variants(idx)):
                extra_annotation = unpack_annotation(entities, idx, variety_idx)
                extra_annotations.append(extra_annotation)

        else:
            extra_annotations = None

        if top_annotation.has_wikidata:
            full_annotation = EntityAnnotationWithExtras(
                **top_annotation.dict(), extras=extra_annotations
            )
            listed_annotations.append(full_annotation)
        else:
            full_annotation = BaseEntityAnnotationWithExtras(
                **top_annotation.dict(), extras=extra_annotations
            )
            unlisted_annotations.append(full_annotation)

    return EntityExtractionAgentResponse(
        annotations=listed_annotations,
        unlisted_annotations=unlisted_annotations,
        lang="en",
        timestamp=datetime.utcnow(),
    )


server_settings = ServerSettings()


@app.post("/")
async def extract(payload: EntityExtractionAgentRequest):
    request_data = EntityExtractionServiceRequest(texts=[payload.text]).dict()
    response = requests.post(server_settings.entity_extraction_url, json=request_data)
    entities = response.json()

    logger.debug(entities)

    entities = EntityExtractionServiceResponse(**entities)
    entities = unpack_entity_extraction_service_response(entities)

    return entities


@app.get("/health")
async def healthcheck():
    return True
