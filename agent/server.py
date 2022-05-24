import logging
from datetime import datetime
from typing import List, Union, Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel, BaseSettings
from starlette.middleware.cors import CORSMiddleware

from constants import (
    TAG_TO_TYPE_MAP,
    WIKIPEDIA_PAGE_URI_PREFIX,
    ONTOLOGY_URI_PREFIX,
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


class ServerSettings(BaseSettings):
    entity_extraction_url: str
    entity_detection_url: str
    entity_linking_url: str
    wiki_parser_url: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"


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

    def types(self, idx):
        types_list = []
        tags = self.tags(idx)

        for t in tags:
            try:
                types_list.append(f"{ONTOLOGY_URI_PREFIX}/{TAG_TO_TYPE_MAP[t]}")
            except KeyError:
                pass

        return types_list

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
        return {"wikipedia": self.page_uri(idx, variety_idx)}

    def image(self, idx, variety_idx):
        return self.image_links[0][idx][variety_idx]

    def image_uri(self, idx, variety_idx):
        image_uri = ""
        image = self.image(idx, variety_idx)
        if image:
            image_uri = f"{WIKIPEDIA_FILE_URI_PREFIX}/{image}".replace(" ", "_")

        return image_uri

    def images(self, idx, variety_idx):
        images_dict = {"full": "", "thumbnail": ""}
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


class EntityAnnotation(BaseModel):
    start: int
    end: int
    spot: str
    confidence: float
    id: str
    title: str
    uri: str
    abstract: str
    label: str
    categories: List[str]
    tags: List[str]
    types: List[str]
    image: Optional[EntityAnnotationImage]
    lod: EntityAnnotationLod


class EntityAnnotationWithExtras(EntityAnnotation):
    extras: List[EntityAnnotation]


class EntityExtractionAgentResponse(BaseModel):
    annotations: List[Union[EntityAnnotation, EntityAnnotationWithExtras]]
    lang: str
    timestamp: datetime


def unpack_annotation(
    entities: EntityExtractionServiceResponse,
    entity_idx: int,
    variety_idx: int,
):
    start_offset, end_offset = entities.offsets(entity_idx)

    return EntityAnnotation(
        start=start_offset,
        end=end_offset,
        spot=entities.substr(entity_idx),
        confidence=entities.conf(entity_idx, variety_idx),
        id=entities.id(entity_idx, variety_idx),
        title=entities.page(entity_idx, variety_idx),
        uri=entities.page_uri(entity_idx, variety_idx),
        abstract=entities.first_paragraph(entity_idx, variety_idx),
        label=entities.page(entity_idx, variety_idx),
        categories=entities.category(entity_idx, variety_idx),
        tags=entities.tags(entity_idx),
        types=entities.types(entity_idx),
        image=EntityAnnotationImage.parse_obj(entities.images(entity_idx, variety_idx)),
        lod=EntityAnnotationLod.parse_obj(entities.lod(entity_idx, variety_idx)),
    )


def unpack_entity_extraction_service_response(
    entities: EntityExtractionServiceResponse,
    include_extras: bool = True,
):
    readable_entities = []

    for idx in range(entities.count_entities()):
        top_annotation = unpack_annotation(entities, idx, 0)

        if include_extras:
            extra_annotations = []

            for variety_idx in range(1, entities.count_entity_variants(idx)):
                extra_annotation = unpack_annotation(
                    entities, idx, variety_idx
                )
                extra_annotations.append(extra_annotation)

            full_annotation = EntityAnnotationWithExtras(
                **top_annotation.dict(), extras=extra_annotations
            )
        else:
            full_annotation = top_annotation

        readable_entities.append(full_annotation)

    return EntityExtractionAgentResponse(
        annotations=readable_entities, lang="en", timestamp=datetime.utcnow()
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
