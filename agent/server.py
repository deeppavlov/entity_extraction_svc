import logging
from datetime import datetime
from typing import List, Union, Optional, Literal

import requests
from fastapi import FastAPI, HTTPException, UploadFile, APIRouter
from fastapi.routing import APIRoute
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from agent.server_utils.config import ServerSettings
from agent.server_utils.constants import (
    WIKIPEDIA_PAGE_URI_PREFIX,
    WIKIPEDIA_FILE_URI_PREFIX,
)
from agent.server_utils import preprocessing
from agent.stats_collector.route_patch import StatsCollectorRoute

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.DEBUG
)
logger = logging.getLogger(__name__)


server_settings = ServerSettings()
if server_settings.collect_stats:
    route_class = StatsCollectorRoute
else:
    route_class = APIRoute


app = FastAPI()
api_router = APIRouter(route_class=route_class)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EntityExtractionAgentRequest(BaseModel):
    """Agent gateway request"""

    text: Optional[str]
    html: Optional[str]
    url: Optional[str]
    parser_engine: Optional[Literal["bs4", "trafilatura"]] = "trafilatura"
    parser_kwargs: Optional[dict] = {}
    attach_parsed_html: bool = False
    include_extras: bool = True


class HtmlParserAgentRequest(BaseModel):
    """Agent HTML parser request"""

    html: Optional[str]
    url: Optional[str]
    parser_engine: Optional[Literal["bs4", "trafilatura"]] = "trafilatura"
    parser_kwargs: Optional[dict] = {}


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
    extras: Optional[List[BaseEntityAnnotation]]


class EntityAnnotationWithExtras(EntityAnnotation):
    extras: Optional[List[EntityAnnotation]]


AnyBaseAnnotation = Union[BaseEntityAnnotation, BaseEntityAnnotationWithExtras]
AnyAnnotation = Union[EntityAnnotation, EntityAnnotationWithExtras]


class EntityExtractionAgentResponse(BaseModel):
    annotations: Optional[List[AnyAnnotation]] = []
    unlisted_annotations: Optional[List[AnyBaseAnnotation]] = []
    parsed_html: Optional[str]
    lang: str
    timestamp: datetime


def preprocess_text(text: str):
    text = preprocessing.add_trailing_period(text)
    text = preprocessing.replace_unprocessable_chars(text)

    return text


def preprocess_html(
    html: Union[bytes, str], engine: Literal["bs4", "trafilatura"], **engine_kwargs
):
    if engine == "bs4":
        text = preprocessing.parse_html_bs4(html, **engine_kwargs)
    elif engine == "trafilatura":
        text = preprocessing.parse_html_trafilatura(html, **engine_kwargs)
    else:
        raise ValueError(f"engine must be either 'bs4' or 'trafilatura', not {engine}")

    text = preprocess_text(text)

    return text


def preprocess_url(
    url: str, html_engine: Literal["bs4", "trafilatura"], **html_engine_kwargs
):
    raw_html = requests.get(url).content
    text = preprocess_html(raw_html, html_engine, **html_engine_kwargs)

    return text


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
            types=entities.types(entity_idx, variety_idx),
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
    parsed_html: str,
    include_extras: bool = True,
):
    unlisted_annotations = []
    listed_annotations = []

    for idx in range(entities.count_entities()):
        top_annotation = unpack_annotation(entities, idx, 0)
        full_annotation_kwargs = top_annotation.dict()

        if include_extras:
            extra_annotations = []
            listed_annotation_class = EntityAnnotationWithExtras
            unlisted_annotation_class = BaseEntityAnnotationWithExtras

            for variety_idx in range(1, entities.count_entity_variants(idx)):
                extra_annotation = unpack_annotation(entities, idx, variety_idx)
                extra_annotations.append(extra_annotation)

            full_annotation_kwargs["extras"] = extra_annotations
        else:
            listed_annotation_class = EntityAnnotation
            unlisted_annotation_class = BaseEntityAnnotation

        if top_annotation.has_wikidata:
            full_annotation = listed_annotation_class(**full_annotation_kwargs)
            listed_annotations.append(full_annotation)
        else:
            full_annotation = unlisted_annotation_class(**full_annotation_kwargs)
            unlisted_annotations.append(full_annotation)

    return EntityExtractionAgentResponse(
        annotations=listed_annotations,
        unlisted_annotations=unlisted_annotations,
        parsed_html=parsed_html,
        lang="en",
        timestamp=datetime.utcnow(),
    )


@api_router.post("/")
async def extract(payload: EntityExtractionAgentRequest):
    text = ""
    n_main_args = sum(
        int(bool(pl_value)) for pl_value in [payload.text, payload.html, payload.url]
    )

    if n_main_args != 1:
        raise HTTPException(status_code=400, detail="Provide only text, html or url")
    elif payload.text:
        text = preprocess_text(payload.text)
    elif payload.html:
        try:
            text = preprocess_html(
                payload.html, payload.parser_engine, **payload.parser_kwargs
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    elif payload.url:
        try:
            text = preprocess_url(
                payload.url, payload.parser_engine, **payload.parser_kwargs
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    request_data = EntityExtractionServiceRequest(texts=[text]).dict()

    try:
        response = requests.post(
            server_settings.entity_extraction_url, json=request_data
        )
        entities = response.json()
        logger.debug(entities)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Oh no, Entity Extraction server is down! {e}"
        )

    entities = EntityExtractionServiceResponse(**entities)
    entities = unpack_entity_extraction_service_response(
        entities, parsed_html=text, include_extras=payload.include_extras
    )

    if payload.html and payload.attach_parsed_html:
        exclude = set()
    else:
        exclude = {"parsed_html"}

    return entities.dict(exclude=exclude)


@api_router.post("/parse_html")
async def parse_html(payload: HtmlParserAgentRequest):
    text = ""

    if payload.html and payload.url:
        raise HTTPException(status_code=400, detail="Provide only html or url")
    elif payload.html:
        try:
            text = preprocess_html(
                payload.html, payload.parser_engine, **payload.parser_kwargs
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    elif payload.url:
        try:
            text = preprocess_url(
                payload.url, payload.parser_engine, **payload.parser_kwargs
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    return text


@api_router.post("/parse_html_file")
async def parse_html_file(html_file: UploadFile):
    contents = await html_file.read()
    text = preprocess_html(contents, "trafilatura")

    return text


app.include_router(api_router)
