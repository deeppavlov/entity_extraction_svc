from dataclasses import dataclass
from typing import Optional

from environs import Env
from requests import Session

ENDPOINT = "https://api.dandelion.eu/datatxt/nex/v1/"
TEXT_TYPES = ["text", "html", "html_fragment"]
INCLUDE_OPTIONS = [
    "types",
    "categories",
    "abstract",
    "image",
    "lod",
    "alternate_labels",
]
LANG_OPTIONS = ["de", "en", "es", "fr", "it", "pt", "ru", "auto"]

env = Env()
env.read_env()

DANDELION_TOKEN = env.str("DANDELION_TOKEN")


@dataclass
class DandelionResponse:
    status_code: int
    data: Optional[dict]
    error: Optional[str]


def extract_dandelion(
    session: Session,
    text: str,
    text_type: str = "text",
    lang: str = None,
    top_entities: int = 0,
    min_confidence: float = 0.6,
    min_length: int = 2,
    include: list = None,
    token: str = None,
):
    """Extracts entities from a text, html or html fragment.
    Refer to https://dandelion.eu/docs/api/datatxt/nex/v1/#parameters for details

    Args:
        session: requests session
        text: text to extract from
        text_type: "text" for plain text, "html" for full html documents or "html_fragment" for html snippets
        lang: explicitly set text language. Detected automatically by default
        top_entities: return only N most important entities if > 0
        min_confidence: return only entities with confidence above threshold
        min_length: return only entities with min length > N
        include: list of additional info keys which should be returned
        token: Dandelion token. If None, loads it from DANDELION_TOKEN env.
            You can export it via terminal: ``export DANDELION_TOKEN="yourtokenhere"``

    Returns:

    """

    if text_type not in TEXT_TYPES:
        raise Exception(f"Choose input text type from {', '.join(TEXT_TYPES)}")

    if include and set(include) - set(INCLUDE_OPTIONS):
        raise Exception(f"Choose include values from {', '.join(INCLUDE_OPTIONS)}")

    if lang and lang not in LANG_OPTIONS:
        raise Exception(f"Choose lang value from {', '.join(LANG_OPTIONS)}")

    if not (token or DANDELION_TOKEN):
        raise Exception(
            f"No token found. Provide it via 'export DANDELION_TOKEN=\"yourtokenhere\"'"
            " or place it inside .env in repo root"
        )

    url_kwargs = {
        text_type: text,
        "lang": lang or "auto",
        "top_entities": top_entities,
        "min_confidence": min_confidence,
        "min_length": min_length,
        "include": ",".join(include) if include else "",
        "token": token if token else DANDELION_TOKEN,
    }

    response = session.get(ENDPOINT, params=url_kwargs)
    data = error = None
    status_code = response.status_code

    try:
        data = response.json()
    except Exception as e:
        error = f"{type(e)}: {e}"

    return DandelionResponse(status_code=status_code, data=data, error=error)
