from dataclasses import dataclass
from typing import Optional

from requests import Session

ENDPOINT = "https://api.dandelion.eu/datatxt/nex/v1/"
TEXT_TYPES = ["text", "html"]


@dataclass
class DpResponse:
    status_code: int
    data: Optional[dict]
    error: Optional[str]


def extract_dp(
    session: Session,
    url: str,
    text: str,
    text_type: str = "text",
    attach_parsed_html: bool = False,
    include_extras: bool = True,
):
    """Extracts entities from a text or html.

    Args:
        session: requests session
        url: DeepPavlov entity extraction url
        text: text to extract from
        text_type: "text" for plain text, "html" for full html documents or "html_fragment" for html snippets
        attach_parsed_html: include parsed html in result
        include_extras: include extra entity variants

    Returns:

    """

    if text_type not in TEXT_TYPES:
        raise Exception(f"Choose input text type from {', '.join(TEXT_TYPES)}")

    json_data = {
        text_type: text,
        "attach_parsed_html": attach_parsed_html,
        "include_extras": include_extras,
    }

    response = session.post(url, json=json_data)
    data = error = None
    status_code = response.status_code

    try:
        data = response.json()
    except Exception as e:
        error = f"{type(e)}: {e}"

    return DpResponse(status_code=status_code, data=data, error=error)
