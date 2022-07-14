import re
from typing import Union

import trafilatura
from bs4 import BeautifulSoup


CORRECT_TRAILING_PUNCTUATION = [".", ",", "?", "!"]
UNPROCESSABLE_CHAR_MAP = {
    "’": "'",
    "”": '"',
    "â€™": "'",
    "â€œ": '"',
    "â€\x9d": '"',
}


def add_trailing_period(text: str):
    """Adds `.` to the end of the text if it doesn't end with one of the punctuation marks

    Args:
        text: raw text

    Returns:
        text with fixed trailing punctuation
    """
    if text and text[-1] not in CORRECT_TRAILING_PUNCTUATION:
        text = f"{text}."

    return text


def replace_unprocessable_chars(text: str):
    """Replaces unprocessable chars with ascii equivalents according to the mapping

    Args:
        text: raw text

    Returns:
        text without unprocessable chars
    """
    for old_symb, new_symb in UNPROCESSABLE_CHAR_MAP.items():
        text = text.replace(old_symb, new_symb)

    return text


def remove_tag_spans(text: str):
    """Removes `<...>` spans from text

    Args:
        text: raw text

    Returns:
        text without tag-like spans
    """
    return re.sub('<[^<]+>', "", text).strip()


def parse_html_bs4(raw_html: Union[bytes, str], parser: str = "html.parser"):
    """Extracts text from html using BeautifulSoup

    Args:
        raw_html: raw html string
        parser: soup parser. See more at https://www.crummy.com/software/BeautifulSoup/bs4/doc/#installing-a-parser

    Returns:
        text content of the html
    """
    soup = BeautifulSoup(raw_html, parser)
    doc = soup.get_text(separator=" ", strip=True)
    doc = re.sub(r"\s+", " ", doc)

    return doc


def parse_html_trafilatura(raw_html: Union[bytes, str], **kwargs):
    """Extracts text from html using trafilatura

    Args:
        raw_html: raw html string
        kwargs: trafilatura extraction arguments

    Returns:
        text content of the html
    """
    return trafilatura.extract(raw_html, **kwargs)
