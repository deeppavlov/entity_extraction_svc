from environs import Env
from requests import Session
from tqdm import tqdm

ENDPOINT = "https://api.dandelion.eu/datatxt/nex/v1/"
TEXT_TYPES = ["text", "html", "html_fragment"]
INCLUDE_OPTIONS = ["types", "categories", "abstract", "image", "lod", "alternate_labels"]
LANG_OPTIONS = ["de", "en", "es", "fr", "it", "pt", "ru", "auto"]

env = Env()
env.read_env()

DANDELION_TOKEN = env.str("DANDELION_TOKEN")


def extract_dandelion(
    session: Session,
    text: str,
    text_type: str = "text",
    lang: str = None,
    top_entities: int = 0,
    min_confidence: float = 0.6,
    min_length: int = 2,
    include: list = None,
):
    """Extract entities from a text, html or html fragment.
    Refer to https://dandelion.eu/docs/api/datatxt/nex/v1/#parameters for details

    :param session: requests session
    :param text: text to extract from
    :param text_type: "text" for plain text, "html" for full html documents or "html_fragment" for html snippets
    :param lang: explicitly set text language. Detected automatically by default
    :param top_entities: return only N most important entities if > 0
    :param min_confidence: return only entities with confidence above threshold
    :param min_length: return only entities with min length > N
    :param include: list of additional info keys which should be returned
    :return:
    """

    if text_type not in TEXT_TYPES:
        raise Exception(f"Choose input text type from {', '.join(TEXT_TYPES)}")

    if include and set(include) - set(INCLUDE_OPTIONS):
        raise Exception(f"Choose include values from {', '.join(INCLUDE_OPTIONS)}")

    if lang and lang not in LANG_OPTIONS:
        raise Exception(f"Choose lang value from {', '.join(LANG_OPTIONS)}")

    url_kwargs = {
        text_type: text,
        "lang": lang or "auto",
        "top_entities": top_entities,
        "min_confidence": min_confidence,
        "min_length": min_length,
        "include": ",".join(include) if include else "",
        "token": DANDELION_TOKEN,
    }

    response = session.get(ENDPOINT, params=url_kwargs)
    data = response.json()

    return data


def extract_batch_dandelion(texts: list, **kwargs):
    results = []

    with Session() as sess:
        for text in tqdm(texts):
            entities = extract_dandelion(sess, text, **kwargs)
            results.append(entities)

    return results


# if __name__ == '__main__':
    # with Session() as sess:
    #     entities = extract_dandelion(sess, "Mona Lisa is located in Louvre")
    #     print(entities)
