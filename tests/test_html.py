import pickle
import requests
from pathlib import Path
from deeppavlov.core.data.utils import simple_download


EX_URL = "http://0.0.0.0:9103/entity_extraction"
DATASET_FLNAME = "test_html_pages.pickle"
TEST_HTML_DATASET_URL = f"http://files.deeppavlov.ai/deeppavlov_data/entity_linking/{DATASET_FLNAME}"
DATASETS_DIR = Path(__file__).parent / "data"

simple_download(TEST_HTML_DATASET_URL, DATASETS_DIR / DATASET_FLNAME)

with open(DATASETS_DIR / DATASET_FLNAME, 'rb') as fl:
    data = pickle.load(fl)
    n = 0
    for page_title in data:
        print(n)
        page_content = data[page_title]
        res = requests.post("http://0.0.0.0:9103/entity_extraction",
                            json={"texts": [page_content]}).json()
        substr_list = res["entity_substr"][0]
        offsets_list = res["entity_offsets"][0]
        ids_list = res["entity_ids"][0]
        tags_list = res["entity_tags"][0]
        conf_list = res["entity_conf"][0]
        pages_list = res["entity_pages"][0]
        image_links_list = res["image_links"][0]
        categories_list = res["categories"][0]
        first_par_list = res["first_paragraphs"][0]
        dbpedia_types_list = res["dbpedia_types"][0]
        length = len(substr_list)
        for elem_name, elem in [("offsets", offsets_list), ("ids", ids_list), ("tags", tags_list), ("conf", conf_list),
                                ("pages", pages_list), ("image_links", image_links_list), ("categories", categories_list),
                                ("first_par", first_par_list), ("dbpedia_types", dbpedia_types_list)]:
            if len(elem) != length:
                print(f"NOT EQUAL LISTS, {page_title} --- {elem_name} {elem}")
        for ids, tags, conf, pages, image_links, categories, first_par, dbpedia_types in \
                zip(ids_list, tags_list, conf_list, pages_list, image_links_list, categories_list,
                    first_par_list, dbpedia_types_list):
            length = len(ids)
            for elem_name, elem in [("tags", tags), ("conf", conf), ("pages", pages), ("image_links", image_links),
                                    ("categories", categories), ("first_par", first_par), ("dbpedia_types", dbpedia_types)]:
                if len(elem) != length:
                    print(f"NOT EQUAL ELEM, {page_title} {ids} --- {elem_name} {elem}")
        n += 1
