from ast import literal_eval
import re

from datasets import load_dataset
import pandas as pd


SCORING_FILE_PATH = "webnlg_scoring.csv"


def _clean_entity_string(text):
    text = text.strip(" \"")
    text = re.sub("_", " ", text)
    return text


def transform_dataset(save=True):
    entex_dataset = []

    for row in load_dataset('web_nlg', 'webnlg_challenge_2017', split="test+train+dev"):
        if row["lex"]["text"]:
            original_entities = []
            cleaned_entities = []

            for tset in row["modified_triple_sets"]["mtriple_set"][0]:
                for ent in tset.split("|")[::2]:
                    # take a, c from "a | b | c"
                    original_entities.append(ent.strip(" "))
                    cleaned_entities.append(_clean_entity_string(ent))

            entex_row = {
                "category": row["category"],
                "size": row["size"],
                "eid": row["eid"],
                "original_entities": list(set(original_entities)),
                "cleaned_entities": list(set(cleaned_entities)),
                "sentences": row["lex"]["text"],
                "text": " ".join(row["lex"]["text"]),
                "dbpedia_links": row["dbpedia_links"],
                "links": row["links"],
            }
            entex_dataset.append(entex_row)

    scoring_df = pd.DataFrame(entex_dataset)
    if save:
        save_scoring_df(scoring_df)

    return scoring_df


def save_scoring_df(scoring_df: pd.DataFrame, path: str = SCORING_FILE_PATH):
    scoring_df.to_csv(path, index=False)


def load_scoring_df(path: str):
    dtype_converters = {
        "original_entities": literal_eval,
        "cleaned_entities": literal_eval,
        "sentences": literal_eval,
        "dbpedia_links": literal_eval,
        "links": literal_eval,
    }
    scoring_df = pd.read_csv(
        path,
        index_col=None,
        converters=dtype_converters
    )
    return scoring_df


if __name__ == '__main__':
    entex_scoring_df = transform_dataset(save=True)
