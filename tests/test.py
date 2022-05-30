import json
from pathlib import Path

import pytest
import requests
from dotenv import load_dotenv

from agent.config import ServerSettings


DATASETS_DIR = Path(__file__).parent / "data"


def _load_txt(dataset_dir, filename):
    text_path = DATASETS_DIR / dataset_dir / filename

    with text_path.open("r", encoding="utf-8") as texts_f:
        text = texts_f.read().strip()

    return text


def _load_json(dataset_dir, filename):
    json_path = DATASETS_DIR / dataset_dir / filename

    with json_path.open("r", encoding="utf-8") as json_f:
        json_data = json.load(json_f)

    return json_data


@pytest.fixture
def api_settings():
    # load explicitly because pydantic fails to load anything except ".env"
    load_dotenv("test.env")

    class TestServerSettings(ServerSettings):
        class Config:
            env_file = "test.env"
            env_file_encoding = "utf-8"
            env_nested_delimiter = "__"
            # env_file = "test.env"

    settings = TestServerSettings()
    return settings


@pytest.fixture
def response_json(api_settings, dataset_dir):
    text = _load_txt(dataset_dir, "input.txt")
    response = requests.post(api_settings.agent_url, json={"text": text})
    return response.json()


@pytest.mark.parametrize("dataset_dir", ["mona_lisa", "spacex"])
def test_output(response_json, dataset_dir):
    expected_output = _load_json(dataset_dir, "output.json")

    n_annotations = len(response_json["annotations"])
    expected_n_annotations = len(expected_output["annotations"])
    assert n_annotations == expected_n_annotations

    n_unlisted_annotations = len(response_json["unlisted_annotations"])
    expected_n_unlisted_annotations = len(expected_output["unlisted_annotations"])
    assert n_unlisted_annotations == expected_n_unlisted_annotations
