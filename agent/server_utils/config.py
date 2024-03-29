from typing import Optional

from pydantic import BaseSettings


class ServerSettings(BaseSettings):
    agent_url: str
    entity_extraction_url: str
    wiki_parser_url: str
    collect_stats: Optional[bool] = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
