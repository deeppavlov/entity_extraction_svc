from pydantic import BaseSettings


class StatsCollectorSettings(BaseSettings):
    stats_db_host: str
    stats_db_port: int
    stats_db_name: str
    stats_db_username: str
    stats_db_password: str
    stats_db_auth_database: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
