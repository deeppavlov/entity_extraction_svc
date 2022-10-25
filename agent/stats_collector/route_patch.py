import json
from uuid import uuid4
from typing import Callable

from fastapi import Request, Response, HTTPException
from fastapi.routing import APIRoute

from agent.stats_collector.config import StatsCollectorSettings
from agent.stats_collector.db import StatsDatabase


stats_collector_settings = StatsCollectorSettings()
stats_db = StatsDatabase(
    stats_collector_settings.stats_db_username,
    stats_collector_settings.stats_db_password,
    stats_collector_settings.stats_db_host,
    stats_collector_settings.stats_db_port,
    stats_collector_settings.stats_db_name,
    auth_database=stats_collector_settings.stats_db_auth_database,
)


class StatsCollectorRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            session_id = uuid4()
            await stats_db.save_request(
                session_id, request.scope["path"], await request.json()
            )

            try:
                response = await original_route_handler(request)
                await stats_db.save_response(session_id, json.loads(response.body))
            except HTTPException as e:
                e_data = {
                    "status_code": e.status_code,
                    "detail": e.detail,
                    "headers": e.headers,
                }
                await stats_db.save_exception(session_id, e_data)
                raise e

            return response

        return custom_route_handler
