from datetime import datetime
from uuid import UUID

from motor.motor_asyncio import AsyncIOMotorClient


class StatsDatabase:
    def __init__(
        self,
        username: str,
        password: str,
        host: str,
        port: int,
        database: str,
        auth_database: str = "",
    ):
        self.client = AsyncIOMotorClient(
            f"mongodb://{username}:{password}@{host}:{port}/{auth_database}?uuidRepresentation=standard"
        )
        self.db = self.client[database]

    async def save_request(self, session_id: UUID, body: dict):
        await self.db.requests.insert_one(
            {"_id": session_id, "timestamp": datetime.utcnow(), "body": body}
        )

    async def save_response(self, session_id: UUID, body: dict):
        await self.db.responses.insert_one(
            {"_id": session_id, "timestamp": datetime.utcnow(), "body": body}
        )
