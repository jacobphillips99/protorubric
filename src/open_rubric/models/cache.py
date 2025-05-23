"""
Cache for model requests and responses. Uses a sqlite database to store the requests and responses.
Optionally invalidate the cache by setting the invalidate_cache flag to True.

TODO: make a LRU or size limit
"""

import time
import typing as t
from contextlib import asynccontextmanager

import aiosqlite

from open_rubric.models.model_types import ModelRequest, ModelResponse

DB_PATH = "assets/request_cache.db"


@asynccontextmanager
async def _conn() -> t.AsyncGenerator[aiosqlite.Connection, None]:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """create table if not exists cache(
                              key text primary key,
                              response text not null,
                              ts real not null
                            );"""
        )
        await db.commit()
        yield db


class AsyncRequestCache:
    def __init__(self) -> None:
        pass

    async def aget(self, req: ModelRequest) -> ModelResponse | None:
        k = req.to_hash()
        async with _conn() as db:
            cursor = await db.execute("select response, ts from cache where key=?;", (k,))
            row = await cursor.fetchone()
        if not row:
            return None
        blob = row[0]
        print("Cache hit!")
        model_response: ModelResponse = ModelResponse.model_validate_json(blob)
        return model_response

    async def aput(self, req: ModelRequest, resp: ModelResponse) -> None:
        async with _conn() as db:
            await db.execute(
                "insert or replace into cache(key, response, ts) values(?,?,?);",
                (req.to_hash(), resp.model_dump_json(), time.time()),
            )
            await db.commit()
        print("Cache miss!")

    async def inspect(self) -> dict[str, t.Any]:
        async with _conn() as db:
            cursor = await db.execute("select count(*) from cache;")
            count = await cursor.fetchone()
            return {"count": count[0]}


ASYNC_REQUEST_CACHE = AsyncRequestCache()
