import asyncio
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def main():
    saver_ctx = AsyncSqliteSaver.from_conn_string("test.db")
    print(type(saver_ctx))
    async with saver_ctx as saver:
        print(type(saver))

asyncio.run(main())
