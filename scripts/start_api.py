import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath("."))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn

if __name__ == "__main__":
    host = os.environ.get("EMS_API_HOST", "0.0.0.0")
    port = int(os.environ.get("EMS_API_PORT", "8000"))
    uvicorn.run("src.api.main:app", host=host, port=port, reload=True, reload_dirs=["src"])
