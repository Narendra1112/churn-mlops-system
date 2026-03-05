from __future__ import annotations

import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any


def _default_workers() -> int:
    cpu = os.cpu_count() or 2
    return min(32, cpu * 4)

_PREDICT_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(os.getenv("PREDICT_WORKERS", _default_workers()))
)

async def run_predict_threadpool(fn: Callable[[], Any]) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_PREDICT_EXECUTOR, fn)
