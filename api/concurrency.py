from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os

MAX_INFLIGHT = int(os.getenv("MAX_INFLIGHT_INFERENCES", "32"))
ACQUIRE_TIMEOUT_S = float(os.getenv("INFERENCE_QUEUE_TIMEOUT_S", "1.0"))


_sema = asyncio.Semaphore(MAX_INFLIGHT)

class BackpressureError(Exception):
    pass

@asynccontextmanager
async def inference_slot():
    try:
        await asyncio.wait_for(_sema.acquire(), timeout=ACQUIRE_TIMEOUT_S)
    except asyncio.TimeoutError as e:
        raise BackpressureError("too many in-flight inferences") from e
    try:
        yield
    finally:
        _sema.release()
