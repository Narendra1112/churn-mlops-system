from __future__ import annotations

import os
import asyncio

INFERENCE_TIMEOUT_S = float(os.getenv("INFERENCE_TIMEOUT_S", "1.0"))
print("INFERENCE_TIMEOUT_S =", INFERENCE_TIMEOUT_S)


class InferenceTimeoutError(Exception):
    pass

async def with_inference_timeout(coro):
    try:
        return await asyncio.wait_for(coro, timeout=INFERENCE_TIMEOUT_S)
    except asyncio.TimeoutError as e:
        raise InferenceTimeoutError("inference timed out") from e
