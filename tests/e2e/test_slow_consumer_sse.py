"""
SSE bridge behavior under subscriber backpressure (unit-level via bridge API).
"""

import asyncio
import queue as queue_std

from apis.detection_stream import DetectionSSEBridge


def test_sse_bridge_overflow_before_drop():
    async def _run():
        bridge = DetectionSSEBridge(
            queue_std.Queue(),
            subscriber_queue_maxsize=1,
        )
        await bridge.start()
        try:
            q = bridge.subscribe()
            await bridge._broadcast({"_seq": 1, "eventType": "TEST"})
            await bridge._broadcast({"_seq": 2, "eventType": "TEST"})
            ov = bridge._overflow[id(q)]
            assert len(ov) >= 1
        finally:
            await bridge.stop()

    asyncio.run(_run())
