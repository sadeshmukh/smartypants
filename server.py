"""FastAPI backend for the live scene-description assistant.

Fast path:  WS /ws/detect      -> continuous YOLOv8 detection (TensorRT on/off toggle)
Slow path:  POST /api/describe -> on-demand VLM caption via local llama-server
"""
import asyncio
import json
import time

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import detector_manager
from vlm_client import describe_image, DEFAULT_PROMPT

app = FastAPI(title="Vision Assistant")


@app.websocket("/ws/detect")
async def ws_detect(ws: WebSocket):
    await ws.accept()
    use_tensorrt = True
    try:
        while True:
            msg = await ws.receive()

            if "bytes" in msg and msg["bytes"] is not None:
                jpeg_bytes = msg["bytes"]
            elif "text" in msg and msg["text"] is not None:
                # control message, e.g. {"type":"set_mode","tensorrt":true}
                try:
                    ctrl = json.loads(msg["text"])
                    if ctrl.get("type") == "set_mode":
                        use_tensorrt = bool(ctrl.get("tensorrt", True))
                except (ValueError, TypeError):
                    pass
                continue
            else:
                continue

            buf = np.frombuffer(jpeg_bytes, np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            t0 = time.time()
            results = await asyncio.to_thread(detector_manager.detect, frame, use_tensorrt)
            rtt_ms = (time.time() - t0) * 1000

            h, w = frame.shape[:2]
            await ws.send_text(json.dumps({
                "boxes": [list(map(float, b)) for b in results["boxes"]],
                "labels": list(results["class_names"]),
                "scores": [float(s) for s in results["scores"]],
                "inference_ms": float(results["inference_time"]),
                "fps": float(results["fps"]),
                "rtt_ms": rtt_ms,
                "tensorrt": use_tensorrt,
                "frame_w": w,
                "frame_h": h,
            }))
    except WebSocketDisconnect:
        pass


class DescribeRequest(BaseModel):
    image_data_url: str
    prompt: str = DEFAULT_PROMPT


@app.post("/api/describe")
async def api_describe(req: DescribeRequest):
    caption = await describe_image(req.image_data_url, req.prompt)
    return {"caption": caption}


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
    )
