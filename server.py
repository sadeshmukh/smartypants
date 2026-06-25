"""FastAPI backend for the live scene-description assistant.

Fast path:  WS /ws/detect      -> continuous YOLOv8 detection (TensorRT on/off toggle)
Slow path:  POST /api/describe -> on-demand VLM caption via local llama-server
Admin:      /admin             -> session management dashboard
"""
import asyncio
import json
import time
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import detector_manager
from vlm_client import describe_image, build_qa_prompt, DEFAULT_PROMPT

app = FastAPI(title="Vision Assistant")

# client_id -> {ws, ip, connected_at, frame_count}
connected_clients: dict[str, dict] = {}


@app.middleware("http")
async def no_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.websocket("/ws/detect")
async def ws_detect(ws: WebSocket):
    await ws.accept()
    client_id = str(uuid.uuid4())
    client_ip = (
        ws.headers.get("x-forwarded-for")
        or (ws.client.host if ws.client else "unknown")
    )
    connected_clients[client_id] = {
        "ws": ws,
        "ip": client_ip,
        "connected_at": time.time(),
        "frame_count": 0,
    }
    use_tensorrt = True
    try:
        while True:
            msg = await ws.receive()

            if "bytes" in msg and msg["bytes"] is not None:
                jpeg_bytes = msg["bytes"]
            elif "text" in msg and msg["text"] is not None:
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

            connected_clients[client_id]["frame_count"] += 1

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
    finally:
        connected_clients.pop(client_id, None)


class DescribeRequest(BaseModel):
    image_data_url: str
    question: str | None = None
    prompt: str = DEFAULT_PROMPT


@app.post("/api/describe")
async def api_describe(req: DescribeRequest):
    prompt = build_qa_prompt(req.question) if req.question else req.prompt
    caption = await describe_image(req.image_data_url, prompt)
    return {"caption": caption}


# ---- Admin routes ----

@app.get("/admin")
async def admin_page():
    return FileResponse("static/admin.html")


@app.get("/api/admin/clients")
async def admin_list_clients():
    now = time.time()
    return [
        {
            "id": cid,
            "ip": info["ip"],
            "connected_seconds": int(now - info["connected_at"]),
            "frame_count": info["frame_count"],
        }
        for cid, info in list(connected_clients.items())
    ]


@app.delete("/api/admin/clients/{client_id}")
async def admin_kick_client(client_id: str):
    info = connected_clients.get(client_id)
    if not info:
        raise HTTPException(status_code=404, detail="client not found")
    try:
        await info["ws"].send_text(json.dumps({"type": "admin_disconnect"}))
        await info["ws"].close(code=4001)
    except Exception:
        pass
    connected_clients.pop(client_id, None)
    return {"ok": True}


@app.delete("/api/admin/clients")
async def admin_kick_all():
    ids = list(connected_clients)
    for cid in ids:
        info = connected_clients.get(cid)
        if not info:
            continue
        try:
            await info["ws"].send_text(json.dumps({"type": "admin_disconnect"}))
            await info["ws"].close(code=4001)
        except Exception:
            pass
    connected_clients.clear()
    return {"ok": True, "kicked": len(ids)}


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
    )
