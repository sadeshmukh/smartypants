# Vision Assistant

A live, interactive scene-description assistant for the visually impaired, built on an NVIDIA Jetson Orin Nano. A browser captures webcam video and streams it to the Jetson over the local network, where two GPU-accelerated paths run side by side: continuous YOLOv8 object detection (TensorRT-accelerated, with a live on/off toggle to compare FPS) overlaid on the video in real time, and an on-demand vision-language model (Qwen3.5-2B) that produces a spoken, natural-language description of the scene via the browser's built-in text-to-speech.

## Architecture

```
Laptop browser (camera + UI)
   │  HTTPS + WebSocket, local wifi
   ▼
FastAPI server (server.py, on the Jetson)
   ├── /ws/detect     continuous fast path
   │     detector_manager.py → ObjectDetectionToolkit (YOLOv8)
   │     keeps a TensorRT-enabled and a plain PyTorch instance both
   │     warm in GPU memory so the on/off toggle switches instantly
   │
   └── /api/describe  on-demand slow path
         vlm_client.py → local llama-server (Qwen3.5-2B, vision-capable)
         returns a one-sentence caption for the current frame
   ▼
Browser draws box overlays on a <canvas> over the live <video>,
and speaks captions aloud via window.speechSynthesis
```

Only box coordinates (not annotated images) are sent back over the WebSocket, keeping the fast path low-latency. No camera, microphone, or speaker hardware is needed on the Jetson itself — the laptop's webcam is the camera, and the browser is the speaker.
