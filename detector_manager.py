"""Caches warm ObjectDetectionToolkit instances (TensorRT on/off) for the live demo.

Reuses the curriculum's detection class instead of reimplementing YOLO/TensorRT
plumbing — see /Developer/edgeAI/jetson/jetson_object_detection_toolkit.py.
"""
import sys
import time

sys.path.append("/Developer/edgeAI/jetson")
from jetson_object_detection_toolkit import ObjectDetectionToolkit  # noqa: E402

_toolkits = {}


def get_toolkit(use_tensorrt: bool) -> ObjectDetectionToolkit:
    key = "trt" if use_tensorrt else "pt"
    if key not in _toolkits:
        print(f"[detector_manager] loading YOLO toolkit (tensorrt={use_tensorrt})...")
        t0 = time.time()
        _toolkits[key] = ObjectDetectionToolkit(
            "yolo", "cuda", model_path="yolov8n.pt", use_tensorrt=use_tensorrt
        )
        print(f"[detector_manager] ready in {time.time() - t0:.1f}s")
    return _toolkits[key]


def detect(frame, use_tensorrt: bool, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
    toolkit = get_toolkit(use_tensorrt)
    return toolkit.detect(frame, conf_threshold=conf_threshold, iou_threshold=iou_threshold)


if __name__ == "__main__":
    import cv2

    img = cv2.imread("/Developer/models/bus.jpg")
    if img is None:
        raise SystemExit("could not read /Developer/models/bus.jpg")

    print("=== non-TensorRT pass ===")
    r1 = detect(img, use_tensorrt=False)
    print(f"objects={len(r1['boxes'])} inference_ms={r1['inference_time']:.1f} fps={r1['fps']:.1f}")

    print("=== TensorRT pass (first call compiles the engine, can take 1-5 min) ===")
    r2 = detect(img, use_tensorrt=True)
    print(f"objects={len(r2['boxes'])} inference_ms={r2['inference_time']:.1f} fps={r2['fps']:.1f}")

    print("=== TensorRT pass #2 (should be fast, engine cached) ===")
    r3 = detect(img, use_tensorrt=True)
    print(f"objects={len(r3['boxes'])} inference_ms={r3['inference_time']:.1f} fps={r3['fps']:.1f}")
