#!/usr/bin/env bash
# Run this BEFORE judging starts (morning of demo, not live).
# Forces the TensorRT engine to compile/load and confirms the local VLM is up.
set -euo pipefail
cd "$(dirname "$0")"

echo "== Warming YOLO toolkit (TensorRT + PyTorch) =="
docker exec -w /workspace/vision-assistant jetson-dev python3 -c "
import detector_manager, cv2
img = cv2.imread('/Developer/models/bus.jpg')
for trt in (False, True):
    r = detector_manager.detect(img, use_tensorrt=trt)
    print(f'tensorrt={trt} objects={len(r[\"boxes\"])} inference_ms={r[\"inference_time\"]:.1f} fps={r[\"fps\"]:.1f}')
"

echo
echo "== Checking llama-server (local VLM, port 8080) =="
if curl -sf -m 5 http://localhost:8080/v1/models >/dev/null; then
  echo "llama-server is up."
else
  echo "llama-server is NOT responding on :8080."
  echo "Try: sjsujetsontool llama qwen2b bg"
  echo "Fallback (exact paths confirmed on this machine):"
  cat <<'EOF'
  docker exec -d jetson-dev /opt/llama.cpp/build_cuda/bin/llama-server \
    -m /models/llama_cache/models--unsloth--Qwen3.5-2B-MTP-GGUF/snapshots/e05864f8066d874d5f85aaff007ae57a2a7d1efe/Qwen3.5-2B-Q4_K_S.gguf \
    --mmproj /models/llama_cache/models--unsloth--Qwen3.5-2B-MTP-GGUF/snapshots/e05864f8066d874d5f85aaff007ae57a2a7d1efe/mmproj-BF16.gguf \
    --host 0.0.0.0 --port 8080 -ngl 99
EOF
  exit 1
fi

echo
echo "Prewarm complete. Start server.py now and do NOT restart it before the demo."
