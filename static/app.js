const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const capture = document.getElementById("capture");
const trtToggle = document.getElementById("trtToggle");
const describeBtn = document.getElementById("describeBtn");
const connStatus = document.getElementById("connStatus");
const inferenceMsEl = document.getElementById("inferenceMs");
const fpsEl = document.getElementById("fps");
const modeBadge = document.getElementById("modeBadge");
const captionLog = document.getElementById("captionLog");

const SEND_INTERVAL_MS = 120; // ~8 fps capture, deliberately below the WS round-trip budget
const CAPTURE_W = 640;
const CAPTURE_H = 480;

capture.width = CAPTURE_W;
capture.height = CAPTURE_H;

let ws = null;
let inFlight = false;
let latestResult = null;

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: CAPTURE_W, height: CAPTURE_H },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  overlay.width = video.clientWidth;
  overlay.height = video.clientHeight;
  window.addEventListener("resize", () => {
    overlay.width = video.clientWidth;
    overlay.height = video.clientHeight;
  });
}

function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/detect`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    connStatus.textContent = "connected";
    sendMode();
  };
  ws.onclose = () => {
    connStatus.textContent = "disconnected — retrying…";
    setTimeout(connectWS, 1500);
  };
  ws.onerror = () => { connStatus.textContent = "connection error"; };

  ws.onmessage = (ev) => {
    inFlight = false;
    const result = JSON.parse(ev.data);
    latestResult = result;
    drawOverlay(result);
    inferenceMsEl.textContent = `${result.inference_ms.toFixed(1)} ms`;
    fpsEl.textContent = result.fps.toFixed(1);
    modeBadge.textContent = `TensorRT: ${result.tensorrt ? "ON" : "OFF"}`;
  };
}

function sendMode() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "set_mode", tensorrt: trtToggle.checked }));
  }
}
trtToggle.addEventListener("change", sendMode);

function drawOverlay(result) {
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const scaleX = overlay.width / result.frame_w;
  const scaleY = overlay.height / result.frame_h;
  ctx.lineWidth = 2;
  ctx.font = "14px system-ui";
  result.boxes.forEach((box, i) => {
    const [x1, y1, x2, y2] = box;
    const label = `${result.labels[i]} ${(result.scores[i] * 100).toFixed(0)}%`;
    ctx.strokeStyle = "#39d353";
    ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
    const textW = ctx.measureText(label).width + 6;
    ctx.fillStyle = "#39d353";
    ctx.fillRect(x1 * scaleX, y1 * scaleY - 18, textW, 18);
    ctx.fillStyle = "#0b0f14";
    ctx.fillText(label, x1 * scaleX + 3, y1 * scaleY - 4);
  });
}

function sendLoop() {
  setInterval(() => {
    if (!ws || ws.readyState !== WebSocket.OPEN || inFlight) return;
    const ctx = capture.getContext("2d");
    ctx.drawImage(video, 0, 0, CAPTURE_W, CAPTURE_H);
    capture.toBlob((blob) => {
      if (!blob) return;
      blob.arrayBuffer().then((buf) => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(buf);
          inFlight = true;
        }
      });
    }, "image/jpeg", 0.6);
  }, SEND_INTERVAL_MS);
}

function addCaptionLogEntry(caption) {
  const li = document.createElement("li");
  const time = document.createElement("time");
  time.textContent = new Date().toLocaleTimeString();
  li.appendChild(time);
  li.appendChild(document.createTextNode(caption));
  captionLog.prepend(li);
}

async function describeScene() {
  describeBtn.disabled = true;
  describeBtn.textContent = "Thinking…";
  try {
    const ctx = capture.getContext("2d");
    ctx.drawImage(video, 0, 0, CAPTURE_W, CAPTURE_H);
    const image_data_url = capture.toDataURL("image/jpeg", 0.8);
    const res = await fetch("/api/describe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_data_url }),
    });
    if (!res.ok) throw new Error(`server returned ${res.status}`);
    const { caption } = await res.json();
    addCaptionLogEntry(caption);
    speechSynthesis.speak(new SpeechSynthesisUtterance(caption));
  } catch (err) {
    addCaptionLogEntry(`(error: ${err.message})`);
  } finally {
    describeBtn.disabled = false;
    describeBtn.textContent = "🔊 Describe Scene";
  }
}
describeBtn.addEventListener("click", describeScene);

(async function init() {
  try {
    await startCamera();
  } catch (err) {
    connStatus.textContent = `camera error: ${err.message}`;
    return;
  }
  connectWS();
  sendLoop();
})();
