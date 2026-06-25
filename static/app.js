const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const capture = document.getElementById("capture");
const trtToggle = document.getElementById("trtToggle");
const connStatus = document.getElementById("connStatus");
const inferenceMsEl = document.getElementById("inferenceMs");
const fpsEl = document.getElementById("fps");
const modeBadge = document.getElementById("modeBadge");
const chatLog = document.getElementById("chatLog");
const voiceStatus = document.getElementById("voiceStatus");
const fallbackForm = document.getElementById("fallbackForm");
const fallbackInput = document.getElementById("fallbackInput");
const debugToggle = document.getElementById("debugToggle");
const debugPanel = document.getElementById("debugPanel");
const interimTranscriptEl = document.getElementById("interimTranscript");
const transcriptLog = document.getElementById("transcriptLog");

const SEND_INTERVAL_MS = 120; // ~8 fps capture, deliberately below the WS round-trip budget
const CAPTURE_W = 640;
const CAPTURE_H = 480;
const WAKE_PHRASE = "hey assistant"; // one-line tweak if it's hard to hear live

capture.width = CAPTURE_W;
capture.height = CAPTURE_H;

let ws = null;
let inFlight = false;

function showKickOverlay() {
  document.getElementById("kickOverlay").classList.remove("hidden");
  if (recognition) { try { recognition.stop(); } catch (_) {} }
  speechSynthesis.cancel();
  setVoiceStatus("Session ended by host", "unsupported");
}

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
  ws.onclose = (ev) => {
    if (ev.code === 4001) { showKickOverlay(); return; }
    connStatus.textContent = "disconnected — retrying…";
    setTimeout(connectWS, 1500);
  };
  ws.onerror = () => { connStatus.textContent = "connection error"; };

  ws.onmessage = (ev) => {
    inFlight = false;
    const msg = JSON.parse(ev.data);
    if (msg.type === "admin_disconnect") { showKickOverlay(); return; }
    drawOverlay(msg);
    inferenceMsEl.textContent = `${msg.inference_ms.toFixed(1)} ms`;
    fpsEl.textContent = msg.fps.toFixed(1);
    modeBadge.textContent = `TensorRT: ${msg.tensorrt ? "ON" : "OFF"}`;
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

function addChatEntry(role, text) {
  const li = document.createElement("li");
  li.className = role === "user" ? "msg-user" : "msg-assistant";
  const time = document.createElement("time");
  time.textContent = new Date().toLocaleTimeString();
  const label = document.createElement("span");
  label.className = "msg-label";
  label.textContent = role === "user" ? "🗣️ You" : "🤖 Assistant";
  const body = document.createElement("p");
  body.textContent = text;
  li.appendChild(time);
  li.appendChild(label);
  li.appendChild(body);
  chatLog.prepend(li);
}

function setVoiceStatus(text, cls) {
  voiceStatus.textContent = text;
  voiceStatus.className = `voice-status ${cls}`;
}

// ---- Debug: live transcript of what the browser's speech recognizer hears ----
function applyDebugVisibility() {
  debugPanel.style.display = debugToggle.checked ? "block" : "none";
}
debugToggle.addEventListener("change", applyDebugVisibility);
applyDebugVisibility();

function setInterimTranscript(text) {
  interimTranscriptEl.textContent = text || "(nothing yet)";
}

function logFinalTranscript(text, heardWake) {
  setInterimTranscript("");
  const li = document.createElement("li");
  li.className = heardWake ? "heard-wake" : "";
  const time = document.createElement("time");
  time.textContent = new Date().toLocaleTimeString();
  li.appendChild(time);
  li.appendChild(document.createTextNode(` "${text}"${heardWake ? " — wake phrase detected" : ""}`));
  transcriptLog.prepend(li);
}

// ---- Voice Q&A (shared by wake-word and the manual text fallback) ----
async function askAboutScene(question) {
  addChatEntry("user", question);
  setVoiceStatus("🧠 Thinking…", "thinking");
  try {
    const res = await fetch("/api/describe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) throw new Error(`server returned ${res.status}`);
    const { caption } = await res.json();
    addChatEntry("assistant", caption);
    setVoiceStatus("🔊 Speaking…", "speaking");
    await speak(caption);
  } catch (err) {
    addChatEntry("assistant", `(error: ${err.message})`);
  } finally {
    setVoiceStatus('👂 Listening for "Hey Assistant"…', "idle");
    resumeRecognition();
  }
}

function speak(text) {
  return new Promise((resolve) => {
    const utter = new SpeechSynthesisUtterance(text);
    utter.onend = resolve;
    utter.onerror = resolve;
    speechSynthesis.speak(utter);
  });
}

// ---- Manual fallback (demo safety net if voice recognition is flaky) ----
fallbackForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = fallbackInput.value.trim();
  if (!q) return;
  fallbackInput.value = "";
  askAboutScene(q);
});

// ---- Wake-word always-listening voice loop ----
const SpeechRecognitionImpl = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let busy = false; // true while thinking/speaking — ignore wake phrase, recognition is paused anyway

function resumeRecognition() {
  busy = false;
  if (recognition) {
    try { recognition.start(); } catch (_e) { /* already started */ }
  }
}

function setupVoice() {
  if (!SpeechRecognitionImpl) {
    setVoiceStatus("⌨️ Voice not supported in this browser — use the text box below", "unsupported");
    return;
  }
  recognition = new SpeechRecognitionImpl();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = "en-US";

  recognition.onresult = (event) => {
    // Debug transcript: show interim words live, log every final utterance,
    // independent of whether it matched the wake phrase — this runs even if
    // we're about to ignore the result below, so you can see exactly what
    // the recognizer heard (useful for tuning WAKE_PHRASE).
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const r = event.results[i];
      if (r.isFinal) {
        const text = r[0].transcript.trim();
        logFinalTranscript(text, text.toLowerCase().includes(WAKE_PHRASE));
      } else {
        interim += r[0].transcript;
      }
    }
    if (interim) setInterimTranscript(interim);

    if (busy) return;
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const result = event.results[i];
      if (!result.isFinal) continue;
      const transcript = result[0].transcript.trim().toLowerCase();
      const idx = transcript.indexOf(WAKE_PHRASE);
      if (idx === -1) continue;

      let question = transcript.slice(idx + WAKE_PHRASE.length).trim();
      if (!question) question = "Describe what you see.";

      busy = true;
      recognition.stop(); // avoid the spoken answer re-triggering the mic
      askAboutScene(question);
      break;
    }
  };

  recognition.onend = () => {
    if (!busy) {
      try { recognition.start(); } catch (_e) { /* ignore */ }
    }
  };
  recognition.onerror = (e) => {
    if (e.error === "no-speech" || e.error === "aborted") return;
    setVoiceStatus(`⚠️ voice error: ${e.error} — retrying…`, "idle");
  };

  recognition.start();
}

(async function init() {
  try {
    await startCamera();
  } catch (err) {
    connStatus.textContent = `camera error: ${err.message}`;
    return;
  }
  connectWS();
  sendLoop();
  setupVoice();
})();
