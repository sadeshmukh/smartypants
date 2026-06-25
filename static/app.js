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
const videoWrap = document.getElementById("videoWrap");
const sparklineCanvas = document.getElementById("sparklineCanvas");

// New DOM Elements for presets, narrator, voice setting and telemetry
const narrateToggle = document.getElementById("narrateToggle");
const voiceSelect = document.getElementById("voiceSelect");
const rateRange = document.getElementById("rateRange");
const pitchRange = document.getElementById("pitchRange");
const rateVal = document.getElementById("rateVal");
const pitchVal = document.getElementById("pitchVal");

const statCpu = document.getElementById("statCpu");
const statGpu = document.getElementById("statGpu");
const statTemp = document.getElementById("statTemp");
const statMem = document.getElementById("statMem");

const SEND_INTERVAL_MS = 120; // ~8 fps capture, deliberately below the WS round-trip budget
const CAPTURE_W = 640;
const CAPTURE_H = 480;
const WAKE_PHRASE = "hey assistant"; // one-line tweak if it's hard to hear live

capture.width = CAPTURE_W;
capture.height = CAPTURE_H;

let ws = null;
let inFlight = false;
let lastSendTime = 0;

// Speech synthesis settings
let selectedVoice = null;
let ttsRate = 1.0;
let ttsPitch = 1.0;

// Sparkline telemetry arrays
const inferenceHistory = [];
const rttHistory = [];
const MAX_HISTORY_LEN = 40;

function showKickOverlay() {
  document.getElementById("kickOverlay").classList.remove("hidden");
  if (recognition) { try { recognition.stop(); } catch (_) {} }
  speechSynthesis.cancel();
  setVoiceStatus("Session ended by host", "unsupported");
  updateGlow("idle");
}

function updateGlow(state) {
  if (videoWrap) {
    videoWrap.className = `video-wrap state-${state}`;
  }
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
    const clientRtt = Date.now() - lastSendTime;
    const msg = JSON.parse(ev.data);
    if (msg.type === "admin_disconnect") { showKickOverlay(); return; }
    drawOverlay(msg);
    inferenceMsEl.textContent = `${msg.inference_ms.toFixed(1)} ms`;
    fpsEl.textContent = msg.fps.toFixed(1);
    modeBadge.textContent = `TensorRT: ${msg.tensorrt ? "ON" : "OFF"}`;

    // Add values to history
    inferenceHistory.push(msg.inference_ms);
    rttHistory.push(clientRtt);
    if (inferenceHistory.length > MAX_HISTORY_LEN) inferenceHistory.shift();
    if (rttHistory.length > MAX_HISTORY_LEN) rttHistory.shift();
    drawSparkline();

    if (!busy) {
      updateGlow("yolo");
    }
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
  ctx.lineWidth = 2.5;
  ctx.font = "bold 13px 'Outfit', sans-serif";

  result.boxes.forEach((box, i) => {
    const [x1, y1, x2, y2] = box;
    const px1 = x1 * scaleX;
    const py1 = y1 * scaleY;
    const pw = (x2 - x1) * scaleX;
    const ph = (y2 - y1) * scaleY;
    const label = `${result.labels[i].toUpperCase()} ${(result.scores[i] * 100).toFixed(0)}%`;
    
    const hue = (i * 75) % 360;
    const strokeColor = `hsla(${hue}, 85%, 60%, 1)`;
    const fillColor = `hsla(${hue}, 85%, 60%, 0.12)`;
    const badgeColor = `hsla(${hue}, 85%, 55%, 1)`;

    ctx.fillStyle = fillColor;
    ctx.fillRect(px1, py1, pw, ph);

    ctx.strokeStyle = strokeColor;
    ctx.shadowColor = "rgba(0, 0, 0, 0.3)";
    ctx.shadowBlur = 4;
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(px1, py1, pw, ph, 4);
    } else {
      ctx.rect(px1, py1, pw, ph);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    const textW = ctx.measureText(label).width + 10;
    ctx.fillStyle = badgeColor;
    
    let badgeY = py1 - 22;
    if (badgeY < 0) badgeY = py1 + 2;

    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(px1, badgeY, textW, 20, 3);
    } else {
      ctx.rect(px1, badgeY, textW, 20);
    }
    ctx.fill();

    ctx.fillStyle = "#000000";
    ctx.fillText(label, px1 + 5, badgeY + 14);
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
          lastSendTime = Date.now();
          ws.send(buf);
          inFlight = true;
        }
      });
    }, "image/jpeg", 0.6);
  }, SEND_INTERVAL_MS);
}

function drawSparkline() {
  if (!sparklineCanvas) return;
  const ctx = sparklineCanvas.getContext("2d");
  const w = sparklineCanvas.width;
  const h = sparklineCanvas.height;
  ctx.clearRect(0, 0, w, h);

  // Draw background grid lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  ctx.lineWidth = 1;
  for (let y = 10; y < h; y += 20) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  const maxVal = Math.max(
    ...inferenceHistory,
    ...rttHistory,
    80 // min scale range 80ms
  ) * 1.15;

  function drawLine(data, color, fillGrad) {
    if (data.length < 2) return;
    ctx.beginPath();
    const step = w / (MAX_HISTORY_LEN - 1);
    
    for (let i = 0; i < data.length; i++) {
      const x = i * step;
      const y = h - (data[i] / maxVal) * (h - 10) - 5;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Fill underneath
    ctx.lineTo((data.length - 1) * step, h);
    ctx.lineTo(0, h);
    ctx.fillStyle = fillGrad;
    ctx.fill();
  }

  // Create gradients
  const yoloGrad = ctx.createLinearGradient(0, 0, 0, h);
  yoloGrad.addColorStop(0, "rgba(57, 211, 83, 0.15)");
  yoloGrad.addColorStop(1, "rgba(57, 211, 83, 0)");

  const rttGrad = ctx.createLinearGradient(0, 0, 0, h);
  rttGrad.addColorStop(0, "rgba(88, 166, 255, 0.15)");
  rttGrad.addColorStop(1, "rgba(88, 166, 255, 0)");

  drawLine(rttHistory, "#58a6ff", rttGrad);
  drawLine(inferenceHistory, "#39d353", yoloGrad);
}

function addChatEntry(role, text, imageSrc = null) {
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

  if (imageSrc) {
    const img = document.createElement("img");
    img.src = imageSrc;
    img.className = "chat-img-thumb";
    img.title = "View snapshot";
    img.addEventListener("click", () => {
      const w = window.open();
      if (w) {
        w.document.write(`
          <html>
            <head><title>Camera Snapshot</title></head>
            <body style="margin:0; background:#0b0f14; display:flex; align-items:center; justify-content:center; height:100vh;">
              <img src="${imageSrc}" style="max-width:100%; max-height:100%; border-radius:8px; box-shadow:0 8px 30px rgba(0,0,0,0.5);" />
            </body>
          </html>
        `);
      }
    });
    li.appendChild(img);
  }

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
  let snapshotUrl = null;
  try {
    const ctx = capture.getContext("2d");
    ctx.drawImage(video, 0, 0, CAPTURE_W, CAPTURE_H);
    snapshotUrl = capture.toDataURL("image/jpeg");
  } catch (_) {}

  addChatEntry("user", question, snapshotUrl);
  setVoiceStatus("🧠 Thinking…", "thinking");
  updateGlow("thinking");
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
    updateGlow("speaking");
    await speak(caption);
  } catch (err) {
    addChatEntry("assistant", `(error: ${err.message})`);
  } finally {
    setVoiceStatus('👂 Listening for "Hey Assistant"…', "idle");
    updateGlow("yolo");
    resumeRecognition();
  }
}

function speak(text) {
  return new Promise((resolve) => {
    const utter = new SpeechSynthesisUtterance(text);
    if (selectedVoice) utter.voice = selectedVoice;
    utter.rate = ttsRate;
    utter.pitch = ttsPitch;
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

// ---- Telemetry & Stats fetching loop ----
function startStatsLoop() {
  setInterval(async () => {
    try {
      const res = await fetch("/api/system/stats");
      if (!res.ok) return;
      const stats = await res.json();
      if (statCpu) statCpu.textContent = `${stats.cpu.toFixed(1)}%`;
      if (statGpu) statGpu.textContent = `${stats.gpu.toFixed(1)}%`;
      if (statTemp) statTemp.textContent = `${stats.temp.toFixed(1)}°C`;
      if (statMem) statMem.textContent = `${stats.mem.toFixed(1)}%`;
    } catch (_) {}
  }, 2000);
}

// ---- Presets & Continuous narrative logic ----
function setupPresetsAndNarrator() {
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const q = btn.getAttribute("data-q");
      if (q && !busy) {
        askAboutScene(q);
      }
    });
  });

  // Continuous Narrator timer
  setInterval(() => {
    if (narrateToggle && narrateToggle.checked && !busy) {
      askAboutScene("Describe what you see.");
    }
  }, 10000);
}

// ---- Speech settings controls ----
function populateVoiceList() {
  if (typeof speechSynthesis === 'undefined') return;
  const voices = speechSynthesis.getVoices();
  if (voiceSelect) {
    const prevVal = voiceSelect.value;
    voiceSelect.innerHTML = '<option value="">Default Voice</option>';
    voices.forEach((v) => {
      const option = document.createElement("option");
      option.textContent = `${v.name} (${v.lang})`;
      option.value = v.voiceURI;
      voiceSelect.appendChild(option);
    });
    if (prevVal) voiceSelect.value = prevVal;
  }
}

function setupSpeechControls() {
  populateVoiceList();
  if (speechSynthesis.onvoiceschanged !== undefined) {
    speechSynthesis.onvoiceschanged = populateVoiceList;
  }

  if (voiceSelect) {
    voiceSelect.addEventListener("change", () => {
      const voices = speechSynthesis.getVoices();
      selectedVoice = voices.find(v => v.voiceURI === voiceSelect.value) || null;
    });
  }
  if (rateRange) {
    rateRange.addEventListener("input", () => {
      ttsRate = parseFloat(rateRange.value);
      if (rateVal) rateVal.textContent = ttsRate.toFixed(1);
    });
  }
  if (pitchRange) {
    pitchRange.addEventListener("input", () => {
      ttsPitch = parseFloat(pitchRange.value);
      if (pitchVal) pitchVal.textContent = ttsPitch.toFixed(1);
    });
  }
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
  
  // Set up new telemetry stats, VLM presets, narrator loops, and TTS controls
  startStatsLoop();
  setupPresetsAndNarrator();
  setupSpeechControls();
})();
