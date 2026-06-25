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

// MediaPipe Toggles
const handsToggle = document.getElementById("handsToggle");
const poseToggle = document.getElementById("poseToggle");

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

// Local MediaPipe state
let mpHands = null;
let mpPose = null;
let showHands = false;
let showPose = false;
let isProcessingHands = false;
let isProcessingPose = false;
let latestHandResults = null;
let latestPoseResults = null;
let latestYoloResults = null;

function showKickOverlay() {
  document.getElementById("kickOverlay").classList.remove("hidden");
  if (recognition) { try { recognition.stop(); } catch (_) {} }
  speechSynthesis.cancel();
  setVoiceStatus("Session Ended", "idle");
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
    connStatus.textContent = "Jetson Connected";
    const statusDot = document.querySelector(".status-indicator");
    if (statusDot) statusDot.className = "status-indicator online";
    sendMode();
  };
  ws.onclose = (ev) => {
    console.log("WebSocket closed. Code:", ev.code, "Reason:", ev.reason);
    if (ev.code === 4001) { showKickOverlay(); return; }
    connStatus.textContent = "Offline — Retrying…";
    const statusDot = document.querySelector(".status-indicator");
    if (statusDot) statusDot.className = "status-indicator";
    setTimeout(connectWS, 1500);
  };
  ws.onerror = (err) => {
    console.error("WebSocket error:", err);
    connStatus.textContent = "Connection Error";
  };

  ws.onmessage = (ev) => {
    inFlight = false;
    const clientRtt = Date.now() - lastSendTime;
    const msg = JSON.parse(ev.data);
    if (msg.type === "admin_disconnect") { showKickOverlay(); return; }
    
    // Store latest results for the unified draw loop
    latestYoloResults = msg;

    inferenceMsEl.textContent = `${msg.inference_ms.toFixed(1)} ms`;
    fpsEl.textContent = msg.fps.toFixed(1);
    if (modeBadge) modeBadge.textContent = msg.tensorrt ? "TensorRT Engine" : "PyTorch Fallback";

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
if (trtToggle) trtToggle.addEventListener("change", sendMode);

function getLabelHue(label) {
  let hash = 0;
  for (let i = 0; i < label.length; i++) {
    hash = label.charCodeAt(i) + ((hash << 5) - hash);
  }
  return Math.abs(hash) % 360;
}

// Unified frame render loop incorporating local MediaPipe drawing
function drawScene() {
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  // 1. Draw YOLO boxes
  if (latestYoloResults) {
    const scaleX = overlay.width / latestYoloResults.frame_w;
    const scaleY = overlay.height / latestYoloResults.frame_h;
    ctx.lineWidth = 2;
    ctx.font = "bold 11px 'Outfit', sans-serif";

    latestYoloResults.boxes.forEach((box, i) => {
      const [x1, y1, x2, y2] = box;
      const px1 = x1 * scaleX;
      const py1 = y1 * scaleY;
      const pw = (x2 - x1) * scaleX;
      const ph = (y2 - y1) * scaleY;
      const labelText = latestYoloResults.labels[i];
      const score = latestYoloResults.scores[i];
      const displayLabel = `${labelText.toUpperCase()} ${(score * 100).toFixed(0)}%`;

      const hue = getLabelHue(labelText);
      const strokeColor = `hsla(${hue}, 70%, 55%, 1)`;
      const fillColor = `hsla(${hue}, 70%, 55%, 0.08)`;
      const badgeColor = `hsla(${hue}, 70%, 50%, 1)`;

      ctx.fillStyle = fillColor;
      ctx.fillRect(px1, py1, pw, ph);

      ctx.strokeStyle = strokeColor;
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(px1, py1, pw, ph, 4);
      } else {
        ctx.rect(px1, py1, pw, ph);
      }
      ctx.stroke();

      const textW = ctx.measureText(displayLabel).width + 10;
      ctx.fillStyle = badgeColor;
      
      let badgeY = py1 - 18;
      if (badgeY < 0) badgeY = py1 + 2;

      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(px1, badgeY, textW, 16, 2);
      } else {
        ctx.rect(px1, badgeY, textW, 16);
      }
      ctx.fill();

      ctx.fillStyle = "#ffffff";
      ctx.fillText(displayLabel, px1 + 5, badgeY + 12);
    });
  }

  // 2. Draw MediaPipe Hands
  if (showHands && latestHandResults && latestHandResults.multiHandLandmarks) {
    latestHandResults.multiHandLandmarks.forEach((landmarks) => {
      ctx.strokeStyle = "rgba(99, 102, 241, 0.7)";
      ctx.lineWidth = 2.5;

      const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [5, 9], [9, 10], [10, 11], [11, 12],
        [9, 13], [13, 14], [14, 15], [15, 16],
        [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]
      ];

      connections.forEach(([p1, p2]) => {
        const pt1 = landmarks[p1];
        const pt2 = landmarks[p2];
        ctx.beginPath();
        ctx.moveTo(pt1.x * overlay.width, pt1.y * overlay.height);
        ctx.lineTo(pt2.x * overlay.width, pt2.y * overlay.height);
        ctx.stroke();
      });

      landmarks.forEach((pt) => {
        ctx.beginPath();
        ctx.arc(pt.x * overlay.width, pt.y * overlay.height, 3.5, 0, 2 * Math.PI);
        ctx.fillStyle = "#ffffff";
        ctx.fill();
        ctx.strokeStyle = "#4f46e5";
        ctx.lineWidth = 1;
        ctx.stroke();
      });
    });
  }

  // 3. Draw MediaPipe Pose
  if (showPose && latestPoseResults && latestPoseResults.poseLandmarks) {
    const landmarks = latestPoseResults.poseLandmarks;
    const poseConnections = [
      [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
      [11, 23], [12, 24], [23, 24],
      [23, 25], [24, 26], [25, 27], [26, 28]
    ];

    ctx.strokeStyle = "rgba(168, 85, 247, 0.7)";
    ctx.lineWidth = 2.5;

    poseConnections.forEach(([p1, p2]) => {
      const pt1 = landmarks[p1];
      const pt2 = landmarks[p2];
      if (pt1.visibility > 0.5 && pt2.visibility > 0.5) {
        ctx.beginPath();
        ctx.moveTo(pt1.x * overlay.width, pt1.y * overlay.height);
        ctx.lineTo(pt2.x * overlay.width, pt2.y * overlay.height);
        ctx.stroke();
      }
    });

    landmarks.forEach((pt) => {
      if (pt.visibility > 0.5) {
        ctx.beginPath();
        ctx.arc(pt.x * overlay.width, pt.y * overlay.height, 3, 0, 2 * Math.PI);
        ctx.fillStyle = "#ffffff";
        ctx.fill();
        ctx.strokeStyle = "#9333ea";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    });
  }

  requestAnimationFrame(drawScene);
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
  ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
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
    ctx.lineWidth = 2.0;
    ctx.stroke();

    // Fill underneath
    ctx.lineTo((data.length - 1) * step, h);
    ctx.lineTo(0, h);
    ctx.fillStyle = fillGrad;
    ctx.fill();
  }

  // Create gradients
  const yoloGrad = ctx.createLinearGradient(0, 0, 0, h);
  yoloGrad.addColorStop(0, "rgba(16, 185, 129, 0.12)");
  yoloGrad.addColorStop(1, "rgba(16, 185, 129, 0)");

  const rttGrad = ctx.createLinearGradient(0, 0, 0, h);
  rttGrad.addColorStop(0, "rgba(99, 102, 241, 0.12)");
  rttGrad.addColorStop(1, "rgba(99, 102, 241, 0)");

  drawLine(rttHistory, "#6366f1", rttGrad);
  drawLine(inferenceHistory, "#10b981", yoloGrad);
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
            <body style="margin:0; background:#09090b; display:flex; align-items:center; justify-content:center; height:100vh;">
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

function setVoiceStatus(text, statusType) {
  if (voiceStatus) {
    voiceStatus.textContent = text;
    voiceStatus.className = `voice-status status-${statusType}`;
  }
}

// ---- Debug: live transcript of what the browser's speech recognizer hears ----
function applyDebugVisibility() {
  if (debugPanel) debugPanel.style.display = debugToggle.checked ? "block" : "none";
}
if (debugToggle) debugToggle.addEventListener("change", applyDebugVisibility);
applyDebugVisibility();

function setInterimTranscript(text) {
  if (interimTranscriptEl) interimTranscriptEl.textContent = text || "...";
}

function logFinalTranscript(text, heardWake) {
  setInterimTranscript("");
  const li = document.createElement("li");
  li.className = heardWake ? "heard-wake" : "";
  const time = document.createElement("time");
  time.textContent = new Date().toLocaleTimeString();
  li.appendChild(time);
  li.appendChild(document.createTextNode(` "${text}"${heardWake ? " — wake phrase detected" : ""}`));
  if (transcriptLog) transcriptLog.prepend(li);
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
    setVoiceStatus("Listening...", "idle");
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
if (fallbackForm) {
  fallbackForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = fallbackInput.value.trim();
    if (!q) return;
    fallbackInput.value = "";
    askAboutScene(q);
  });
}

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
    setVoiceStatus("Speech Error", "idle");
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
    setVoiceStatus("Listening...", "idle");
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
  if (typeof speechSynthesis === 'undefined') return;
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

// ---- Local MediaPipe vision tasks loop ----
function setupMediaPipe() {
  if (typeof Hands !== "undefined") {
    mpHands = new Hands({locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }});
    mpHands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });
    mpHands.onResults((results) => {
      latestHandResults = results;
    });
  }

  if (typeof Pose !== "undefined") {
    mpPose = new Pose({locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
    }});
    mpPose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });
    mpPose.onResults((results) => {
      latestPoseResults = results;
    });
  }

  // Hook up MediaPipe local toggles
  if (handsToggle) {
    handsToggle.addEventListener("change", () => {
      showHands = handsToggle.checked;
      if (!showHands) latestHandResults = null;
    });
  }

  if (poseToggle) {
    poseToggle.addEventListener("change", () => {
      showPose = poseToggle.checked;
      if (!showPose) latestPoseResults = null;
    });
  }
}

async function processLocalVision() {
  if (video.readyState >= 2) {
    if (showHands && mpHands && !isProcessingHands) {
      isProcessingHands = true;
      try {
        await mpHands.send({image: video});
      } catch (err) {
        console.error("Hands error", err);
      }
      isProcessingHands = false;
    } else if (!showHands) {
      latestHandResults = null;
    }

    if (showPose && mpPose && !isProcessingPose) {
      isProcessingPose = true;
      try {
        await mpPose.send({image: video});
      } catch (err) {
        console.error("Pose error", err);
      }
      isProcessingPose = false;
    } else if (!showPose) {
      latestPoseResults = null;
    }
  }
  setTimeout(processLocalVision, 33);
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
  
  startStatsLoop();
  setupPresetsAndNarrator();
  setupSpeechControls();
  
  // Set up and start MediaPipe local perception tasks
  setupMediaPipe();
  processLocalVision();
  
  // Start the unified 2D canvas drawing scene loop
  drawScene();
})();
