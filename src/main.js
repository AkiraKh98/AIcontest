// =================================
// SCENE NARRATOR
// =================================
import {
  FilesetResolver,
  PoseLandmarker,
  FaceLandmarker,
  DrawingUtils
} from '@mediapipe/tasks-vision';

console.log('🎬 Scene Narrator loaded');

// =================================
// STAGE 2 — MediaPipe (pose + face)
// =================================
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const overlayCtx = overlay.getContext('2d');
const placeholder = document.querySelector('.camera-placeholder');

let poseLandmarker = null;
let faceLandmarker = null;
let videoRunning = false;
let lastVideoTime = -1;

window.latestPose = null;
window.latestFace = null;
window.latestFeatures = null;

async function initMediaPipe() {
  placeholder.textContent = '⏳ Loading AI models…';

  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/wasm'
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
      delegate: 'GPU'
    },
    runningMode: 'VIDEO',
    numPoses: 2
  });

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      delegate: 'GPU'
    },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: false
  });

  placeholder.textContent = '📷 Starting camera…';
  await startCamera();
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    });
    video.srcObject = stream;

    video.addEventListener('loadedmetadata', () => {
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
      placeholder.style.display = 'none';
      videoRunning = true;
      predictLoop();
    });
  } catch (err) {
    console.error('Camera error:', err);
    placeholder.textContent = '❌ Camera blocked. Allow access and refresh.';
  }
}

function predictLoop() {
  if (!videoRunning) return;

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const now = performance.now();

    try {
      window.latestPose = poseLandmarker.detectForVideo(video, now);
      window.latestFace = faceLandmarker.detectForVideo(video, now);
      drawResults();
    } catch (err) {
      console.warn('Detection error:', err);
    }
  }

  requestAnimationFrame(predictLoop);
}

function drawResults() {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  const draw = new DrawingUtils(overlayCtx);

  if (window.latestPose?.landmarks) {
    for (const lm of window.latestPose.landmarks) {
      draw.drawLandmarks(lm, { color: '#5fb3a1', radius: 3 });
      draw.drawConnectors(lm, PoseLandmarker.POSE_CONNECTIONS, {
        color: '#5fb3a1',
        lineWidth: 2
      });
    }
  }

  if (window.latestFace?.faceLandmarks) {
    for (const lm of window.latestFace.faceLandmarks) {
      draw.drawLandmarks(lm, { color: '#d4a574', radius: 0.8 });
    }
  }
}

initMediaPipe().catch((err) => {
  console.error('MediaPipe init failed:', err);
  placeholder.textContent = '❌ Model load failed. Check console.';
});

// =================================
// STAGE 2.5 — Feature Extractor
// (raw landmarks → human-readable words)
// =================================

// Posture: ratio of hip→knee distance to shoulder→hip distance.
// Small ratio = knees close to hips vertically = sitting.
function getPosture(lm) {
  const ls = lm[11], rs = lm[12], lh = lm[23], lk = lm[25];
  if (lh.visibility < 0.5 || lk.visibility < 0.5) return 'partial view';
  const hipKnee = Math.abs(lk.y - lh.y);
  const shoulderHip = Math.abs(lh.y - ls.y);
  if (shoulderHip < 0.01) return 'unknown';
  const ratio = hipKnee / shoulderHip;
  if (ratio < 0.7) return 'sitting';
  return 'standing';
}

// Hand height: average wrist Y vs shoulder/hip Y (image-y grows downward).
function getHandsPosition(lm) {
  const ls = lm[11], rs = lm[12], lw = lm[15], rw = lm[16], lh = lm[23], rh = lm[24];
  if (lw.visibility < 0.3 && rw.visibility < 0.3) return 'out of frame';
  const shoulderY = (ls.y + rs.y) / 2;
  const hipY = (lh.y + rh.y) / 2;
  const wristY = (lw.y + rw.y) / 2;
  if (wristY < shoulderY) return 'raised above shoulders';
  if (wristY < hipY) return 'at chest level';
  return 'down at sides';
}

// Expression: mouth corner lift relative to mouth center,
// normalized by mouth height so close/far faces both work.
function getExpression(flm) {
  const lc = flm[61], rc = flm[291], up = flm[13], lo = flm[14];
  const cornerY = (lc.y + rc.y) / 2;
  const centerY = (up.y + lo.y) / 2;
  const mouthH = Math.abs(lo.y - up.y);
  if (mouthH < 0.001) return 'neutral';
  const lift = (centerY - cornerY) / mouthH;
  if (lift > 0.35) return 'smiling';
  if (lift < -0.35) return 'frowning';
  return 'neutral';
}

// Head roll: eye y-difference normalized by eye distance.
function getHeadTilt(flm) {
  const le = flm[33], re = flm[263];
  const dx = re.x - le.x;
  const dy = re.y - le.y;
  const eyeDist = Math.hypot(dx, dy);
  if (eyeDist < 0.001) return 'unknown';
  const ratio = dy / eyeDist;
  if (ratio > 0.1) return 'tilted right';
  if (ratio < -0.1) return 'tilted left';
  return 'straight';
}

function extractFeatures() {
  const f = {
    people_count: 0,
    posture: '—',
    expression: '—',
    hands: '—',
    head_tilt: '—'
  };
  const pose = window.latestPose;
  const face = window.latestFace;

  if (pose?.landmarks?.length) {
    f.people_count = pose.landmarks.length;
    const lm = pose.landmarks[0];
    f.posture = getPosture(lm);
    f.hands = getHandsPosition(lm);
  }
  if (face?.faceLandmarks?.length) {
    const flm = face.faceLandmarks[0];
    f.expression = getExpression(flm);
    f.head_tilt = getHeadTilt(flm);
  }
  return f;
}

function updateFeaturesPanel() {
  const f = extractFeatures();
  window.latestFeatures = f;
  const list = document.getElementById('features-list');

  if (f.people_count === 0) {
    list.innerHTML = '<li style="color: var(--muted)">No person detected</li>';
    return;
  }

  list.innerHTML = `
    <li>${f.people_count} ${f.people_count === 1 ? 'person' : 'people'} detected</li>
    <li>Posture: <b>${f.posture}</b></li>
    <li>Expression: <b>${f.expression}</b></li>
    <li>Hands: <b>${f.hands}</b></li>
    <li>Head: <b>${f.head_tilt}</b></li>
  `;
}

setInterval(updateFeaturesPanel, 500);

// =================================
// STAGE 3 — Voice → Text
// =================================
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const micBtn = document.getElementById('mic-btn');
const transcriptEl = document.getElementById('transcript');

if (!SpeechRecognition) {
  transcriptEl.textContent = '⚠️ Speech recognition not supported. Use Chrome or Edge.';
  micBtn.disabled = true;
} else {
  const recognition = new SpeechRecognition();
  recognition.lang = 'en-US';
  recognition.continuous = false;
  recognition.interimResults = true;

  let isListening = false;

  recognition.onstart = () => {
    isListening = true;
    micBtn.textContent = '🔴 Listening… (click to stop)';
    micBtn.classList.add('btn-listening');
    transcriptEl.textContent = '...';
  };
  recognition.onresult = (event) => {
    let text = '';
    for (let i = 0; i < event.results.length; i++) text += event.results[i][0].transcript;
    transcriptEl.textContent = `"${text.trim()}"`;
  };
  recognition.onerror = (event) => {
    console.error('Speech error:', event.error);
    transcriptEl.textContent = `⚠️ ${event.error}. Try again.`;
  };
  recognition.onend = () => {
    isListening = false;
    micBtn.textContent = '🎤 Start Listening';
    micBtn.classList.remove('btn-listening');
  };

  micBtn.addEventListener('click', () => {
    if (isListening) recognition.stop();
    else recognition.start();
  });
}

function getTranscript() {
  return transcriptEl.textContent.replace(/^["']|["']$/g, '').trim();
}
window.getTranscript = getTranscript;

// =================================
// STAGE 4 — Gemini (Step 6)
// =================================
document.getElementById('generate-btn').addEventListener('click', () => {
  alert('Gemini stage — coming in Step 6');
});

document.getElementById('capture-btn').addEventListener('click', () => {
  alert('Capture feature — coming in Step 8');
});