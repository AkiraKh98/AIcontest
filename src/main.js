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

// 👇 PASTE YOUR TEACHABLE MACHINE MODEL URL HERE (keep the trailing slash)
const TM_MODEL_URL = 'https://teachablemachine.withgoogle.com/models/vhBhcXpCZ/';

const GEMINI_KEY = import.meta.env.VITE_GEMINI_KEY;
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key=${GEMINI_KEY}`;

// =================================
// STAGE 1 — Teachable Machine
// =================================
let tmModel = null;
window.latestScene = null;

async function initTeachableMachine() {
  try {
    tmModel = await window.tmImage.load(
      TM_MODEL_URL + 'model.json',
      TM_MODEL_URL + 'metadata.json'
    );
    console.log('✅ Teachable Machine model loaded');
  } catch (err) {
    console.error('TM load failed:', err);
    document.getElementById('scene-label').textContent = 'Model not loaded';
    document.getElementById('scene-conf').textContent = 'Check TM_MODEL_URL';
  }
}

async function predictScene() {
  if (!tmModel || !videoRunning) return;
  try {
    const preds = await tmModel.predict(video);
    preds.sort((a, b) => b.probability - a.probability);
    const top = preds[0];

    window.latestScene = {
      scene: top.className,
      confidence: top.probability
    };

    document.getElementById('scene-label').textContent = top.className;
    const pct = Math.round(top.probability * 100);
    document.getElementById('scene-bar').style.width = pct + '%';
    document.getElementById('scene-conf').textContent = pct + '% confident';
  } catch (err) {
    console.warn('TM predict error:', err);
  }
}

setInterval(predictScene, 1000);
initTeachableMachine();

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
// =================================
function getPosture(lm) {
  const ls = lm[11], lh = lm[23], lk = lm[25];
  if (lh.visibility < 0.5 || lk.visibility < 0.5) return 'partial view';
  const hipKnee = Math.abs(lk.y - lh.y);
  const shoulderHip = Math.abs(lh.y - ls.y);
  if (shoulderHip < 0.01) return 'unknown';
  return (hipKnee / shoulderHip) < 0.7 ? 'sitting' : 'standing';
}
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
function getHeadTilt(flm) {
  const le = flm[33], re = flm[263];
  const dx = re.x - le.x, dy = re.y - le.y;
  const eyeDist = Math.hypot(dx, dy);
  if (eyeDist < 0.001) return 'unknown';
  const ratio = dy / eyeDist;
  if (ratio > 0.1) return 'tilted right';
  if (ratio < -0.1) return 'tilted left';
  return 'straight';
}

function extractFeatures() {
  const f = { people_count: 0, posture: '—', expression: '—', hands: '—', head_tilt: '—' };
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
// STAGE 4 — Gemini
// =================================
window.latestNarration = null;

async function generateNarration() {
  const narrationEl = document.getElementById('narration-text');
  const btn = document.getElementById('generate-btn');

  const scene = window.latestScene;
  const features = window.latestFeatures;
  const transcript = window.getTranscript();

  if (!scene && (!features || features.people_count === 0)) {
    narrationEl.textContent = '⚠️ No perceptions yet. Wait a moment for the models to detect something.';
    return;
  }

  const sceneStr = scene
    ? `${scene.scene} (${Math.round(scene.confidence * 100)}% confidence)`
    : 'unknown';

  const featureStr = features?.people_count > 0
    ? `${features.people_count} ${features.people_count === 1 ? 'person' : 'people'}, ${features.posture}, ${features.expression}, hands ${features.hands}, head ${features.head_tilt}`
    : 'no person detected';

  const userRequest = transcript && transcript !== '...' && !transcript.startsWith('⚠️')
    ? transcript
    : 'describe the scene in a vivid, cinematic way';

  const prompt = `You are a cinematic scene narrator.

VISUAL CLASSIFICATION: ${sceneStr}
BODY LANGUAGE: ${featureStr}
USER REQUEST: "${userRequest}"

Write 1–2 vivid sentences honoring the user's request in tone and style. Be specific and evocative. Do not list the inputs back; weave them into prose.`;

  btn.disabled = true;
  btn.textContent = '⏳ Thinking…';
  narrationEl.textContent = '…';

  try {
    const res = await fetch(GEMINI_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }]
      })
    });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`HTTP ${res.status} — ${errText.slice(0, 200)}`);
    }
    const data = await res.json();
    const text = data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
    if (!text) throw new Error('Empty response from Gemini');

    narrationEl.textContent = text;
    narrationEl.textContent = text;
    speakNarration(text);   // 👈 add this

    window.latestNarration = {
      scene: sceneStr,
      features: featureStr,
      transcript: userRequest,
      narration: text,
      timestamp: new Date().toISOString()
    };
  } catch (err) {
    console.error('Gemini error:', err);
    narrationEl.textContent = `⚠️ ${err.message}`;
  } finally {
    btn.disabled = false;
    btn.textContent = '✨ Generate Narration';
  }
}
document.getElementById('generate-btn').addEventListener('click', generateNarration);

// =================================
// STAGE 5 — Capture & History
// =================================
const history = [];

function captureSnapshot() {
  if (!window.latestNarration) {
    alert('Generate a narration first!');
    return;
  }

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);
  ctx.drawImage(overlay, 0, 0);
  const thumbnail = canvas.toDataURL('image/jpeg', 0.7);

  history.unshift({
    id: Date.now(),
    thumbnail,
    ...window.latestNarration
  });
  renderHistory();
}

function renderHistory() {
  const list = document.getElementById('history-list');
  if (history.length === 0) {
    list.innerHTML = '<p class="empty">Your captured scenes will appear here.</p>';
    return;
  }

  list.innerHTML = history
    .map(
      (h) => `
    <article class="history-card">
      <img src="${h.thumbnail}" alt="captured frame" />
      <div class="history-data">
        <div class="history-stage">
          <span class="stage-label">👁 Scene</span>
          <span class="stage-value">${h.scene}</span>
        </div>
        <div class="history-stage">
          <span class="stage-label">🦴 Body Language</span>
          <span class="stage-value">${h.features}</span>
        </div>
        <div class="history-stage">
          <span class="stage-label">🎙 Request</span>
          <span class="stage-value">"${h.transcript}"</span>
        </div>
        <div class="history-stage">
          <span class="stage-label">📖 Narration</span>
          <span class="stage-value narration">${h.narration}</span>
        </div>
        <button class="btn-secondary remove-btn" data-id="${h.id}">Remove</button>
      </div>
    </article>
  `
    )
    .join('');

  document.querySelectorAll('.remove-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      const id = Number(e.target.dataset.id);
      const idx = history.findIndex((h) => h.id === id);
      if (idx > -1) history.splice(idx, 1);
      renderHistory();
    });
  });
}

document.getElementById('capture-btn').addEventListener('click', captureSnapshot);


// =================================
// HUD — live clock
// =================================
function tickClock() {
  const now = new Date();
  const t = now.toTimeString().slice(0, 8);
  const elClock = document.getElementById('clock');
  const elCam = document.getElementById('cam-time');
  if (elClock) elClock.textContent = t;
  if (elCam) elCam.textContent = t;
}
setInterval(tickClock, 1000);
tickClock();

// =================================
// STAGE 5 — Embodiment (3D avatar + voice)
// =================================
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const AVATAR_URL = '/avatar.glb';

const avatarContainer = document.getElementById('avatar-container');
const avatarStatus = document.getElementById('avatar-status');

const avScene = new THREE.Scene();

// Tighter FOV for a flatter, more cinematic portrait look
const avCamera = new THREE.PerspectiveCamera(22, 1, 0.05, 100);

// Default framing — refined once the avatar loads (see frameOnHead)
const cameraTarget = new THREE.Vector3(0, 1.62, 0);
avCamera.position.set(0, 1.62, 1.4);
avCamera.lookAt(cameraTarget);

const avRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
avRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
avRenderer.outputColorSpace = THREE.SRGBColorSpace;
avRenderer.toneMapping = THREE.ACESFilmicToneMapping;
avatarContainer.appendChild(avRenderer.domElement);

function resizeAvatar() {
  const w = avatarContainer.clientWidth;
  const h = avatarContainer.clientHeight;
  if (w === 0 || h === 0) return;
  avRenderer.setSize(w, h, false);
  avCamera.aspect = w / h;
  avCamera.updateProjectionMatrix();
}
resizeAvatar();
window.addEventListener('resize', resizeAvatar);
// Container can resize independently of the window (panel reflow, font load,
// devtools open). ResizeObserver keeps the canvas aligned in those cases.
if (typeof ResizeObserver !== 'undefined') {
  new ResizeObserver(resizeAvatar).observe(avatarContainer);
}

// Themed lighting — phosphor key + amber rim
avScene.add(new THREE.AmbientLight(0xffffff, 0.35));
const keyLight = new THREE.DirectionalLight(0x5ff898, 1.4);
keyLight.position.set(1.2, 2.5, 2.2);
avScene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xffb86c, 0.45);
fillLight.position.set(-2, 1.5, 1);
avScene.add(fillLight);
const rimLight = new THREE.DirectionalLight(0xffffff, 0.4);
rimLight.position.set(0, 1.8, -2);
avScene.add(rimLight);

let avatar = null;
let headBone = null;
// Track every mesh that exposes a mouth blendshape — RPM/Avaturn split the
// face across Wolf3D_Head / Wolf3D_Teeth / Wolf3D_Tongue, and animating just
// one leaves the jaw partly frozen.
const jawTargets = [];

// Match bones across naming conventions: "LeftArm", "mixamorig:LeftArm",
// "leftUpperArm" (VRM-ish), "Left_Arm", etc.
function matchBone(name, side, part) {
  const n = name.toLowerCase().replace(/[_:.\s-]/g, '');
  const s = side.toLowerCase();
  if (part === 'arm') {
    if (n.includes('forearm') || n.includes('lowerarm') || n.includes('shoulder')) return false;
    return n.includes(s + 'arm') || n.includes(s + 'upperarm');
  }
  if (part === 'forearm') {
    return n.includes(s + 'forearm') || n.includes(s + 'lowerarm');
  }
  if (part === 'hand') {
    return n.endsWith(s + 'hand');
  }
  if (part === 'head') {
    return n.endsWith('head') && !n.includes('neck');
  }
  return false;
}

const MOUTH_KEYS = ['jawOpen', 'mouthOpen', 'viseme_aa', 'viseme_O', 'viseme_E', 'mouth_open', 'A25_JawOpen'];

const gltfLoader = new GLTFLoader();
gltfLoader.load(
  AVATAR_URL,
  (gltf) => {
    avatar = gltf.scene;
    avatar.position.set(0, 0, 0);
    avScene.add(avatar);

    // Quaternion path (not Euler) — glTF bones can carry rotation orders
    // that make .rotation.set(...) a silent no-op after import.
    const Z_AXIS = new THREE.Vector3(0, 0, 1);
    const Y_AXIS = new THREE.Vector3(0, 1, 0);
    function poseArm(bone, sign) {
      bone.quaternion
        .setFromAxisAngle(Z_AXIS, 1.45 * sign)
        .multiply(new THREE.Quaternion().setFromAxisAngle(Y_AXIS, -0.1 * sign));
      bone.updateMatrix();
    }
    function poseAxis(bone, axis, angle) {
      bone.setRotationFromAxisAngle(axis, angle);
      bone.updateMatrix();
    }

    // Some glTF exporters expose joints only through skeleton.bones, not as
    // isBone scene nodes — walk skeletons explicitly and dedupe.
    const visited = new Set();
    avatar.traverse((child) => {
      if (child.isSkinnedMesh && child.skeleton) {
        for (const bone of child.skeleton.bones) {
          if (visited.has(bone)) continue;
          visited.add(bone);
          const name = bone.name || '';
          if (matchBone(name, 'Left', 'arm')) poseArm(bone, 1);
          else if (matchBone(name, 'Right', 'arm')) poseArm(bone, -1);
          else if (matchBone(name, 'Left', 'forearm')) poseAxis(bone, Y_AXIS, -0.15);
          else if (matchBone(name, 'Right', 'forearm')) poseAxis(bone, Y_AXIS, 0.15);
          else if (matchBone(name, 'Left', 'hand')) poseAxis(bone, Z_AXIS, 0.1);
          else if (matchBone(name, 'Right', 'hand')) poseAxis(bone, Z_AXIS, -0.1);
          else if (matchBone(name, '', 'head')) headBone = bone;
        }
      }

      if (child.isMesh && child.morphTargetDictionary) {
        const dict = child.morphTargetDictionary;
        let idx = null;
        let key = null;
        for (const k of MOUTH_KEYS) {
          if (k in dict) { idx = dict[k]; key = k; break; }
        }
        if (idx === null) {
          for (const k of Object.keys(dict)) {
            if (/jawopen|mouthopen/i.test(k)) { idx = dict[k]; key = k; break; }
          }
        }
        if (idx !== null) {
          if (!child.morphTargetInfluences) {
            child.morphTargetInfluences = new Array(Object.keys(dict).length).fill(0);
          }
          jawTargets.push({ mesh: child, index: idx, key });
        }
      }
    });

    frameOnHead();

    if (jawTargets.length > 0) {
      avatarStatus.textContent = '● READY';
    } else {
      console.warn('⚠️ Avatar loaded but no jawOpen morph — mouth won\'t animate');
      avatarStatus.textContent = '● READY (no lipsync)';
    }
  },
  undefined,
  (err) => {
    console.error('Avatar load failed:', err);
    avatarContainer.innerHTML =
      '<div class="avatar-error">// AVATAR LOAD FAILED<br>CHECK AVATAR_URL IN main.js</div>';
    avatarStatus.textContent = '● ERROR';
    avatarStatus.style.color = 'var(--crimson)';
  }
);

// Tight head-only framing: hides the body below the neck so the
// T-pose arms stay out of view.
function frameOnHead() {
  if (!avatar) return;
  const headPos = new THREE.Vector3();
  if (headBone) {
    headBone.updateWorldMatrix(true, false);
    headPos.setFromMatrixPosition(headBone.matrixWorld);
  } else {
    const box = new THREE.Box3().setFromObject(avatar);
    headPos.set((box.min.x + box.max.x) / 2, box.max.y - 0.12, (box.min.z + box.max.z) / 2);
  }
  cameraTarget.set(headPos.x, headPos.y + 0.04, headPos.z);
  avCamera.position.set(headPos.x, headPos.y + 0.04, headPos.z + 0.7);
  avCamera.lookAt(cameraTarget);
}

let isSpeaking = false;
let jawPhase = 0;
let jawSettled = true;
const avClock = new THREE.Clock();

function animateAvatar() {
  requestAnimationFrame(animateAvatar);
  const elapsed = avClock.getElapsedTime();

  if (avatar) {
    avatar.rotation.y = Math.sin(elapsed * 0.35) * 0.07;
    avatar.position.y = Math.sin(elapsed * 0.6) * 0.004;
  }

  if (jawTargets.length > 0 && (isSpeaking || !jawSettled)) {
    let target;
    if (isSpeaking) {
      jawPhase += 0.32;
      target =
        (Math.sin(jawPhase) * 0.5 + 0.5) * 0.42 +
        (Math.sin(jawPhase * 2.7) * 0.5 + 0.5) * 0.12 +
        Math.random() * 0.05;
      jawSettled = false;
    } else {
      target = 0;
    }
    let maxInfl = 0;
    for (const t of jawTargets) {
      const infl = t.mesh.morphTargetInfluences;
      if (!infl) continue;
      const cur = infl[t.index] || 0;
      const next = cur + (target - cur) * (isSpeaking ? 0.55 : 0.18);
      infl[t.index] = next;
      if (next > maxInfl) maxInfl = next;
    }
    if (!isSpeaking && maxInfl < 1e-4) {
      for (const t of jawTargets) {
        if (t.mesh.morphTargetInfluences) t.mesh.morphTargetInfluences[t.index] = 0;
      }
      jawSettled = true;
    }
  }

  avRenderer.render(avScene, avCamera);
}
animateAvatar();

// Speech synthesis
function speakNarration(text) {
  if (!('speechSynthesis' in window)) {
    console.warn('Speech synthesis unavailable');
    return;
  }

  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 0.9;
  u.pitch = 0.9;
  u.volume = 1.0;

  const voices = speechSynthesis.getVoices();
  const preferred =
    voices.find((v) => /Microsoft.*Guy.*Online.*Natural/i.test(v.name)) ||
    voices.find((v) => /Microsoft.*Davis.*Online.*Natural/i.test(v.name)) ||
    voices.find((v) => /Microsoft.*Brian.*Online.*Natural/i.test(v.name)) ||
    voices.find((v) => /Microsoft.*Guy/i.test(v.name)) ||
    voices.find((v) => /Microsoft.*David/i.test(v.name)) ||
    voices.find((v) => /Microsoft.*Mark/i.test(v.name)) ||
    voices.find((v) => /Google UK English Male/i.test(v.name)) ||
    voices.find((v) => /Daniel/i.test(v.name)) ||
    voices.find((v) => /Alex/i.test(v.name)) ||
    voices.find((v) => v.lang.startsWith('en'));
  if (preferred) u.voice = preferred;

  u.onstart = () => {
    isSpeaking = true;
    avatarStatus.textContent = '● SPEAKING';
    avatarStatus.style.color = 'var(--crimson)';
  };
  u.onend = () => {
    isSpeaking = false;
    avatarStatus.textContent = '● READY';
    avatarStatus.style.color = '';
  };
  u.onerror = () => {
    isSpeaking = false;
    avatarStatus.textContent = '● ERROR';
  };

  speechSynthesis.speak(u);
}
window.speakNarration = speakNarration;

// Wire the manual buttons
document.getElementById('speak-btn').addEventListener('click', () => {
  if (isSpeaking) return;
  if (window.latestNarration?.narration) {
    speakNarration(window.latestNarration.narration);
  } else {
    alert('Generate a narration first');
  }
});

document.getElementById('stop-btn').addEventListener('click', () => {
  speechSynthesis.cancel();
  isSpeaking = false;
});

// Some browsers populate voices async; warm them up
speechSynthesis.onvoiceschanged = () => { };