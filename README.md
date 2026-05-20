# PIPEL\AINE

A live, browser-only scene narrator. Point your webcam, speak a request,
and a 3D avatar speaks back with a short narration that weaves together
what the camera sees and what you asked for.

**Live demo: https://mjeed.netlify.app/**

![PIPEL\AINE running — five panels showing scene tag, biometrics, voice input,
generated narration, and the speaking 3D avatar](docs/screenshot.png)

## What it does

Each "narration" is generated from three live signals fused together:

```
 webcam frame ─── Teachable Machine ──► scene tag (e.g. "close_up")
 webcam frame ─── MediaPipe ─────────► pose + face landmarks
 microphone ───── Web Speech API ────► transcribed user request
                                       │
                                       ▼
                                Gemini 2.5 Flash Lite
                                       │
                                       ▼
                     Web Speech TTS ─► 3D avatar (three.js)
                     with synced jaw morph
```

The UI is laid out as five panels: **[01] SCENE**, **[02] BIOMETRICS**,
**[03] OPERATOR INPUT**, **[04] NARRATION**, **[05] EMBODIMENT** — so a
judge or visitor can see every stage of the pipeline running in real time.

## Try the live demo

1. Open https://mjeed.netlify.app/ in **desktop Chrome** (it's the only
   browser that supports both Web Speech recognition and the WebGL features
   used here).
2. Allow **camera** and **microphone** access when prompted.
3. Stand in frame, wait for **panels [01] and [02]** to light up `LIVE`.
4. Click **ENGAGE MIC** and speak a request — e.g.
   *"Describe this scene like a film noir narrator."*
5. Click **GENERATE NARRATION**. The avatar speaks the result.

The Gemini API key is proxied server-side; visitors don't need their own
key to try the demo.

## Run it yourself

You'll need Node 18+ and a Gemini API key from
[ai.google.dev](https://ai.google.dev/).

```bash
git clone https://github.com/AkiraKh98/AIcontest.git
cd AIcontest
npm install
npm install -g netlify-cli   # one-time
```

Create a `.env` (gitignored) at the repo root:

```
GEMINI_KEY=<your Gemini API key>
ALLOWED_ORIGIN=http://localhost:8888
```

Then:

```bash
netlify dev
```

Open **http://localhost:8888** (not `:5173` — the function won't be
reachable there). Hardware: webcam, microphone, and speakers. Chrome only.

## How it's wired

- **Vite 8** — dev server / bundler
- **three.js** — 3D avatar scene, GLB loading, jaw morph for lip sync
- **MediaPipe Tasks Vision** — pose + face landmarking, fully in-browser
- **Teachable Machine** — drop-in image classifier
  ([model URL](https://teachablemachine.withgoogle.com/models/vhBhcXpCZ/))
- **Web Speech API** — STT (Chrome `webkitSpeechRecognition`) and TTS
- **Gemini 2.5 Flash Lite** — narration, called through a Netlify Function
  proxy so the API key never ships in the bundle
- **Netlify** — static hosting + serverless function for the proxy

The entire client is a single file: `src/main.js`. The function is
`netlify/functions/gemini.mjs`.

## Deploying your own copy

1. Fork the repo and push to GitHub.
2. In Netlify: **Add new site → Import from Git** → pick your fork. Build
   settings auto-fill from `netlify.toml`.
3. **Site settings → Environment variables** — add:
   - `GEMINI_KEY` — your Gemini API key
   - `ALLOWED_ORIGIN` — the deployed site URL (trailing slash tolerated)
4. Trigger a redeploy if you added env vars after the first build.

Pushes to `master` auto-deploy after that.

## How the proxy protects the key

`netlify/functions/gemini.mjs` is a thin pass-through that:

- Requires `POST` from an `Origin` matching `ALLOWED_ORIGIN`
- Caps request body at 8192 characters
- Re-serializes the body to forward only the `contents` field, stripping
  any `safetySettings`, `generationConfig`, `tools`, etc.
- Sends the API key as the `x-goog-api-key` header (not in the URL)

This stops casual abuse but not a determined attacker who spoofs `Origin`.
For higher-stakes use add a per-session token or a CAPTCHA in front of
the function.

## Known issues

- The avatar still loads in a T-pose; the visible "fix" in `frameOnHead()`
  crops the camera tightly to the head. Upper-arm Z-axis rotation silently
  no-ops — likely a glTF rest-pose / parent-transform mismatch. Doesn't
  affect the head-and-shoulders shot the demo uses.
- Demo quota is shared. If `[04]` shows `HTTP 429 — quota exceeded`, the
  free-tier daily limit on the deployed key has been hit; try again
  tomorrow or run your own copy.
