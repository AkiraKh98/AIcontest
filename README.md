# PIPEL\AINE

A multimodal scene-narrator demo. The browser watches the user through their
webcam, classifies the scene, reads pose and face landmarks, takes a voice
prompt, and asks Gemini to write a short narration that a 3D avatar then
speaks aloud.

## Stack

- **Vite** — dev server / bundler
- **three.js** — avatar scene + animation
- **MediaPipe Tasks Vision** — pose + face landmarking in-browser
- **Teachable Machine** — image classifier for scene tagging
- **Gemini 2.5 Flash Lite** — narration generation, proxied through a Netlify Function
- **Web Speech API** — TTS

Single-file app: everything is in `src/main.js`.

## Layout

```
src/main.js              Whole app
public/                  Static assets (avatar GLB, etc.)
netlify.toml             Build + dev config
netlify/functions/
  gemini.mjs             Server-side Gemini proxy (hides API key)
```

## Local development

You need the Netlify CLI so the function runs alongside Vite:

```bash
npm install -g netlify-cli
npm install
netlify dev
```

`netlify dev` starts Vite on 5173 and proxies it on **http://localhost:8888**,
where the function is also served. Don't open 5173 directly — the function
call will 404.

### Required env vars

Create a `.env` (gitignored) at the repo root:

```
GEMINI_KEY=<your Gemini API key>
ALLOWED_ORIGIN=http://localhost:8888
```

`netlify dev` reads `.env` (not `.env.local`) for function env vars.

## Deploy (Netlify)

1. Push to GitHub.
2. In Netlify: **Add new site → Import from Git** → pick this repo. Build
   settings auto-fill from `netlify.toml`.
3. **Site settings → Environment variables** — add:
   - `GEMINI_KEY` — your Gemini API key
   - `ALLOWED_ORIGIN` — the deployed site URL, no trailing slash (e.g.
     `https://your-site.netlify.app`)
4. Trigger a redeploy if you added env vars after the first build.

Pushes to `master` auto-deploy after that.

## The Gemini proxy

The client calls `/.netlify/functions/gemini` instead of Google directly, so
the API key never ships in the bundle. The function:

- Requires `POST` from an `Origin` matching `ALLOWED_ORIGIN`
- Caps request body at 8192 characters
- Re-serializes the body to forward only the `contents` field, stripping
  `safetySettings`, `generationConfig`, `tools`, etc.

This stops casual abuse but not a determined attacker who spoofs `Origin`. If
you see unexpected Gemini usage in production, add a per-session token or a
CAPTCHA (Cloudflare Turnstile, hCaptcha) in front of the function.

## Known issues

- The avatar still loads in a T-pose; the visible "fix" is `frameOnHead()`
  cropping the camera tightly to the head. Upper-arm Z-axis rotation
  silently no-ops — likely a glTF rest-pose / parent-transform mismatch.
  Deferred.
