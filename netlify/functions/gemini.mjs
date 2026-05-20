const MAX_BODY_BYTES = 8192;
const MAX_PROMPT_CHARS = 4000;

export default async (req) => {
  if (req.method !== 'POST') {
    return new Response('Method Not Allowed', { status: 405 });
  }

  const allowed = process.env.ALLOWED_ORIGIN;
  if (!allowed) {
    return new Response('Server misconfigured: ALLOWED_ORIGIN missing', { status: 500 });
  }
  const stripSlash = (s) => (s || '').replace(/\/+$/, '');
  if (stripSlash(req.headers.get('origin')) !== stripSlash(allowed)) {
    return new Response('Forbidden', { status: 403 });
  }

  const key = process.env.GEMINI_KEY;
  if (!key) {
    return new Response('Server misconfigured: GEMINI_KEY missing', { status: 500 });
  }

  const raw = await req.text();
  if (raw.length > MAX_BODY_BYTES) {
    return new Response('Body too large', { status: 413 });
  }

  let parsed;
  try { parsed = JSON.parse(raw); }
  catch { return new Response('Invalid JSON', { status: 400 }); }

  if (!parsed || !Array.isArray(parsed.contents)) {
    return new Response('Invalid body: expected { contents: [...] }', { status: 400 });
  }
  const totalChars = JSON.stringify(parsed.contents).length;
  if (totalChars > MAX_PROMPT_CHARS) {
    return new Response('Prompt too large', { status: 413 });
  }

  const safeBody = JSON.stringify({ contents: parsed.contents });
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key=${key}`;
  const upstream = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: safeBody,
  });
  return new Response(upstream.body, {
    status: upstream.status,
    headers: { 'Content-Type': 'application/json' },
  });
};
