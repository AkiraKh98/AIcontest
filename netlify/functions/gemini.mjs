const MAX_BODY_CHARS = 8192;
const GEMINI_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent';
const stripSlash = (s) => (s || '').replace(/\/+$/, '');

export default async (req) => {
  if (req.method !== 'POST') {
    return new Response('Method Not Allowed', { status: 405 });
  }

  const allowed = process.env.ALLOWED_ORIGIN;
  if (!allowed) {
    return new Response('Server misconfigured: ALLOWED_ORIGIN missing', { status: 500 });
  }
  if (stripSlash(req.headers.get('origin')) !== stripSlash(allowed)) {
    return new Response('Forbidden', { status: 403 });
  }

  const key = process.env.GEMINI_KEY;
  if (!key) {
    return new Response('Server misconfigured: GEMINI_KEY missing', { status: 500 });
  }

  const raw = await req.text();
  if (raw.length > MAX_BODY_CHARS) {
    return new Response('Body too large', { status: 413 });
  }

  let parsed;
  try { parsed = JSON.parse(raw); }
  catch { return new Response('Invalid JSON', { status: 400 }); }

  if (!parsed || !Array.isArray(parsed.contents)) {
    return new Response('Invalid body: expected { contents: [...] }', { status: 400 });
  }

  const safeBody = JSON.stringify({ contents: parsed.contents });
  const upstream = await fetch(GEMINI_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-goog-api-key': key,
    },
    body: safeBody,
  });
  return new Response(upstream.body, {
    status: upstream.status,
    headers: { 'Content-Type': 'application/json' },
  });
};
