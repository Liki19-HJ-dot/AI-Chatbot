/**
 * Chat service: answers questions, searches the web, writes creative copy,
 * and generates images, all through the Google Gemini API.
 *
 * AI_MODE=mock  -> placeholder replies, no key needed (good for testing the UI)
 * AI_MODE=live  -> real Gemini calls (needs GEMINI_API_KEY)
 */

const MODES = ["auto", "chat", "search", "creative", "image"];
const MAX_HISTORY = 20; // messages sent to the model
const MAX_CHARS = 4000; // per message
const TEXT_TIMEOUT_MS = 60000;
const IMAGE_TIMEOUT_MS = 120000;

function cfg() {
  return {
    live: (process.env.AI_MODE || "mock").toLowerCase() === "live",
    apiKey: process.env.GEMINI_API_KEY || "",
    baseUrl: (process.env.GEMINI_BASE_URL || "https://generativelanguage.googleapis.com/v1beta").replace(/\/+$/, ""),
    textModel: process.env.GEMINI_TEXT_MODEL || "gemini-3.6-flash",
    imageModel: process.env.GEMINI_IMAGE_MODEL || "gemini-3.1-flash-image",
  };
}

class ChatError extends Error {
  constructor(message, status = 502, publicMessage) {
    super(message);
    this.status = status;
    this.publicMessage =
      publicMessage || (status >= 500 ? "Something went wrong. Please try again." : message);
  }
}

const SYSTEM_PROMPTS = {
  chat:
    "You are Open AI Chat, a friendly and accurate assistant. Answer every question helpfully. " +
    "For programming questions, briefly explain the approach, give complete working code in labelled Markdown code blocks, " +
    "and mention common mistakes. For simple or casual questions keep the answer short. " +
    "Use Markdown (lists, code blocks) when it helps. If you are not sure about something, say so instead of guessing.",
  search:
    "You are a research assistant with web search. Use search to answer with current, factual information. " +
    "Summarise clearly in Markdown, mention dates when they matter, and say so when sources disagree or you could not find something. " +
    "Never invent facts or sources.",
  creative:
    "You are a creative partner. Write captions, hooks, scripts, slogans, stories and ideas with a strong voice. " +
    "Unless asked for just one, give 3 distinct options. Match the platform and tone the user mentions. Format with Markdown.",
};

// ---------- Image intent detection ----------
// An image is generated ONLY when the message clearly asks to make one.
// Talking ABOUT images ("how do I edit an image?") stays a normal chat reply.
const IMAGE_NOUNS =
  "(?:image|images|picture|pictures|pic|pics|photo|photos|photograph|illustration|artwork|drawing|painting|sketch|" +
  "logo|poster|banner|thumbnail|wallpaper|avatar|icon|infographic|flyer|meme|portrait|render|cartoon|graphic)";
const IMAGE_VERBS =
  "(?:generate|create|make|draw|design|paint|produce|render|sketch|show me|give me|imagine|illustrate|build)";

const IMAGE_PATTERNS = [
  new RegExp(`\\b${IMAGE_VERBS}\\b[^.!?\\n]{0,40}?\\b${IMAGE_NOUNS}\\b`, "i"),
  /^\s*(?:please\s+|pls\s+|can you\s+|could you\s+)?(?:draw|paint|sketch|illustrate)\s+(?:me\s+)?(?:a|an|the|some)\b/i,
  new RegExp(`^\\s*(?:an?\\s+)?${IMAGE_NOUNS}\\s+(?:of|showing|with)\\b`, "i"),
  /^\s*\/(?:image|imagine|img|draw)\b/i,
];

function wantsImage(text) {
  if (typeof text !== "string") return false;
  const t = text.trim();
  return !!t && IMAGE_PATTERNS.some((re) => re.test(t));
}

function cleanImagePrompt(text) {
  let t = text.trim().replace(/^\/(?:image|imagine|img|draw)\s*/i, "");
  t = t.replace(
    new RegExp(
      `^(?:please\\s+|pls\\s+|can you\\s+|could you\\s+)?(?:${IMAGE_VERBS})\\s+(?:me\\s+)?(?:an?\\s+|the\\s+|some\\s+)?(?:${IMAGE_NOUNS})\\s+(?:of|showing|with|for|about)\\s+`,
      "i"
    ),
    ""
  );
  return (t || text).trim().slice(0, 1000);
}

function lastUserText(messages) {
  const last = [...(Array.isArray(messages) ? messages : [])]
    .reverse()
    .find((m) => m && m.role === "user");
  return typeof last?.content === "string" ? last.content.trim() : "";
}

// Browser [{role, content}] -> valid model input: user/assistant only, trimmed,
// alternating roles, starting and ending with the user.
function buildContents(messages) {
  if (!Array.isArray(messages)) throw new ChatError("messages must be an array.", 400);
  const cleaned = [];
  for (const m of messages.slice(-MAX_HISTORY)) {
    if (!m || (m.role !== "user" && m.role !== "assistant")) continue;
    const text = typeof m.content === "string" ? m.content.trim().slice(0, MAX_CHARS) : "";
    if (!text) continue;
    const last = cleaned[cleaned.length - 1];
    if (last && last.role === m.role) last.content += "\n\n" + text;
    else cleaned.push({ role: m.role, content: text });
  }
  while (cleaned.length && cleaned[0].role !== "user") cleaned.shift();
  if (!cleaned.length || cleaned[cleaned.length - 1].role !== "user") {
    throw new ChatError("Type a message to start the chat.", 400);
  }
  return cleaned;
}

function extractSources(candidate) {
  const seen = new Set();
  const out = [];
  for (const chunk of candidate?.groundingMetadata?.groundingChunks || []) {
    const w = chunk?.web;
    if (!w?.uri || seen.has(w.uri)) continue;
    seen.add(w.uri);
    out.push({ title: w.title || w.uri, url: w.uri });
  }
  return out.slice(0, 6);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Turns a Gemini error response into a ChatError. The real detail is always logged on the
// server; the browser gets a short friendly message (or the raw detail if DEBUG_ERRORS=true).
function geminiError(status, bodyText) {
  const detail = String(bodyText || "").slice(0, 600);
  let apiStatus = "";
  let message = "";
  try {
    const e = JSON.parse(detail).error || {};
    apiStatus = e.status || "";
    message = e.message || "";
  } catch (_) {}
  const raw = `Gemini error ${status}${apiStatus ? ` (${apiStatus})` : ""}: ${message || detail}`;
  const debug = String(process.env.DEBUG_ERRORS || "").toLowerCase() === "true";
  const pick = (msg) => (debug ? raw : msg);
  const text = `${apiStatus} ${message}`;

  if (/API key not valid|API_KEY_INVALID|API key expired/i.test(text) || status === 401) {
    return new ChatError(raw, 503, pick("The AI key was rejected. Check GEMINI_API_KEY on the server."));
  }
  if (status === 403) {
    return new ChatError(raw, 503, pick("The AI key doesn't have permission for this request. Check your Gemini API key and project."));
  }
  if (status === 404 || (status === 400 && /model/i.test(message))) {
    return new ChatError(raw, 502, pick("That AI model isn't available for this key. Check GEMINI_TEXT_MODEL / GEMINI_IMAGE_MODEL."));
  }
  if (status === 429) {
    const daily = /per ?day|daily/i.test(text);
    return new ChatError(
      raw,
      429,
      pick(daily
        ? "The free daily AI limit has been used up. Try again tomorrow, or enable billing for the Gemini key."
        : "The AI is busy right now. Please try again in a moment.")
    );
  }
  if (status === 503) {
    return new ChatError(raw, 503, pick("The AI is overloaded right now. Please try again in a moment."));
  }
  return new ChatError(raw, 502, pick("The AI service had a problem. Please try again."));
}

async function geminiPost(model, body, timeoutMs) {
  const { apiKey, baseUrl } = cfg();
  if (!apiKey) {
    throw new ChatError(
      "AI_MODE is 'live' but GEMINI_API_KEY is not set.",
      503,
      "The AI isn't available right now. Please try again later."
    );
  }

  const MAX_TRIES = 3; // retry short "busy" errors, never key / model / daily-limit errors
  for (let attempt = 1; attempt <= MAX_TRIES; attempt++) {
    let resp;
    try {
      resp = await fetch(`${baseUrl}/models/${encodeURIComponent(model)}:generateContent`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-goog-api-key": apiKey },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(timeoutMs),
      });
    } catch (err) {
      throw new ChatError(`Gemini request failed: ${err.message}`, 504, "The AI took too long to answer. Please try again.");
    }
    if (resp.ok) return resp.json();

    const errText = await resp.text();
    const error = geminiError(resp.status, errText);
    const daily = /per ?day|daily/i.test(errText);
    const transient = (resp.status === 429 && !daily) || resp.status === 503;
    if (transient && attempt < MAX_TRIES) {
      const ra = Number(resp.headers.get("retry-after"));
      const wait = Math.min(Number.isFinite(ra) && ra > 0 ? ra * 1000 : attempt * 1500, 6000);
      console.warn(`${error.message} - retrying in ${Math.round(wait / 1000)}s (attempt ${attempt}/${MAX_TRIES})`);
      await sleep(wait);
      continue;
    }
    throw error;
  }
}

// Browser/user messages -> Gemini "contents" (assistant role is called "model").
function toGeminiContents(contents) {
  return contents.map((m) => ({ role: m.role === "assistant" ? "model" : "user", parts: [{ text: m.content }] }));
}

function partsOf(data) {
  return data?.candidates?.[0]?.content?.parts || [];
}

async function callText(contents, mode) {
  const { textModel } = cfg();
  const body = {
    systemInstruction: { parts: [{ text: SYSTEM_PROMPTS[mode] }] },
    contents: toGeminiContents(contents),
    generationConfig: { maxOutputTokens: 8000 },
  };
  if (mode === "search") body.tools = [{ google_search: {} }];

  const data = await geminiPost(textModel, body, TEXT_TIMEOUT_MS);
  const candidate = data?.candidates?.[0];

  const text = partsOf(data)
    .filter((p) => typeof p.text === "string" && !p.thought)
    .map((p) => p.text)
    .join("")
    .trim();

  if (!text) {
    const blocked = data?.promptFeedback?.blockReason || /SAFETY|BLOCK|PROHIBITED/i.test(candidate?.finishReason || "");
    return {
      reply: blocked
        ? "I can't help with that request. Try rephrasing it or asking something different."
        : "I didn't get an answer back. Please try asking again.",
      sources: [],
    };
  }
  return { reply: text, sources: mode === "search" ? extractSources(candidate) : [] };
}

async function callImage(prompt) {
  const { imageModel } = cfg();
  const data = await geminiPost(
    imageModel,
    {
      contents: [{ role: "user", parts: [{ text: `Create an image: ${prompt}` }] }],
      generationConfig: { responseModalities: ["TEXT", "IMAGE"] },
    },
    IMAGE_TIMEOUT_MS
  );
  for (const p of partsOf(data)) {
    const inline = p.inlineData || p.inline_data;
    if (inline?.data) {
      const mime = inline.mimeType || inline.mime_type || "image/png";
      return `data:${mime};base64,${inline.data}`;
    }
  }
  throw new ChatError(
    "Gemini returned no image: " + JSON.stringify(data).slice(0, 300),
    422,
    "The image couldn't be created. Please try a different description."
  );
}

// ---------- Mock mode (no API key) ----------
function escapeXml(s) {
  return String(s).replace(/[<>&"']/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;", "'": "&apos;" }[c]));
}

function mockImage(prompt) {
  const words = escapeXml(prompt.slice(0, 80));
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">` +
    `<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6d5efc"/><stop offset="1" stop-color="#22d3ee"/></linearGradient></defs>` +
    `<rect width="1024" height="1024" fill="url(#g)"/>` +
    `<circle cx="780" cy="240" r="120" fill="#ffffff" opacity="0.18"/><circle cx="240" cy="800" r="200" fill="#ffffff" opacity="0.12"/>` +
    `<text x="512" y="480" font-family="Arial, sans-serif" font-size="44" font-weight="700" fill="#fff" text-anchor="middle">Demo image</text>` +
    `<text x="512" y="550" font-family="Arial, sans-serif" font-size="30" fill="#fff" text-anchor="middle">${words}</text>` +
    `</svg>`;
  return "data:image/svg+xml;base64," + Buffer.from(svg).toString("base64");
}

function mockReply(mode, lastText) {
  const topic = lastText.length > 120 ? lastText.slice(0, 117) + "..." : lastText;
  const tail =
    mode === "creative"
      ? "With a live key you'd get 3 ready-to-use creative options for this."
      : mode === "search"
        ? "With a live key I'd search the web and answer with source links."
        : "With a live key I'd answer this in full and remember the conversation.";
  return {
    reply:
      `**Demo mode:** no Gemini key is connected yet, so this is a placeholder reply.\n\n` +
      `You asked: "${topic}"\n\n${tail}\n\n` +
      `To go live, set \`AI_MODE=live\` and \`GEMINI_API_KEY\` on the server.`,
    sources: [],
    mock: true,
  };
}

async function respond({ messages, mode = "auto" }) {
  if (!MODES.includes(mode)) throw new ChatError(`mode must be one of: ${MODES.join(", ")}`, 400);

  let effective = mode;
  if (mode === "auto") effective = wantsImage(lastUserText(messages)) ? "image" : "chat";

  if (effective === "image") {
    const raw = lastUserText(messages);
    if (!raw) throw new ChatError("Describe the image you want.", 400);
    const prompt = mode === "image" ? raw.slice(0, 1000) : cleanImagePrompt(raw);
    if (!cfg().live) {
      return { mode: "image", reply: "Here's a demo image (connect a Gemini key for real images).", image: mockImage(prompt), sources: [], mock: true };
    }
    const image = await callImage(prompt);
    return { mode: "image", reply: "Here's your image.", image, sources: [] };
  }

  const contents = buildContents(messages);
  if (!cfg().live) {
    return { mode: effective, ...mockReply(effective, contents[contents.length - 1].content) };
  }
  return { mode: effective, ...(await callText(contents, effective)) };
}

module.exports = { respond, wantsImage, lastUserText, ChatError, MODES };
