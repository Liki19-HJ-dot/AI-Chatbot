require("dotenv").config();
const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
const rateLimit = require("express-rate-limit");
const path = require("path");
const { respond, wantsImage, lastUserText, ChatError, MODES } = require("./chat");

const app = express();
const PORT = process.env.PORT || 4000;

app.set("trust proxy", 1); // correct client IPs behind Render/Heroku/etc. (needed for rate limits)
app.use(helmet({ contentSecurityPolicy: false })); // the page uses an inline script
app.use(cors({ origin: process.env.APP_ORIGIN || false }));
app.use(express.json({ limit: "100kb" }));

// Protect the AI bill: per-IP limits, stricter for images.
const chatLimiter = rateLimit({
  windowMs: 10 * 60 * 1000,
  max: 40,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "You're sending messages very quickly. Please wait a few minutes and try again." },
});
const imageLimiter = rateLimit({
  windowMs: 10 * 60 * 1000,
  max: 10,
  skip: (req) => {
    const mode = req.body?.mode || "auto";
    return !(mode === "image" || (mode === "auto" && wantsImage(lastUserText(req.body?.messages))));
  },
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Image limit reached for now. Please wait a few minutes and try again." },
});

// POST /api/chat  { messages: [{role, content}], mode?: "auto"|"chat"|"search"|"creative"|"image" }
app.post("/api/chat", chatLimiter, imageLimiter, async (req, res) => {
  const { messages, mode = "auto" } = req.body || {};
  if (!MODES.includes(mode)) {
    return res.status(400).json({ error: `mode must be one of: ${MODES.join(", ")}` });
  }
  try {
    res.json(await respond({ messages, mode }));
  } catch (err) {
    if (err instanceof ChatError) {
      console.error("Chat failed:", err.message); // always log the real reason on the server
      return res.status(err.status).json({ error: err.publicMessage });
    }
    console.error("Chat failed:", err);
    res.status(500).json({ error: "Something went wrong. Please try again." });
  }
});

app.get("/health", (req, res) =>
  res.json({ status: "ok", mode: (process.env.AI_MODE || "mock").toLowerCase() })
);

app.use(express.static(path.join(__dirname, "..", "public"), { extensions: ["html"] }));

app.use("/api", (req, res) => res.status(404).json({ error: "That request couldn't be found." }));

app.use((err, req, res, next) => {
  if (res.headersSent) return next(err);
  if (err && err.type === "entity.too.large") {
    return res.status(413).json({ error: "That message is too large. Please send something shorter." });
  }
  if (err && err.type === "entity.parse.failed") {
    return res.status(400).json({ error: "We couldn't read that request. Please try again." });
  }
  console.error("Unhandled error:", err && err.message ? err.message : err);
  res.status(500).json({ error: "Something went wrong. Please try again." });
});

app.listen(PORT, () => {
  console.log(`Open AI Chat running on http://localhost:${PORT}`);
  console.log(`AI_MODE=${(process.env.AI_MODE || "mock").toLowerCase()}`);
  if ((process.env.AI_MODE || "mock").toLowerCase() === "live") {
    console.log(process.env.GEMINI_API_KEY
      ? `GEMINI_API_KEY: found (ends in ...${process.env.GEMINI_API_KEY.slice(-4)})`
      : "WARNING: AI_MODE is live but GEMINI_API_KEY is empty. Put your key in .env and restart.");
  }
});
