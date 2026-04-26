"""
AI-Powered News Analysis Application
=====================================
Backend : Flask (Python)
News API: NewsAPI.org  — free at newsapi.org
AI API  : Groq         — free at console.groq.com (no credit card needed)

Features:
  - Fetch trending news by category & country
  - AI analysis (summary, sentiment, key point) per article
  - AI editorial digest across all headlines
  - News chatbot — ask anything about the loaded articles

Setup:
  pip install -r requirements.txt
  cp .env.example .env
  python app.py  →  open http://localhost:5000
"""

import os
import json
import requests
from datetime import datetime
from flask import Flask, jsonify, request, render_template

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

NEWSAPI_HEADLINES  = "https://newsapi.org/v2/top-headlines"
NEWSAPI_EVERYTHING = "https://newsapi.org/v2/everything"
GROQ_API_URL       = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL         = "llama-3.1-8b-instant"

CATEGORY_KEYWORDS = {
    "general":       "world news today",
    "technology":    "technology AI software",
    "business":      "business economy finance",
    "health":        "health medicine",
    "science":       "science research discovery",
    "sports":        "sports",
    "entertainment": "entertainment movies music",
}

# ─────────────────────────────────────────────────────────────────────────────
# Groq helpers
# ─────────────────────────────────────────────────────────────────────────────

def groq_chat(messages: list, max_tokens: int = 500) -> str:
    """
    Send a list of {role, content} messages to Groq.
    Supports multi-turn conversation for the chatbot.
    """
    if not GROQ_API_KEY:
        return None
    resp = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={"model": GROQ_MODEL, "max_tokens": max_tokens, "messages": messages},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def groq_single(prompt: str, max_tokens: int = 400) -> str:
    """Convenience wrapper for single-turn prompts."""
    return groq_chat([{"role": "user", "content": prompt}], max_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# News fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_news(category="general", country="us", page_size=12, api_key=""):
    key = api_key or NEWS_API_KEY
    if not key:
        return {"status": "error", "message": "NEWS_API_KEY not set. Add it to your .env file."}

    def clean(arts):
        return [a for a in arts if a.get("title") and a["title"] != "[Removed]"]

    # Try paid endpoint first
    try:
        r = requests.get(NEWSAPI_HEADLINES, params={
            "country": country, "category": category,
            "pageSize": page_size, "apiKey": key,
        }, timeout=10)
        data = r.json()
        if data.get("status") == "ok":
            data["articles"] = clean(data.get("articles", []))
            data["source"] = "top-headlines"
            return data
    except Exception:
        pass

    # Free tier fallback
    try:
        query = CATEGORY_KEYWORDS.get(category, category)
        r = requests.get(NEWSAPI_EVERYTHING, params={
            "q": query, "language": "en",
            "sortBy": "publishedAt", "pageSize": page_size, "apiKey": key,
        }, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "ok":
            data["articles"] = clean(data.get("articles", []))
            data["source"] = "everything"
        return data
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# AI — per-article analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_article(title, description, source):
    if not GROQ_API_KEY:
        return {
            "summary":   f"{source} reports: {description or title}",
            "sentiment": "neutral",
            "keyPoint":  "Add GROQ_API_KEY to .env for free AI analysis.",
            "demo":      True,
        }
    prompt = f"""Analyze this news article. Return ONLY valid JSON — no markdown.

Title: "{title}"
Description: "{description or 'N/A'}"
Source: {source}

JSON shape: {{"summary":"<2 sentences>","sentiment":"<positive|negative|neutral|mixed>","keyPoint":"<one short phrase>"}}"""

    try:
        raw = groq_single(prompt, 350).replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"summary": raw, "sentiment": "neutral", "keyPoint": ""}
    except Exception as e:
        return {"error": str(e), "summary": "", "sentiment": "neutral", "keyPoint": ""}


# ─────────────────────────────────────────────────────────────────────────────
# AI — editorial digest
# ─────────────────────────────────────────────────────────────────────────────

def generate_digest(articles):
    headlines = "\n".join(
        f"{i+1}. {a['title']} ({a.get('source', {}).get('name', 'Unknown')})"
        for i, a in enumerate(articles[:8])
    )
    if not GROQ_API_KEY:
        sources = ", ".join(filter(None, {a.get("source", {}).get("name") for a in articles[:4]}))
        return (f"Today's digest covers {len(articles)} stories from {sources or 'various sources'}.\n\n"
                "Set GROQ_API_KEY in your .env to unlock free AI-generated editorial digests.")
    try:
        return groq_single(
            "You are an editorial AI for a prestigious news publication. "
            "Write a concise intelligence digest in 3 short paragraphs (2-3 sentences each). "
            "Journalistic, slightly literary tone. Synthesize the key narrative thread. "
            "Do NOT list headlines — write flowing prose.\n\n"
            f"Headlines:\n{headlines}", 700)
    except Exception as e:
        return f"Digest generation failed: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# AI — news chatbot  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def build_news_context(articles: list) -> str:
    """Format loaded articles into a context block for the chatbot."""
    if not articles:
        return "No articles have been loaded yet."
    lines = []
    for i, a in enumerate(articles[:12], 1):
        lines.append(
            f"[{i}] {a.get('title','')}\n"
            f"    Source: {a.get('source',{}).get('name','Unknown')}  |  "
            f"Published: {a.get('publishedAt','')[:10]}\n"
            f"    {a.get('description','No description.')}"
        )
    return "\n\n".join(lines)


CHATBOT_SYSTEM = """You are a helpful news assistant. You have access to a set of currently loaded news articles provided in the context below. Your job is to:
- Answer questions about these specific news articles
- Summarize individual stories when asked
- Compare or group stories by topic
- Explain what's happening in simple terms
- Answer follow-up questions about any article

Rules:
- Only discuss the news articles provided. Do not make up information.
- If the user asks about something not in the articles, politely say it's not in the current feed.
- Keep answers concise and clear.
- When referencing an article, mention the source name.
"""


def chatbot_reply(user_message: str, history: list, articles: list) -> str:
    """
    Generate a chatbot reply given:
      user_message — latest user text
      history      — list of {role, content} for prior turns
      articles     — currently loaded news articles
    """
    if not GROQ_API_KEY:
        return ("⚠️ Groq API key not set. Add GROQ_API_KEY to your .env file "
                "to enable the news chatbot.")

    news_context = build_news_context(articles)

    # Build message list: system → history → new user message
    messages = [
        {"role": "system", "content": f"{CHATBOT_SYSTEM}\n\n--- CURRENT NEWS ARTICLES ---\n{news_context}\n--- END OF ARTICLES ---"},
        *history[-10:],   # keep last 10 turns to stay within context limit
        {"role": "user",  "content": user_message},
    ]

    try:
        return groq_chat(messages, max_tokens=600)
    except Exception as e:
        return f"Sorry, I couldn't process that: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "status":        "ok",
        "newsApiKeySet": bool(NEWS_API_KEY),
        "groqKeySet":    bool(GROQ_API_KEY),
        "groqModel":     GROQ_MODEL,
        "timestamp":     datetime.utcnow().isoformat(),
    })


@app.route("/api/news")
def api_news():
    data = fetch_news(
        category  = request.args.get("category", "general"),
        country   = request.args.get("country",  "us"),
        page_size = int(request.args.get("pageSize", 12)),
        api_key   = request.args.get("apiKey", ""),
    )
    return jsonify(data)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    body = request.get_json(force=True) or {}
    if not body.get("title"):
        return jsonify({"error": "title is required"}), 400
    return jsonify(analyze_article(
        body.get("title", ""),
        body.get("description", ""),
        body.get("source", "Unknown"),
    ))


@app.route("/api/digest", methods=["POST"])
def api_digest():
    body = request.get_json(force=True) or {}
    if not body.get("articles"):
        return jsonify({"error": "articles list is required"}), 400
    return jsonify({
        "digest":      generate_digest(body["articles"]),
        "generatedAt": datetime.utcnow().isoformat(),
    })


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    POST /api/chat
    Body: {
      message:  "What happened with the tech layoffs?",
      history:  [{role:"user",content:"..."},{role:"assistant",content:"..."}],
      articles: [...]   — the currently loaded news articles
    }
    Response: { reply: "..." }
    """
    body     = request.get_json(force=True) or {}
    message  = body.get("message", "").strip()
    history  = body.get("history",  [])
    articles = body.get("articles", [])

    if not message:
        return jsonify({"error": "message is required"}), 400

    reply = chatbot_reply(message, history, articles)
    return jsonify({"reply": reply})


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 56)
    print("  AI News Analysis - Groq Edition (100% Free)")
    print("=" * 56)
    print(f"  NEWS_API_KEY : {'[OK] set' if NEWS_API_KEY else '[X] not set - add to .env'}")
    print(f"  GROQ_API_KEY : {'[OK] set' if GROQ_API_KEY else '[X] not set - get free key at console.groq.com'}")
    print(f"  Groq model   : {GROQ_MODEL}")
    print("  URL          : http://localhost:5000")
    print("=" * 56)
    app.run(debug=True, host="0.0.0.0", port=5000)
