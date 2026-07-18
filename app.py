#!/usr/bin/env python3
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import json
import re
import time
import io
import threading
import datetime
import ssl
import urllib.parse
import urllib.request
import traceback
from flask import Flask, jsonify, request, render_template

# Try loading env variables from .env if it exists, otherwise fall back to .env.example
for env_file in [".env", ".env.example"]:
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        # Load values if not already present or if loading a non-empty value in .env
                        if key:
                            if env_file == ".env" or not os.environ.get(key):
                                if val: # only set if there is a value
                                    os.environ[key] = val
            print(f"Loaded environment variables from {env_file} in Python", file=sys.stdout)
        except Exception as e:
            print(f"Error loading {env_file} in Python: {e}", file=sys.stderr)

# Initialize Flask App
app = Flask(__name__)
PORT = 5000

# Cache buffers for session feed ingestion
NEWS_FEED_BUFFER = []
SOCIAL_FEED_BUFFER = []

NEWS_CACHE = None    # {"timestamp": float, "data": list}
SOCIAL_CACHE = None  # {"timestamp": float, "data": list}
CACHE_TTL = 300.0    # 5 minutes in seconds

# No pre-populated mock data — dashboard starts empty; real data arrives via live API ingestion
STATIC_DISASTER_REPORTS = []

# ─────────────────────────────────────────────────────────────────────────────
# NLP PIPELINE — imported from nlp/pipeline.py
# ALL 3 models perform ALL 5 tasks independently (research comparison).
# Models: google/mt5-small | ai4bharat/IndicBART | facebook/mbart-large-50
# Tasks:  disaster_classification | location_extraction | translation |
#         sentiment | summarization
# Fine-tuned checkpoints (nlp/train_all.py) used when available.
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from nlp.pipeline import (
        run_single_model   as _run_single_model,
        run_comparison     as _run_comparison,
        detect_language,
        extract_numeric_impact,
        load_eval_metrics,
        MODEL_REGISTRY,
        TASK_NAMES,
    )
    _NLP_OK = True
    print("[App] nlp.pipeline loaded.", file=sys.stdout, flush=True)
except Exception as _nlp_err:
    print(f"[App] WARNING — nlp.pipeline unavailable: {_nlp_err}", file=sys.stderr)
    _NLP_OK = False
    MODEL_REGISTRY = {
        "mt5":       {"display": "mT5-Small"},
        "indicbart": {"display": "IndicBART"},
        "mbart":     {"display": "mBART-50"},
    }
    TASK_NAMES = [
        "disaster_classification", "location_extraction",
        "translation", "sentiment", "summarization",
    ]

DISASTER_KEYWORDS = [
    "flood", "landslide", "cyclone", "earthquake", "flash flood", "heatwave", "drought", 
    "tsunami", "avalanche", "cloudburst", "ndrf", "sdrf", "rescue", "evacu", "casualt",
    "पूर", "दरड", "कोसळ", "भूकंप", "चक्रवात", "वादळ", "उष्णतेची लाट", "बाढ़", "भूस्खलन", "तूफान"
]

def is_disaster_related(title: str, text: str) -> bool:
    content = (title + " " + text).lower()
    return any(kw in content for kw in DISASTER_KEYWORDS)


def extract_disaster_intel(text: str, model_name: str = "mt5") -> dict:
    return _run_single_model(text, model_key=model_name)


@app.route("/api/fetch-news", methods=["GET"])
def fetch_news():
    global NEWS_FEED_BUFFER, NEWS_CACHE
    now = time.time()
    if NEWS_CACHE and (now - NEWS_CACHE["timestamp"] < CACHE_TTL):
        print("[News Cache] Serving cached news results (Python)...", file=sys.stdout)
        gateways = list(set(a.get("apiGateway", "Cached") for a in NEWS_CACHE["data"]))
        return jsonify({
            "success": True,
            "count": len(NEWS_CACHE["data"]),
            "data": NEWS_CACHE["data"],
            "gateways": gateways
        })

    gnews_key = os.getenv("GNEWS_KEY")
    newsapi_key = os.getenv("NEWSAPI_KEY") or os.getenv("NEWS_API_KEY")

    query = "disaster OR flood OR earthquake OR landslide OR cyclone India"
    # For NewsAPI, expand query with Hindi/Marathi equivalents to fetch multilingual results
    expanded_news_query = "disaster OR flood OR earthquake OR landslide OR cyclone OR बाढ़ OR भूकंप OR भूस्खलन OR पूर OR वादळ India"
    encoded_query = urllib.parse.quote(expanded_news_query)

    # For GDELT, use proper Boolean logic. NOTE: GDELT does not natively parse or support 
    # non-Latin script characters (Devanagari) in URL queries, which causes HTTP 400/403/429.
    # GDELT interprets spaces as AND. We use OR logic to find disaster coverage in India.
    gdelt_query = "India (disaster OR flood OR landslide OR earthquake OR cyclone)"
    encoded_gdelt = urllib.parse.quote(gdelt_query)

    # SSL context that works across networks (fixes GDELT handshake timeouts)
    _ssl_ctx = ssl.create_default_context()
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode = ssl.CERT_NONE

    def fetch_once(url, timeout=12):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; CrisisBot/2.0)"})
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx) as resp:
            content_bytes = resp.read()
            content_str = content_bytes.decode("utf-8")
            try:
                return json.loads(content_str)
            except Exception as e:
                if "limit requests" in content_str or "throttle" in content_str or "Too Many Requests" in content_str:
                    raise Exception("GDELT_RATE_LIMIT")
                raise e

    all_articles = []
    seen_urls    = set()

    # --- Gateway 1: GNews (if key available) ---
    # Query en/hi/mr explicitly. Previously hardcoded to lang=en only, which
    # meant the Hindi/Marathi filters always came up empty - not because the
    # filter was broken, but because no non-English content was ever fetched.
    if gnews_key:
        for gnews_lang in ("en", "hi", "mr"):
            try:
                lang_query = query
                api_lang = gnews_lang
                if gnews_lang == "hi":
                    lang_query = "आपदा OR बाढ़ OR भूकंप OR भूस्खलन OR चक्रवात OR तूफान India"
                elif gnews_lang == "mr":
                    lang_query = "आपत्ती OR पूर OR भूकंप OR दरड OR वादळ India"
                    # GNews does not officially support 'mr' language parameter.
                    # We query using 'hi' (Devanagari script index) so GNews can search Devanagari Marathi text.
                    api_lang = "hi"
                encoded_lang_query = urllib.parse.quote(lang_query)
                gnews_url = f"https://gnews.io/api/v4/search?q={encoded_lang_query}&lang={api_lang}&country=in&max=6&apikey={gnews_key}"
                print(f"Requesting news from GNews gateway (lang={gnews_lang})...", file=sys.stdout)
                # Respect GNews free-tier rate limit (1 req/sec) — pause between language queries
                if gnews_lang != "en":
                    time.sleep(2)
                data = fetch_once(gnews_url)
                if "articles" in data:
                    for i, art in enumerate(data["articles"]):
                        url = art.get("url", "#")
                        if not url or url in seen_urls:
                            continue
                        text = f"{art.get('title','')}. {art.get('description','')}. {art.get('content','')}"
                        if not is_disaster_related(art.get("title",""), text):
                            continue
                        seen_urls.add(url)
                        all_articles.append({
                            "id": f"gnews-{gnews_lang}-{i}-{int(time.time())}",
                            "source": "News",
                            "title": art.get("title", "Live Ingested Disaster Brief"),
                            "headline": art.get("title", "Live Ingested"),
                            "timestamp": art.get("publishedAt", datetime.datetime.now().isoformat()),
                            "rawText": text,
                            "url": url,
                            "sourceName": art.get("source", {}).get("name", "GNews"),
                            "apiGateway": "GNews"
                        })
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "Too Many Requests" in err_str:
                    print(f"⚠️ GNews rate limit hit (lang={gnews_lang}) — waiting 5s before next gateway.", file=sys.stdout)
                    time.sleep(5)
                else:
                    print(f"⚠️ GNews gateway (lang={gnews_lang}) unavailable: {e}. Continuing.", file=sys.stdout)

    # --- Gateway 2: NewsAPI (if key available) ---
    if newsapi_key:
        try:
            newsapi_url = f"https://newsapi.org/v2/everything?q={encoded_query}&sortBy=publishedAt&pageSize=8&apiKey={newsapi_key}"
            print(f"Requesting news from NewsAPI gateway...", file=sys.stdout)
            data = fetch_once(newsapi_url)
            if "articles" in data:
                for i, art in enumerate(data["articles"]):
                    url = art.get("url", "#")
                    if not url or url in seen_urls:
                        continue
                    text = f"{art.get('title','')}. {art.get('description','')}. {art.get('content','')}"
                    if not is_disaster_related(art.get("title",""), text):
                        continue
                    seen_urls.add(url)
                    all_articles.append({
                        "id": f"newsapi-{i}-{int(time.time())}",
                        "source": "News",
                        "title": art.get("title", "Disaster Alert Report"),
                        "headline": art.get("title", "No Headline Available"),
                        "timestamp": art.get("publishedAt", datetime.datetime.now().isoformat()),
                        "rawText": text,
                        "url": url,
                        "sourceName": art.get("source", {}).get("name", "NewsAPI"),
                        "apiGateway": "NewsAPI"
                    })
        except Exception as e:
            print(f"⚠️ NewsAPI gateway unavailable: {e}. Continuing with other sources.", file=sys.stdout)

    # --- Gateway 3: GDELT 2.0 (no API key required) ---
    try:
        gdelt_url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={encoded_gdelt}&mode=artlist&format=json&maxrecords=8"
        print(f"Requesting news from GDELT 2.0 gateway...", file=sys.stdout)
        data = fetch_once(gdelt_url, timeout=20)
        if data and "articles" in data:
            for i, art in enumerate(data["articles"]):
                url = art.get("url", "#")
                if not url or url in seen_urls:
                    continue
                rawText = f"{art.get('title','')}. Domain: {art.get('domain','')}. Language: {art.get('language','')}. Source country: {art.get('sourcecountry','')}"
                if not is_disaster_related(art.get("title",""), rawText):
                    continue
                seen_urls.add(url)
                seendate = art.get("seendate", "")
                timestamp = datetime.datetime.now().isoformat()
                try:
                    clean_date = "".join([c for c in seendate if c.isdigit()])
                    if len(clean_date) >= 14:
                        y, m, d, h, mn, s = clean_date[0:4], clean_date[4:6], clean_date[6:8], clean_date[8:10], clean_date[10:12], clean_date[12:14]
                        timestamp = datetime.datetime(int(y), int(m), int(d), int(h), int(mn), int(s)).isoformat()
                except Exception:
                    pass
                all_articles.append({
                    "id": f"gdelt-{i}-{int(time.time())}",
                    "source": "News",
                    "title": art.get("title", "GDELT Incident Dispatch"),
                    "headline": art.get("title", "GDELT News Flash"),
                    "timestamp": timestamp,
                    "rawText": rawText,
                    "url": url,
                    "sourceName": art.get("domain", "GDELT 2.0"),
                    "apiGateway": "GDELT"
                })
    except Exception as e:
        print(f"⚠️ GDELT gateway unavailable: {e}.", file=sys.stdout)

    if not all_articles:
        return jsonify({
            "success": True,
            "count": 0,
            "data": [],
            "note": "No active live disaster-related news articles found."
        })

    # Run SOTA extraction on all ingested articles
    analyzed_articles = []
    for art in all_articles:
        parsed = extract_disaster_intel(art["rawText"])
        analyzed_articles.append({
            **parsed,
            "id": art["id"],
            "source": "News",
            "title": art["title"],
            "headline": art["headline"],
            "rawText": art["rawText"],
            "timestamp": art["timestamp"],
            "author": art["sourceName"],
            "apiGateway": art.get("apiGateway", "Unknown")
        })

    NEWS_FEED_BUFFER = analyzed_articles
    NEWS_CACHE = {
        "timestamp": time.time(),
        "data": analyzed_articles
    }
    gateways_used = list(set(a.get("apiGateway", "") for a in analyzed_articles))
    return jsonify({"success": True, "count": len(NEWS_FEED_BUFFER), "data": NEWS_FEED_BUFFER,
                    "gateways": gateways_used})


@app.route("/api/fetch-social", methods=["GET"])
def fetch_social():
    global SOCIAL_FEED_BUFFER, SOCIAL_CACHE
    now = time.time()
    if SOCIAL_CACHE and (now - SOCIAL_CACHE["timestamp"] < CACHE_TTL):
        print("[Social Cache] Serving cached social posts (Python)...", file=sys.stdout)
        return jsonify({"success": True, "count": len(SOCIAL_CACHE["data"]), "data": SOCIAL_CACHE["data"]})

    api_key = os.getenv("MASTODON_API_KEY") or os.getenv("SOCIAL_MEDIA_API_KEY")
    gateway = request.args.get("gateway", "mastodon")

    try:
        raw_posts = []
        seen_texts = set()

        # Twitter API requires valid Bearer Token secrets
        if gateway == "twitter" and os.getenv("SOCIAL_MEDIA_API_KEY"):
            query = "(disaster OR flood OR earthquake OR landslide OR NDRF) India"
            encoded_query = urllib.parse.quote(query)
            api_url = f"https://api.twitter.com/2/tweets/search/recent?query={encoded_query}&max_results=12&tweet.fields=created_at,author_id,text"

            headers = {
                "Authorization": f"Bearer {os.getenv('SOCIAL_MEDIA_API_KEY')}",
                "Content-Type": "application/json"
            }
            req = urllib.request.Request(api_url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as response:
                data = json.loads(response.read().decode('utf-8'))

            if "data" in data:
                for tweet in data["data"]:
                    text = tweet.get("text", "")
                    if not text or text in seen_texts:
                        continue
                    if not is_disaster_related("", text):
                        continue
                    seen_texts.add(text)
                    raw_posts.append({
                        "id": f"tweet-{tweet['id']}",
                        "author": f"@User_{tweet['author_id']}",
                        "timestamp": tweet.get("created_at", datetime.datetime.now().isoformat()),
                        "rawText": text
                    })
        else:
            # Fetch public disaster-related social media posts from Mastodon
            q = request.args.get("q") or request.args.get("query") or "disaster OR flood OR earthquake OR landslide OR NDRF OR evacuation"
            import re
            terms = [t.strip() for t in re.split(r'\s+or\s+', q, flags=re.IGNORECASE) if t.strip()]

            statuses = []
            instances_to_try = ["mstdn.social", "mastodon.online", "fosstodon.org", "mastodon.world", "mastodon.social"]

            def fetch_url_json(url):
                try:
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                    if api_key:
                        headers["Authorization"] = f"Bearer {api_key}"
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=6) as response:
                        return json.loads(response.read().decode('utf-8'))
                except Exception as ex:
                    print(f"Failed to fetch from {url}: {ex}", file=sys.stderr)
                    return None

            for instance in instances_to_try:
                if statuses:
                    break
                print(f"Attempting to fetch live social data from Mastodon instance: {instance}", file=sys.stdout)
                try:
                    if len(terms) > 1:
                        target_terms = terms[:3]
                        for term in target_terms:
                            clean_term = term.replace('#', '')
                            tag_url = f"https://{instance}/api/v1/timelines/tag/{urllib.parse.quote(clean_term)}?limit=6"
                            tag_res = fetch_url_json(tag_url)
                            if isinstance(tag_res, list):
                                statuses.extend(tag_res)
                    else:
                        search_url = f"https://{instance}/api/v2/search?q={urllib.parse.quote(q)}&type=statuses&limit=12"
                        search_res = fetch_url_json(search_url)
                        if isinstance(search_res, dict) and "statuses" in search_res and search_res["statuses"]:
                            statuses.extend(search_res["statuses"])
                        else:
                            clean_term = q.replace('#', '')
                            tag_url = f"https://{instance}/api/v1/timelines/tag/{urllib.parse.quote(clean_term)}?limit=12"
                            tag_res = fetch_url_json(tag_url)
                            if isinstance(tag_res, list):
                                statuses.extend(tag_res)

                    if not statuses:
                        disaster_res = fetch_url_json(f"https://{instance}/api/v1/timelines/tag/disaster?limit=12")
                        if isinstance(disaster_res, list):
                            statuses.extend(disaster_res)
                except Exception as instance_ex:
                    print(f"⚠️ Error on Mastodon instance {instance}: {instance_ex}", file=sys.stderr)

            for post in statuses:
                content_html = post.get("content", "")
                import re
                clean_text = re.sub('<[^<]+?>', '', content_html)
                clean_text = clean_text.replace("&quot;", '"').replace("&amp;", "&").strip()

                if not clean_text or clean_text in seen_texts:
                    continue
                if not is_disaster_related("", clean_text):
                    continue
                seen_texts.add(clean_text)

                raw_posts.append({
                    "id": f"mastodon-{post.get('id')}",
                    "author": f"@{post.get('account', {}).get('username', 'Anonymous')}",
                    "timestamp": post.get("created_at", datetime.datetime.now().isoformat()),
                    "rawText": clean_text
                })

        if not raw_posts:
            raise ValueError("No matching social mentions found on public feeds.")

        # Run extraction
        analyzed_posts = []
        for post in raw_posts:
            parsed = extract_disaster_intel(post["rawText"])
            analyzed_posts.append({
                **parsed,
                "id": post["id"],
                "source": "Social",
                "author": post["author"],
                "timestamp": post["timestamp"],
                "rawText": post["rawText"],
                "title": f"Social Intelligence Report from {post['author']}",
                "headline": "Emergency Citizen Toot"
            })

        SOCIAL_FEED_BUFFER = analyzed_posts
        SOCIAL_CACHE = {
            "timestamp": time.time(),
            "data": analyzed_posts
        }
        return jsonify({"success": True, "count": len(SOCIAL_FEED_BUFFER), "data": SOCIAL_FEED_BUFFER})

    except Exception as e:
        print(f"Error fetching social feeds: {e}", file=sys.stderr)
        SOCIAL_FEED_BUFFER = []
        return jsonify({
            "success": True, 
            "count": 0, 
            "data": [],
            "note": f"Live social feeds unavailable ({str(e)}). Configure MASTODON_API_KEY for live data."
        })


@app.route("/api/clear-caches", methods=["POST"])
def clear_caches_endpoint():
    global NEWS_CACHE, SOCIAL_CACHE, NEWS_FEED_BUFFER, SOCIAL_FEED_BUFFER
    NEWS_CACHE = None
    SOCIAL_CACHE = None
    NEWS_FEED_BUFFER = []
    SOCIAL_FEED_BUFFER = []
    print("[Caches] Cleared backend live feed caches and buffers in Python", file=sys.stdout)
    return jsonify({"success": True, "message": "Server caches and buffers cleared."})


@app.route("/api/analyze", methods=["POST"])
def analyze_text():
    req_data = request.json or {}
    text = req_data.get("text", "")
    model_name = req_data.get("modelName", "mt5")

    if not text.strip():
        return jsonify({"success": False, "error": "Input text cannot be empty."}), 400

    try:
        parsed = extract_disaster_intel(text, model_name=model_name)
        return jsonify({"success": True, "data": parsed})
    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"Error in analyze_text: {err_msg}", file=sys.stderr)
        return jsonify({"success": False, "error": str(e), "traceback": err_msg}), 500


@app.route("/api/analyze-document", methods=["POST"])
def analyze_document():
    """
    Phase 6 — Document Analysis: accepts TXT, PDF, DOCX file uploads.
    Extracts text, runs the same 5-task NLP pipeline.
    No mock data, no templates, no Gemini.
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded."}), 400
    f = request.files["file"]
    fname = f.filename.lower() if f.filename else ""
    text = ""

    try:
        if fname.endswith(".txt"):
            text = f.read().decode("utf-8", errors="replace")
        elif fname.endswith(".pdf"):
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(io.BytesIO(f.read()))
                text = " ".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                return jsonify({"success": False, "error": "PyPDF2 not installed. Run: pip install PyPDF2"}), 500
        elif fname.endswith(".docx"):
            try:
                import docx
                doc = docx.Document(io.BytesIO(f.read()))
                text = " ".join(p.text for p in doc.paragraphs)
            except ImportError:
                return jsonify({"success": False, "error": "python-docx not installed. Run: pip install python-docx"}), 500
        else:
            return jsonify({"success": False, "error": "Unsupported file type. Use TXT, PDF, or DOCX."}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"File read error: {e}"}), 500

    if not text.strip():
        return jsonify({"success": False, "error": "Document contains no extractable text."}), 400

    parsed = extract_disaster_intel(text)
    return jsonify({"success": True, "data": parsed, "extractedLength": len(text)})


# ─── NEW: Research comparison endpoint ────────────────────────────────────────

@app.route("/api/analyze-compare", methods=["POST"])
def analyze_compare():
    """
    Run ALL 5 TASKS with ALL 3 MODELS (mt5, indicbart, mbart) independently.
    Returns a 3×5 comparison matrix for the research table in the dashboard.
    This satisfies the project requirement: "Develop minimum 3 recent NLP models
    and compare the accuracy in all 5 tasks."
    """
    req_data = request.json or {}
    text = req_data.get("text", "").strip()
    if not text:
        return jsonify({"success": False, "error": "Input text cannot be empty."}), 400
    try:
        result = _run_comparison(text)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        print(f"[analyze-compare] {e}", file=sys.stderr)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/evaluation-metrics", methods=["GET"])
def get_evaluation_metrics():
    """
    Return stored evaluation metrics from evaluation_results/metrics.json.
    Run nlp/evaluate_all.py to populate this file.
    Returns empty dict if evaluation has not been run yet.
    """
    try:
        metrics = load_eval_metrics()
        return jsonify({
            "success": True,
            "data": metrics,
            "has_results": bool(metrics),
            "models": list(MODEL_REGISTRY.keys()),
            "tasks": TASK_NAMES,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/run-evaluation", methods=["POST"])
def run_evaluation_endpoint():
    """
    Trigger evaluation asynchronously in a background thread.
    Returns immediately with a job-started message.
    Results are written to evaluation_results/metrics.json (read via /api/evaluation-metrics).
    """
    req_data = request.json or {}
    models   = req_data.get("models", None)  # None => all
    tasks    = req_data.get("tasks",  None)  # None => all
    n        = int(req_data.get("n", 200))

    if not _NLP_OK:
        return jsonify({
            "success": False,
            "error": "NLP pipeline not available. Install dependencies: pip install -r requirements_nlp.txt"
        }), 503

    def _run_bg():
        try:
            from nlp.evaluate_all import run_all
            run_all(models=models, tasks=tasks, n=n)
            print("[Eval] Background evaluation complete.", flush=True)
        except Exception as e:
            print(f"[Eval] Background evaluation failed: {e}", file=sys.stderr)

    import threading
    t = threading.Thread(target=_run_bg, daemon=True)
    t.start()
    return jsonify({
        "success": True,
        "message": "Evaluation started in background. Poll /api/evaluation-metrics for results.",
        "n": n,
        "models": models or list(MODEL_REGISTRY.keys()),
        "tasks": tasks or TASK_NAMES,
    })


@app.route("/api/run-training", methods=["POST"])
def run_training_endpoint():
    req_data = request.json or {}
    epochs = int(req_data.get("epochs", 1))
    batch_size = int(req_data.get("batch_size", 1))
    max_train = int(req_data.get("max_train", 3))
    max_val = int(req_data.get("max_val", 2))

    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(base_dir, "training.log")
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now()}] Starting Training Job...\n")

    def _run_bg():
        import traceback
        try:
            import subprocess
            cmd = [
                sys.executable,
                os.path.join(base_dir, "train_and_evaluate.py"),
                "--epochs", str(epochs),
                "--batch_size", str(batch_size),
                "--max_train", str(max_train),
                "--max_val", str(max_val)
            ]
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"Executing: {' '.join(cmd)}\n")
                lf.flush()
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8"
                )
                for line in process.stdout:
                    lf.write(line)
                    lf.flush()
                process.wait()
                if process.returncode == 0:
                    lf.write("\n[Training SUCCESS]\n")
                else:
                    lf.write(f"\n[Training FAILED with exit code {process.returncode}]\n")
        except Exception as e:
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"\n[Fatal Error]: {e}\n{traceback.format_exc()}\n")

    import threading
    t = threading.Thread(target=_run_bg, daemon=True)
    t.start()
    return jsonify({
        "success": True,
        "message": "Training started in background.",
        "log_path": log_file
    })


@app.route("/api/training-logs", methods=["GET"])
def get_training_logs():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(base_dir, "training.log")
    if not os.path.exists(log_file):
        return jsonify({"success": True, "logs": "No active training logs."})
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            logs = f.read()
        finished = "[Training SUCCESS]" in logs or "[Training FAILED" in logs or "[Fatal Error]" in logs
        success = "[Training SUCCESS]" in logs
        return jsonify({
            "success": True,
            "logs": logs,
            "finished": finished,
            "training_success": success
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500




@app.route("/")
def index():
    # Detect configured secret keys to inform the user through UI badges
    news_configured = bool(os.getenv("NEWS_API_KEY") or os.getenv("GNEWS_KEY") or os.getenv("NEWSAPI_KEY"))
    social_configured = bool(os.getenv("SOCIAL_MEDIA_API_KEY") or os.getenv("MASTODON_API_KEY") or os.getenv("SOCIAL_API_KEY") or os.getenv("TWITTER_API_KEY"))

    return render_template(
        "index.html",
        news_configured=news_configured,
        social_configured=social_configured,
        static_disasters=STATIC_DISASTER_REPORTS
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
