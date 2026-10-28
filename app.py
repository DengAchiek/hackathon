"""GhostDemand AI turns lost sales into structured demand signals."""
import json
import csv
import hashlib
import io
import math
import os
import re
import secrets
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import click
from flask import Flask, Response, g, has_request_context, jsonify, render_template, request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.exceptions import HTTPException

from demand_ml import REASON_LABELS, ReasonClassifier, TrainingError
from intelligence import CURRENCIES, REASON_NAMES, SOURCE_NAMES, STATUSES, detect_currency, enrich_signal, follow_up_queue, product_key, summarize

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(ROOT))
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
app.secret_key = os.environ.get("GHOST_SECRET_KEY") or secrets.token_hex(32)
preview_signer = URLSafeTimedSerializer(app.secret_key, salt="ghost-preview-v1")
DB = os.environ.get("GHOST_DB", str(ROOT / "ghostdemand.db"))
DEMO_DB = os.environ.get("GHOST_DEMO_DB", str(ROOT / "ghostdemand-demo.db"))
DEFAULT_CURRENCY = os.environ.get("GHOST_CURRENCY", "KES")
BUSINESS_TIMEZONE = ZoneInfo(os.environ.get("GHOST_TIMEZONE", "Africa/Nairobi"))
if DEFAULT_CURRENCY not in CURRENCIES:
    raise ValueError("GHOST_CURRENCY must be one of " + ", ".join(CURRENCIES))
ML = ReasonClassifier(
    os.environ.get("GHOST_ML_PATH", str(ROOT / "models" / "reason_classifier.json")),
    enabled=os.environ.get("GHOST_ML_ENABLED", "1").lower() not in ("0", "false", "no"),
    threshold=float(os.environ.get("GHOST_ML_THRESHOLD", "0.55")),
)
DEMO_ML = ReasonClassifier(ROOT / "models" / "demo_reason_classifier.json")
SOURCES = list(SOURCE_NAMES)


def is_demo():
    return has_request_context() and request.headers.get("X-Ghost-Demo") == "1"


def active_model():
    return DEMO_ML if is_demo() else ML


# ---------- storage (additive migration preserves existing signals) ----------
def db():
    if "db" not in g:
        g.db = sqlite3.connect(DEMO_DB if is_demo() else DB, timeout=15)
        g.db.row_factory = sqlite3.Row
        g.db.execute("""CREATE TABLE IF NOT EXISTS signals(
            id INTEGER PRIMARY KEY, ts TEXT, source TEXT, raw TEXT, product TEXT, pkey TEXT,
            qty INTEGER, budget REAL, reason TEXT, urgency INTEGER, sentiment TEXT,
            location TEXT, customer TEXT)""")
        additions = {
            "reason_engine": "TEXT NOT NULL DEFAULT 'rules'", "ml_reason": "TEXT",
            "ml_confidence": "REAL", "model_version": "TEXT", "reviewed_reason": "TEXT",
            "reviewed_at": "TEXT",
            "currency": "TEXT NOT NULL DEFAULT 'UNKNOWN'", "external_id": "TEXT",
            "status": "TEXT NOT NULL DEFAULT 'open'", "owner": "TEXT", "next_action": "TEXT",
            "recovered_amount": "REAL", "recovered_currency": "TEXT", "updated_at": "TEXT",
            "follow_up_on": "TEXT",
        }
        columns = {row[1] for row in g.db.execute("PRAGMA table_info(signals)")}
        if set(additions) - columns:
            # Recheck under a write lock when multiple requests start together.
            g.db.execute("BEGIN IMMEDIATE")
            columns = {row[1] for row in g.db.execute("PRAGMA table_info(signals)")}
            for name, sql_type in additions.items():
                if name not in columns:
                    g.db.execute(f"ALTER TABLE signals ADD COLUMN {name} {sql_type}")
            g.db.commit()
        g.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS signal_event ON signals(source,external_id) WHERE external_id IS NOT NULL")
        g.db.execute("""CREATE TABLE IF NOT EXISTS catalog(
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, sku TEXT, created_at TEXT NOT NULL)""")
        g.db.execute("""CREATE TABLE IF NOT EXISTS product_aliases(
            alias_key TEXT PRIMARY KEY, catalog_id INTEGER NOT NULL REFERENCES catalog(id))""")
    return g.db


@app.teardown_appcontext
def close(_):
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


# ---------- free text -> structured demand signal ----------
REASONS = {
    "out_of_stock": r"out of stock|sold out|no stock|finished|unavailable|restock|not in stock",
    "price_too_high": r"too (expensive|high|much)|cheaper|discount|pricey|over budget|lower price|best price",
    "not_stocked": r"don'?t (have|sell|stock)|not available|do you (have|sell|stock)|looking for|any .* available",
    "quote_rejected": r"declined|rejected|went (with|elsewhere)|competitor|chose another|lost the (deal|quote)",
    "lead_time": r"too long|delivery time|lead time|when (will|can)|wait",
    "abandoned": r"abandon|never replied|went quiet|left the (cart|chat)|no response|ghosted",
}
URGENT = r"urgent|asap|today|immediately|tomorrow|right now|emergency"
NEG = r"disappoint|annoy|angry|frustrat|useless|never again|terrible|waste"
POS = r"thanks|great|please|love|interested"


def parse_money(value):
    value = value.replace(",", "").lower()
    return float(value[:-1]) * 1000 if value.endswith("k") else float(value)


def normalize(product):
    return product_key(product)


def rule_extract(text, source):
    text = text.strip()
    low = text.lower()
    reason = next((r for r, pattern in REASONS.items() if r != "not_stocked" and re.search(pattern, low)), None)
    if not reason and re.search(REASONS["not_stocked"], low):
        reason = "not_stocked"
    reason = reason or {"out_of_stock": "out_of_stock", "rejected_quote": "quote_rejected",
                        "abandoned_enquiry": "abandoned", "website_search": "not_stocked"}.get(source, "not_stocked")
    qty = re.search(r"(\d{1,9})\s*(?:pcs|pieces|units|bags|boxes|cartons|kg|litres|liters|tons?|x)\b", low) \
        or re.search(r"\b(\d{1,5})\s+(?=[a-z])", low)
    budget = re.search(r"(?:kshs?|kes|usd|eur|gbp|ugx|tzs|\$|€|£|budget(?: of)?|up to)\s*([\d,]+(?:\.\d+)?k?)", low)
    loc = re.search(r"(?:deliver(?:ed)? to|in|to|at)\s+([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)?)", text)
    prod = re.search(r"(?:looking for|do you (?:have|sell|stock)|need|want|searching for|searched for|"
                     r"requested|asked for|enquir\w+ (?:about|for)|out of stock[: ]+|quote for)\s+(.+?)"
                     r"(?:\?|\.|,|;| for | in | to | by | at | but | because | urgent| asap|$)", text, re.I)
    product = prod.group(1) if prod else (text if len(text.split()) <= 6 else " ".join(text.split()[:6]))
    product = re.sub(r"^\d+(?:\s*(?:pcs|pieces|units|bags|boxes|cartons|kg|litres|liters|tons?|x)\b\s*|\s+)(?:of\s+)?",
                     "", product.strip(), flags=re.I).strip(" -:") or text[:80]
    sentiment = "negative" if re.search(NEG, low) else "positive" if re.search(POS, low) else "neutral"
    urgency = 3 if re.search(URGENT, low) else 2 if reason in ("out_of_stock", "quote_rejected") else 1
    money = parse_money(budget.group(1)) if budget else None
    return dict(product=product.title(), qty=max(1, int(qty.group(1))) if qty else 1,
                budget=money if money is None or math.isfinite(money) else None, reason=reason,
                urgency=urgency, sentiment=sentiment, location=loc.group(1) if loc else None)


def llm_extract(text, source):
    """Optional richer field extraction when ANTHROPIC_API_KEY is set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        message = anthropic.Anthropic(timeout=20, max_retries=0).messages.create(
            model=os.environ.get("GHOST_MODEL", "claude-sonnet-5-5"), max_tokens=300,
            system="Extract a lost-sale demand signal. Reply with JSON only: product (str), qty (int), "
                   "budget (number|null), reason (out_of_stock|price_too_high|not_stocked|quote_rejected|"
                   "lead_time|abandoned), urgency (1-3), sentiment (negative|neutral|positive), location (str|null).",
            messages=[{"role": "user", "content": f"Source: {source}\nText: {text}"}])
        payload = json.loads(re.sub(r"```json|```", "", message.content[0].text).strip())
        return payload if isinstance(payload, dict) else None
    except Exception:
        app.logger.warning("Claude extraction unavailable; using local extraction.")
        return None


def validated_fields(candidate):
    """External model output cannot replace trusted fields with invalid values."""
    clean = {}
    for key in ("product", "location"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            clean[key] = value.strip()[:250]
    for key, allowed in (("reason", REASON_LABELS), ("sentiment", ("negative", "neutral", "positive"))):
        if candidate.get(key) in allowed:
            clean[key] = candidate[key]
    if type(candidate.get("qty")) is int and 1 <= candidate["qty"] <= 1_000_000_000:
        clean["qty"] = candidate["qty"]
    if type(candidate.get("urgency")) is int and candidate["urgency"] in (1, 2, 3):
        clean["urgency"] = candidate["urgency"]
    budget = candidate.get("budget")
    if type(budget) in (float, int) and 0 <= budget <= 1e15 and math.isfinite(budget):
        clean["budget"] = budget
    return clean


def extract(text, source):
    base = rule_extract(text, source)
    richer = validated_fields((llm_extract(text, source) if not is_demo() else None) or {})
    base.update(richer)
    base["reason_engine"] = "claude" if "reason" in richer else "rules"
    prediction = active_model().predict(text)
    if prediction and prediction["used"]:
        base["reason"] = prediction["reason"]
        base["reason_engine"] = "local_ml"
    base["ml_reason"] = prediction["reason"] if prediction else None
    base["ml_confidence"] = prediction["confidence"] if prediction else None
    base["model_version"] = prediction["model_version"] if prediction else None
    return base


def prepare_signal(item):
    if not isinstance(item, dict):
        raise ValueError("Each signal must be a JSON object.")
    text = item.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > 10000:
        raise ValueError("'text' must contain between 1 and 10,000 characters.")
    source = item.get("source", "chatbot")
    if source not in SOURCES:
        raise ValueError("'source' must be a supported channel.")
    customer = item.get("customer")
    if customer is not None and (not isinstance(customer, str) or len(customer) > 250):
        raise ValueError("'customer' must be a string of up to 250 characters.")
    return text.strip(), source, customer


def input_options(payload):
    currency = payload.get("currency", DEFAULT_CURRENCY)
    if currency not in CURRENCIES:
        raise ValueError("Choose a supported currency.")
    external_id = payload.get("external_id")
    if external_id is not None and (not isinstance(external_id, str) or len(external_id) > 200):
        raise ValueError("'external_id' must be a string of up to 200 characters.")
    external_id = external_id or None
    return currency, external_id


def catalog_match(text, product):
    aliases = db().execute("SELECT a.alias_key,c.id,c.name FROM product_aliases a JOIN catalog c ON c.id=a.catalog_id").fetchall()
    text_key = " " + product_key(text) + " "
    product_normalized = product_key(product)
    matches = {r["id"]: r["name"] for r in aliases
               if r["alias_key"] == product_normalized or " " + r["alias_key"] + " " in text_key}
    warnings = []
    if len(matches) > 1:
        warnings.append("This message mentions multiple catalog products. Capture each product separately.")
        return product, warnings
    if len(matches) == 1:
        canonical = next(iter(matches.values()))
        variants = re.findall(r"(?:size\s*\w+|\b\d+\s*(?:gb|tb|mm|inch)\b)", text, re.I)
        if variants and any(product_key(v) not in product_key(canonical) for v in variants):
            warnings.append("A size or specification needs review. Keep different variants separate.")
            return product, warnings
        return canonical, warnings
    return product, warnings


def draft_signal(text, source, currency):
    signal = extract(text, source)
    signal["product"], warnings = catalog_match(text, signal["product"])
    signal["currency"] = detect_currency(text, currency)
    if source == "website_search":
        warnings.append("A failed search can reflect naming or filters. Check actual product availability before treating it as a range gap.")
    if signal["currency"] == "UNKNOWN":
        warnings.append("The budget currency is unclear. Confirm the currency before saving.")
    if signal["reason_engine"] == "rules":
        warnings.append("Check the product and blocker against the original customer request.")
    elif signal.get("ml_confidence") is not None and signal["reason_engine"] != "local_ml":
        warnings.append("The blocker needs confirmation. Please verify it before saving.")
    return signal, warnings


def apply_fields(signal, fields):
    if not isinstance(fields, dict):
        raise ValueError("'fields' must be a JSON object.")
    allowed = {"product", "qty", "budget", "reason", "urgency", "sentiment", "location", "currency"}
    if set(fields) - allowed:
        raise ValueError("Unsupported editable signal field.")
    cleaned = validated_fields(fields)
    if "currency" in fields and fields["currency"] in CURRENCIES:
        cleaned["currency"] = fields["currency"]
    for key, value in fields.items():
        if value is None and key in ("budget", "location"):
            cleaned[key] = None
        elif key not in cleaned:
            raise ValueError(f"Invalid value for '{key}'.")
    if "reason" in cleaned and cleaned["reason"] != signal["reason"]:
        signal["reason_engine"] = "human"
    signal.update(cleaned)
    return signal


def build_preview(payload):
    text, source, customer = prepare_signal(payload)
    currency, external_id = input_options(payload)
    signal, warnings = draft_signal(text, source, currency)
    identity = dict(text=text, source=source, customer=customer, currency=currency, external_id=external_id)
    token = preview_signer.dumps(dict(identity=identity, signal=signal))
    return dict(**identity, fields=signal, warnings=warnings, preview_token=token)


def save_signal(text, source="chatbot", customer=None, *, commit=True, currency=None,
                external_id=None, fields=None, verified=False, preview_token=None, event=False):
    connection = db()
    if external_id:
        previous = connection.execute("SELECT * FROM signals WHERE source=? AND external_id=?", (source, external_id)).fetchone()
        if previous:
            return dict(previous, deduplicated=True)
    currency = currency or DEFAULT_CURRENCY
    if preview_token:
        if not isinstance(preview_token, str):
            raise ValueError("Preview the request again before saving.")
        try:
            preview = preview_signer.loads(preview_token, max_age=3600)
        except (BadSignature, SignatureExpired) as exc:
            raise ValueError("The preview expired. Preview this request again before saving.") from exc
        identity = dict(text=text, source=source, customer=customer, currency=currency, external_id=external_id)
        if preview["identity"] != identity:
            raise ValueError("The request changed. Preview it again before saving.")
        signal = dict(preview["signal"])
    else:
        signal, _ = draft_signal(text, source, currency)
    if fields is not None:
        apply_fields(signal, fields)
    if event:
        signal["reason_engine"] = "channel_event"
    if type(verified) is not bool:
        raise ValueError("'verified' must be true or false.")
    now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    signal["reviewed_reason"] = signal["reason"] if verified else None
    signal["reviewed_at"] = now if verified else None
    columns = ("product", "qty", "budget", "reason", "urgency", "sentiment", "location",
               "reason_engine", "ml_reason", "ml_confidence", "model_version", "currency",
               "reviewed_reason", "reviewed_at")
    cursor = connection.execute(
        "INSERT OR IGNORE INTO signals(ts,source,raw,pkey,customer,external_id," + ",".join(columns) + ")"
        " VALUES(" + ",".join("?" for _ in range(6 + len(columns))) + ")",
        (now, source, text, normalize(signal["product"]), customer, external_id, *(signal[k] for k in columns)))
    if commit:
        connection.commit()
    if cursor.rowcount == 0:
        row = connection.execute("SELECT * FROM signals WHERE source=? AND external_id=?", (source, external_id)).fetchone()
        return dict(row, deduplicated=True)
    return {"id": cursor.lastrowid, "source": source, "raw": text, "deduplicated": False, **signal}


def save_payload(payload, commit=True):
    text, source, customer = prepare_signal(payload)
    currency, external_id = input_options(payload)
    return save_signal(text, source, customer, commit=commit, currency=currency, external_id=external_id,
                       fields=payload.get("fields"), verified=payload.get("verified", False),
                       preview_token=payload.get("preview_token"))


def reviewed_examples():
    return [dict(text=row["raw"], reason=row["reviewed_reason"], reviewed_at=row["reviewed_at"])
            for row in db().execute("SELECT raw, reviewed_reason, reviewed_at FROM signals "
                                    "WHERE reviewed_reason IS NOT NULL ORDER BY reviewed_at, id")]


def model_status():
    status = active_model().status()
    status["unreviewed_signals"] = db().execute(
        "SELECT count(*) FROM signals WHERE reviewed_reason IS NULL").fetchone()[0]
    status["reviewed_signals"] = db().execute(
        "SELECT count(*) FROM signals WHERE reviewed_reason IS NOT NULL").fetchone()[0]
    status["pending_reviews"] = db().execute(
        "SELECT count(*) FROM signals WHERE reviewed_at > ?",
        ((status["metadata"] or {}).get("reviewed_through", ""),)).fetchone()[0]
    return status


# ---------- routes ----------
@app.errorhandler(ValueError)
def invalid_input(error):
    return jsonify(error=str(error)), 400


@app.errorhandler(HTTPException)
def http_error(error):
    if request.path.startswith(("/api/", "/webhook/")):
        return jsonify(error=error.description), error.code
    return error


@app.get("/")
def landing():
    return render_template("landing.html")


@app.get("/dashboard")
@app.get("/demo")
def index():
    return render_template("index.html", demo=request.path == "/demo", config=dict(
        sources=SOURCE_NAMES, reasons=REASON_NAMES, statuses=STATUSES, currencies=CURRENCIES,
        defaultCurrency=DEFAULT_CURRENCY, demo=request.path == "/demo"))


@app.post("/api/signals")
def add_signal():
    return jsonify(save_payload(request.get_json())), 201


@app.post("/api/signals/bulk")
def add_bulk():
    payload = request.get_json()
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not 1 <= len(items) <= 100:
        raise ValueError("'items' must be a list containing 1 to 100 signals.")
    for item in items:
        prepare_signal(item)
        input_options(item)
    connection = db()
    with connection:
        signals = [save_payload(item, commit=False) for item in items]
    return jsonify(signals), 201


@app.post("/webhook/whatsapp")
def whatsapp():
    body = request.values.get("Body", "").strip()
    if body:
        save_signal(*prepare_signal(dict(text=body, source="whatsapp", customer=request.values.get("From"))),
                    external_id=request.values.get("MessageSid") or request.values.get("SmsMessageSid"))
    return "", 204


@app.get("/api/model")
def get_model():
    return jsonify(model_status())


@app.post("/api/model/train")
def train_model():
    examples = reviewed_examples()
    try:
        active_model().train(examples)
    except TrainingError as error:
        return jsonify(error=str(error)), 409
    except OSError:
        app.logger.exception("Unable to read training data or save the trained model")
        return jsonify(error="Unable to read training data or save the model. Check server file permissions."), 503
    return jsonify(model_status())


@app.patch("/api/signals/<int:signal_id>/review")
def review_signal(signal_id):
    payload = request.get_json()
    reason = payload.get("reason") if isinstance(payload, dict) else None
    if reason not in REASON_LABELS:
        raise ValueError("Choose a supported reason label.")
    cursor = db().execute(
        "UPDATE signals SET reviewed_reason=?, reviewed_at=? WHERE id=?",
        (reason, datetime.now(timezone.utc).isoformat(timespec="microseconds"), signal_id))
    if cursor.rowcount == 0:
        return jsonify(error="Signal not found."), 404
    db().commit()
    return jsonify(id=signal_id, reviewed_reason=reason)


def selected_rows():
    clauses, params = [], []
    for name, column, allowed in (("source", "source", SOURCES), ("reason", "COALESCE(reviewed_reason,reason)", REASON_LABELS),
                                  ("status", "status", STATUSES)):
        value = request.args.get(name)
        if value:
            if value not in allowed:
                raise ValueError(f"Unsupported {name} filter.")
            clauses.append(f"{column}=?")
            params.append(value)
    days = request.args.get("days", "all")
    if days != "all":
        if days not in ("7", "30", "90"):
            raise ValueError("Choose a 7, 30, 90 day or all-time view.")
        clauses.append("ts>=?")
        params.append((datetime.now(timezone.utc) - timedelta(days=int(days))).isoformat())
    query = request.args.get("q", "").strip()[:150]
    if query:
        clauses.append("(product LIKE ? OR raw LIKE ? OR customer LIKE ?)")
        params.extend([f"%{query}%"] * 3)
    if request.args.get("review") == "needed":
        clauses.append("reviewed_reason IS NULL")
    sql = "SELECT * FROM signals" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY ts DESC, id DESC"
    return [enrich_signal(row) for row in db().execute(sql, params)]


@app.get("/api/insights")
def insights():
    return jsonify(summarize(selected_rows()))


@app.get("/api/signals")
def list_signals():
    rows = selected_rows()
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError as exc:
        raise ValueError("Invalid page number.") from exc
    page_size = 15
    return jsonify(items=rows[(page-1)*page_size:page*page_size], total=len(rows), page=page,
                   pages=max(1, math.ceil(len(rows)/page_size)))


@app.get("/api/operations/follow-ups")
def get_follow_ups():
    rows = [enrich_signal(row) for row in db().execute("SELECT * FROM signals")]
    return jsonify(follow_up_queue(
        rows, today=datetime.now(BUSINESS_TIMEZONE).date(),
        status=request.args.get("status", "active"), assignment=request.args.get("assignment", "any"),
        owner=request.args.get("owner", "")[:100], urgency=request.args.get("urgency", "all"),
        due=request.args.get("due", "all"), query=request.args.get("q", "").strip()[:150],
        page=request.args.get("page", "1")))


@app.get("/api/opportunities/evidence")
def opportunity_evidence():
    rows = [row for row in selected_rows() if row["pkey"] == request.args.get("key")]
    if not rows:
        return jsonify(error="This opportunity has no evidence in the current view."), 404
    summary = summarize(rows)["demand"][0]
    return jsonify(product=summary["product"], items=rows, summary=summary)


@app.post("/api/signals/preview")
def preview_signal():
    return jsonify(build_preview(request.get_json()))


@app.patch("/api/signals/<int:signal_id>")
def update_signal(signal_id):
    payload = request.get_json()
    if not isinstance(payload, dict):
        raise ValueError("Use a JSON object to update a signal.")
    current = db().execute("SELECT * FROM signals WHERE id=?", (signal_id,)).fetchone()
    if current is None:
        return jsonify(error="Signal not found."), 404
    values = dict(current)
    editable = {k: v for k, v in payload.items() if k in ("product", "qty", "budget", "urgency", "location")}
    apply_fields(values, editable)
    if "reason" in payload:
        if payload["reason"] not in REASON_LABELS:
            raise ValueError("Choose a supported reason label.")
        values["reviewed_reason"] = payload["reason"]
        values["reviewed_at"] = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    for name, limit in (("owner", 100), ("next_action", 600)):
        if name in payload:
            if not isinstance(payload[name], str) or len(payload[name]) > limit:
                raise ValueError(f"'{name}' must be text of up to {limit} characters.")
            values[name] = payload[name].strip() or None
    if "status" in payload:
        if not isinstance(payload["status"], str) or payload["status"] not in STATUSES:
            raise ValueError("Choose a supported follow-up status.")
        values["status"] = payload["status"]
    if "follow_up_on" in payload:
        follow_up_on = payload["follow_up_on"]
        if follow_up_on not in (None, ""):
            try:
                if not isinstance(follow_up_on, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", follow_up_on):
                    raise ValueError()
                date.fromisoformat(follow_up_on)
            except ValueError as exc:
                raise ValueError("Choose a valid follow-up date (YYYY-MM-DD).") from exc
        values["follow_up_on"] = follow_up_on or None
    for name in ("currency", "recovered_currency"):
        if name in payload:
            if payload[name] not in CURRENCIES:
                raise ValueError("Choose a supported currency.")
            values[name] = payload[name]
    if "recovered_amount" in payload:
        amount = payload["recovered_amount"]
        if amount is not None and (type(amount) not in (int, float) or not math.isfinite(amount) or not 0 <= amount <= 1e15):
            raise ValueError("Recovered value must be a non-negative amount.")
        values["recovered_amount"] = amount
    if values["status"] == "recovered" and values.get("recovered_amount") is not None and not values.get("recovered_currency"):
        raise ValueError("Choose the currency for the recovered value.")
    values["pkey"] = normalize(values["product"])
    values["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    keys = ("product", "pkey", "qty", "budget", "urgency", "location", "currency", "reviewed_reason", "reviewed_at",
            "status", "owner", "next_action", "follow_up_on", "recovered_amount", "recovered_currency", "updated_at")
    db().execute("UPDATE signals SET " + ",".join(f"{key}=?" for key in keys) + " WHERE id=?", (*(values[k] for k in keys), signal_id))
    db().commit()
    return jsonify(enrich_signal(values))


@app.post("/api/import/preview")
def preview_import():
    upload = request.files.get("file")
    if upload is None:
        raise ValueError("Choose a CSV file with a 'text' column.")
    try:
        content = upload.read().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(content), strict=True)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("The CSV needs a 'text' column. Optional: source, customer, currency, external_id.")
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError("Use a valid UTF-8 CSV file.") from exc
    if not 1 <= len(rows) <= 100:
        raise ValueError("Import 1 to 100 requests at a time.")
    drafts, errors = [], []
    fingerprint = hashlib.sha256(content.encode()).hexdigest()[:24]
    for index, row in enumerate(rows, 2):
        item = {key: value for key, value in row.items() if key in ("text", "source", "customer", "currency", "external_id") and value}
        item.setdefault("external_id", f"csv-{fingerprint}-{index}")
        try:
            drafts.append(build_preview(item))
        except ValueError as exc:
            errors.append(dict(row=index, error=str(exc)))
    return jsonify(drafts=drafts, errors=errors, total=len(rows))


@app.get("/api/signals/export")
def export_signals():
    fields = ("id", "ts", "source", "product", "qty", "budget", "currency", "effective_reason", "customer", "status", "owner", "recovered_amount", "recovered_currency", "raw")
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(fields)
    for row in selected_rows():
        writer.writerow([("'" + row.get(k)) if isinstance(row.get(k), str) and row[k].startswith(("=", "+", "-", "@")) else row.get(k) for k in fields])
    return Response(stream.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=ghostdemand-signals.csv"})


@app.get("/api/catalog")
def get_catalog():
    rows = [dict(row) for row in db().execute("SELECT * FROM catalog ORDER BY name")]
    for row in rows:
        row["aliases"] = [r[0] for r in db().execute("SELECT alias_key FROM product_aliases WHERE catalog_id=? ORDER BY alias_key", (row["id"],))]
    return jsonify(items=rows)


@app.post("/api/catalog")
def add_catalog():
    payload = request.get_json()
    if not isinstance(payload, dict):
        raise ValueError("Use a JSON object for a product.")
    name, aliases, sku = payload.get("name"), payload.get("aliases", []), payload.get("sku", "")
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 200 or not product_key(name):
        raise ValueError("Enter a product name of 1 to 200 characters.")
    if not isinstance(aliases, list) or len(aliases) > 20 or any(not isinstance(a, str) or not a.strip() or len(a) > 200 for a in aliases):
        raise ValueError("Use up to 20 non-empty product aliases.")
    if not isinstance(sku, str) or len(sku) > 80:
        raise ValueError("SKU must contain up to 80 characters.")
    keys = {product_key(a) for a in [name, *aliases]}
    if "" in keys:
        raise ValueError("Aliases must include letters or numbers.")
    connection = db()
    try:
        with connection:
            cursor = connection.execute("INSERT INTO catalog(name,sku,created_at) VALUES(?,?,?)", (name.strip(), sku.strip(), datetime.now(timezone.utc).isoformat()))
            connection.executemany("INSERT INTO product_aliases(alias_key,catalog_id) VALUES(?,?)", [(key, cursor.lastrowid) for key in keys])
            for row in connection.execute("SELECT id,product FROM signals").fetchall():
                if product_key(row["product"]) in keys:
                    connection.execute("UPDATE signals SET product=?,pkey=? WHERE id=?", (name.strip(), normalize(name), row["id"]))
    except sqlite3.IntegrityError as exc:
        raise ValueError("A name or alias already belongs to another catalog product. Keep each alias unique.") from exc
    return jsonify(id=cursor.lastrowid, name=name.strip(), aliases=sorted(keys)), 201


@app.post("/api/events/website-search")
def website_event():
    payload = request.get_json()
    if not isinstance(payload, dict) or type(payload.get("results_count")) is not int or payload["results_count"] < 0:
        raise ValueError("Provide a non-negative integer 'results_count'.")
    if payload["results_count"] != 0:
        return jsonify(captured=False, message="Search returned results; no failed-search signal captured.")
    item = dict(text=payload.get("query"), source="website_search", customer=payload.get("customer"),
                external_id=payload.get("event_id"), currency=payload.get("currency", DEFAULT_CURRENCY))
    result = save_payload(item)
    return jsonify(captured=True, signal=result), 201


@app.post("/api/events/stockout")
def stockout_event():
    payload = request.get_json()
    if not isinstance(payload, dict) or not isinstance(payload.get("product"), str):
        raise ValueError("Provide a product name.")
    fields = dict(product=payload["product"], qty=payload.get("qty", 1), reason="out_of_stock")
    apply_fields(dict(reason="out_of_stock"), fields)
    item = dict(text=f"Out of stock: {fields['qty']} units {fields['product']}", source="out_of_stock",
                customer=payload.get("customer"), external_id=payload.get("event_id"), currency=payload.get("currency", DEFAULT_CURRENCY))
    text, source, customer = prepare_signal(item)
    currency, external_id = input_options(item)
    return jsonify(save_signal(text, source, customer, currency=currency, external_id=external_id, fields=fields, event=True)), 201


@app.post("/api/demo/seed")
def seed_demo():
    if not is_demo():
        raise ValueError("Demo data can only be loaded in the demo workspace.")
    from demo_data import seed
    if not active_model().status()["ready"]:
        active_model().train([])
    return jsonify(seed(db(), save_signal))


@app.cli.command("train-model")
def train_model_command():
    """Train on synthetic starter examples and explicitly reviewed signals."""
    try:
        metadata = active_model().train(reviewed_examples())
    except (TrainingError, OSError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=int(os.environ.get("PORT", "5000")))
