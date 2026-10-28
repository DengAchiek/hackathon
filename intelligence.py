"""Evidence-based opportunity summaries. Suggestions are explicit business rules."""
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

SOURCE_NAMES = {
    "whatsapp": "WhatsApp", "website_search": "Website search", "salesperson": "Sales notes",
    "salesperson_voice": "Voice note", "chatbot": "Chatbot", "rejected_quote": "Unaccepted quote",
    "unavailable_product": "Unavailable product", "out_of_stock": "Stockout", "abandoned_enquiry": "Abandoned enquiry",
}
REASON_NAMES = {
    "out_of_stock": "Out of stock", "price_too_high": "Price concern", "not_stocked": "Product not offered",
    "quote_rejected": "Quote not accepted", "lead_time": "Delivery too slow", "abandoned": "Enquiry abandoned",
}
STATUSES = {"open": "Open", "follow_up": "Following up", "recovered": "Recovered", "closed": "Closed"}
CURRENCIES = ("KES", "USD", "EUR", "GBP", "UGX", "TZS", "UNKNOWN")
ACTIONS = {
    "out_of_stock": ("Restock or offer a substitute", "Check supplier availability and confirm a restock date with interested customers."),
    "not_stocked": ("Validate a new product opportunity", "Confirm specifications with interested customers before adding a product to the range."),
    "price_too_high": ("Review the price or bundle", "Compare stated budgets with your offer; test an alternative or bundle without assuming a discount is profitable."),
    "quote_rejected": ("Review and follow up on the quote", "Ask what prevented acceptance and check whether a revised offer would help."),
    "lead_time": ("Offer a faster delivery option", "Check a closer supplier or collection option, then confirm the customer's deadline."),
    "abandoned": ("Follow up on unfinished enquiries", "Prepare a relevant follow-up for customers who agreed to be contacted."),
}


def product_key(value):
    # Preserve dimensions, brands, sizes and model numbers; no automatic stemming.
    return " ".join(re.findall(r"[\w]+", value.casefold(), re.UNICODE))


def detect_currency(text, default="KES"):
    patterns = {"KES": r"\b(?:kes|ksh|kshs)\b", "USD": r"\busd\b|\$", "EUR": r"\beur\b|€",
                "GBP": r"\bgbp\b|£", "UGX": r"\bugx\b", "TZS": r"\btzs\b"}
    found = [code for code, pattern in patterns.items() if re.search(pattern, text, re.I)]
    return found[0] if len(found) == 1 else "UNKNOWN" if len(found) > 1 else default


def money_totals(rows, amount="budget", currency="currency"):
    totals = defaultdict(float)
    for row in rows:
        value = row.get(amount)
        if value is not None:
            totals[row.get(currency) or "UNKNOWN"] += value
    return {code: round(value, 2) for code, value in sorted(totals.items())}


def parse_timestamp(value):
    result = datetime.fromisoformat(value)
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def enrich_signal(row):
    row = dict(row)
    # Older databases used a stemmed/sorted key; group by the current product
    # name without rewriting original records just to read the dashboard.
    row["pkey"] = product_key(row["product"])
    row["effective_reason"] = row.get("reviewed_reason") or row["reason"]
    row["needs_review"] = not bool(row.get("reviewed_reason"))
    row["status"] = row.get("status") or "open"
    return row


def follow_up_queue(rows, *, today, status="active", assignment="any", owner="",
                    urgency="all", due="all", query="", page=1):
    """A business work queue, independent of dashboard filters or ML status."""
    if status not in ("active", "all", *STATUSES):
        raise ValueError("Choose a supported follow-up status.")
    if assignment not in ("any", "assigned", "unassigned"):
        raise ValueError("Choose a supported owner filter.")
    if urgency not in ("all", "urgent"):
        raise ValueError("Choose all requests or urgent requests.")
    if due not in ("all", "overdue", "today", "unscheduled"):
        raise ValueError("Choose a supported follow-up date filter.")
    try:
        page = max(1, int(page))
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid page number.") from exc
    day = today.isoformat()
    active = [row for row in rows if row["status"] in ("open", "follow_up")]
    recovered = [row for row in rows if row["status"] == "recovered"]
    is_overdue = lambda row: row["status"] in ("open", "follow_up") and bool(row.get("follow_up_on")) and row["follow_up_on"] < day
    summary = dict(active=len(active), unassigned=sum(not row.get("owner") for row in active),
                   following_up=sum(row["status"] == "follow_up" for row in active),
                   overdue=sum(is_overdue(row) for row in active), recovered=len(recovered),
                   recovered_value=money_totals(recovered, "recovered_amount", "recovered_currency"),
                   needs_review=sum(row["needs_review"] for row in rows))
    owners = {}
    for row in rows:
        if row.get("owner"):
            owners.setdefault(row["owner"].casefold(), row["owner"])
    found = []
    for original in rows:
        row = dict(original)
        if status == "active" and row["status"] not in ("open", "follow_up"):
            continue
        if status not in ("active", "all") and row["status"] != status:
            continue
        if assignment == "unassigned" and row.get("owner") or assignment == "assigned" and not row.get("owner"):
            continue
        if owner and (row.get("owner") or "").casefold() != owner.casefold():
            continue
        if urgency == "urgent" and row["urgency"] != 3:
            continue
        if due == "overdue" and not is_overdue(row):
            continue
        if due == "today" and (row.get("follow_up_on") != day or row["status"] not in ("open", "follow_up")):
            continue
        if due == "unscheduled" and row.get("follow_up_on"):
            continue
        haystack = " ".join(str(row.get(key) or "") for key in ("product", "raw", "customer", "owner", "next_action"))
        if query and query.casefold() not in haystack.casefold():
            continue
        row["overdue"] = is_overdue(row)
        row["due_today"] = row.get("follow_up_on") == day and row["status"] in ("open", "follow_up")
        row["suggested_action"] = ACTIONS.get(row["effective_reason"], ("Review the request", ""))[0]
        found.append(row)
    found.sort(key=lambda row: (not row["overdue"], -row["urgency"], row.get("follow_up_on") or "9999-12-31", row["ts"], row["id"]))
    pages = max(1, (len(found) + 11) // 12)
    page = min(page, pages)
    return dict(items=found[(page-1)*12:page*12], total=len(found), page=page, pages=pages,
                summary=summary, owners=sorted(owners.values(), key=str.casefold), today=day)


def summarize(rows, now=None):
    now = now or datetime.now(timezone.utc)
    groups = {}
    for row in rows:
        key = row["pkey"] or product_key(row["product"])
        group = groups.setdefault(key, {"key": key, "product": row["product"], "rows": []})
        group["rows"].append(row)
    demand = []
    for group in groups.values():
        evidence = group.pop("rows")
        active = [r for r in evidence if r["status"] in ("open", "follow_up")]
        reason_evidence = active or evidence
        reasons = Counter(r["effective_reason"] for r in reason_evidence)
        sources = Counter(r["source"] for r in evidence)
        known = {r["customer"].strip().casefold() for r in evidence if r.get("customer")}
        urgent = sum(r["urgency"] == 3 for r in active)
        # Comparable score: no currency arithmetic, inferred buyers or forecast.
        score = len(active) * 2 + urgent * 3 + len(known) + len(sources)
        top_reason = reasons.most_common(1)[0][0]
        action, explanation = ACTIONS[top_reason]
        if top_reason == "not_stocked" and "website_search" in sources:
            action = "Check search coverage and product range"
            explanation = ("Replay the failed query and check filters and product aliases. "
                           "Confirm whether the product is actually missing before adding it to the range.")
        recent = sum(parse_timestamp(r["ts"]) >= now - timedelta(days=7) for r in evidence)
        previous = sum(now - timedelta(days=14) <= parse_timestamp(r["ts"]) < now - timedelta(days=7) for r in evidence)
        group.update(count=len(evidence), open_requests=len(active), customers=len(known),
                     unidentified_requests=sum(not bool(r.get("customer")) for r in evidence),
                     qty=sum(r["qty"] for r in evidence), budgets=money_totals(active),
                     value=sum(r.get("budget") or 0 for r in evidence) if len(money_totals(evidence)) <= 1 else None,
                     sources=dict(sources), reasons=dict(reasons), top_reason=top_reason, urgent=urgent,
                     score=score, priority="High" if score >= 16 or urgent >= 2 else "Medium" if score >= 8 else "Early signal",
                     suggested_action=action, action_detail=explanation,
                     action_basis=f"{reasons[top_reason]} of {len(reason_evidence)} {'open ' if active else ''}requests: {REASON_NAMES[top_reason].lower()}.",
                     recent_count=recent, previous_count=previous,
                     trend=f"{recent} in last 7 days · {previous} in prior 7 days",
                     review_count=sum(r["needs_review"] for r in evidence),
                     last=max(r["ts"] for r in evidence), signal_ids=[r["id"] for r in evidence],
                     state="active" if active else "resolved")
        demand.append(group)
    demand.sort(key=lambda x: (-x["score"], -x["count"], x["product"]))
    active_rows = [r for r in rows if r["status"] in ("open", "follow_up")]
    recovered = [r for r in rows if r["status"] == "recovered"]
    return dict(
        kpis=dict(signals=len(rows), products=len(demand), active_opportunities=sum(g["state"] == "active" for g in demand),
                  needs_review=sum(r["needs_review"] for r in rows), urgent=sum(r["urgency"] == 3 for r in active_rows),
                  recovered=len(recovered), recovered_value=money_totals(recovered, "recovered_amount", "recovered_currency"),
                  stated_budgets=money_totals(active_rows),
                  stated_value=sum(r.get("budget") or 0 for r in rows) if len(money_totals(rows)) <= 1 else None),
        demand=demand, recent=rows[:12],
        reasons=dict(Counter(r["effective_reason"] for r in active_rows)),
        sources=dict(Counter(r["source"] for r in rows)),
        note="Stated budgets are expressions of interest, not lost revenue. Recovery values are recorded by your team.",
    )
