# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Python cell notebook: PesaShield AI Roadmap
#
# This is ordinary Python, arranged into `# %%` notebook cells.
# Open it in VS Code with the Python and Jupyter extensions and select
# **Run Cell** or **Run All**, or run `python PesaShield_AI_Roadmap.py`.
# Install `requirements.txt` first. See `README.md` for the full setup.
#
# Source: `PesaShield_AI/PesaShield_AI_Roadmap.ipynb`. Original explanations and model
# logic are retained. Outputs go to a separate `runs/PesaShield_AI_Roadmap/`
# folder. The scripts use practice data and never deploy a live model.

# %%
from pathlib import Path
import os
from IPython import get_ipython
from IPython.display import display

# In VS Code, open this file and use Run Cell or Run All. In Jupyter, first
# change to the folder containing the Python notebooks. Python scripts also work.
NOTEBOOK_DIR = (Path(__file__).resolve().parent if "__file__" in globals()
                else globals().get("NOTEBOOK_DIR", Path.cwd().resolve()))
DATA_DIR = NOTEBOOK_DIR / "data"
RUN_DIR = NOTEBOOK_DIR / "runs" / "PesaShield_AI_Roadmap"
RUN_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(RUN_DIR)
os.environ.setdefault("MPLCONFIGDIR", str(RUN_DIR / ".matplotlib"))

IN_NOTEBOOK = get_ipython() is not None
import matplotlib
if IN_NOTEBOOK:
    get_ipython().run_line_magic("matplotlib", "inline")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

_plot_number = 0
def show_plot():
    """Display charts interactively, or save them when run with python."""
    global _plot_number
    if IN_NOTEBOOK:
        plt.show()
        return
    for number in plt.get_fignums():
        _plot_number += 1
        path = RUN_DIR / f"figure_{_plot_number:02d}.png"
        plt.figure(number).savefig(path, bbox_inches="tight", dpi=140)
        print("Saved chart:", path)
    plt.close("all")

print("Learning outputs:", RUN_DIR)


# %% [markdown]
# # 🛡️ PesaShield AI | Improved Colab Roadmap and Working MVP Starter
# ### Detect suspicious M-Pesa-style SMS patterns — **not** confirm payments
# **Version:** 2 (reviewed and improved)  
# **Purpose:** beginner-friendly, reproducible Kenyan hackathon demonstration using synthetic data.
#
# **What changed:** corrected README statistics, duplicate checks, safe grouped evaluation, error analysis, clearer limitations, persistent SQLite, functional FastAPI example, and local API tests. The baseline remains TF-IDF + Logistic Regression. **No real payment provider verification is performed.**

# %% [markdown]
# ## 📍 Contents
# 1. Problem, scope and success criteria
# 2. Architecture & stakeholder workflow
# 3. Data collection, governance and evaluation design
# 4. **Run in Colab:** dataset loading, EDA, split, model, evaluation, inference
# 5. Optional lightweight SQLite demonstration
# 6. API contract and backend implementation plan
# 7. Frontend/dashboard pages and design
# 8. Seven-day hackathon build plan, responsibilities, tests, risks
# 9. GitHub contributor workflow and deliverable checklist
# 10. Live demo and 3-minute pitch
#
# Each numbered stage distinguishes **what is planned** from **what notebook cells actually execute**.

# %% [markdown]
# ## Phase 1 — Problem definition
# **Problem:** Kenyan customers and merchants can be targeted with forged M-Pesa payment confirmations, phishing links, PIN/OTP requests, payment-release demands, reversal pressure, and advance-fee requests. People can lose money or release goods without a confirmed payment.
#
# **Primary user:** Merchant or mobile money user checking suspicious SMS.  
# **Secondary user:** Analyst reviewing flags and recording outcomes.
#
# **Core objective:** Predict *suspicious-text likelihood*, explain warning patterns, offer a human review workflow, and clearly distinguish **message risk** from **payment verification**.
#
# **MVP must-have**
# - Paste SMS → text classifier → risk result, explanation and verification state (`not_checked` by default)
# - Extract some non-sensitive cues (amounts, URLs, reference-like tokens) with regex/rules
# - Store minimum necessary review metadata; demo dashboard with investigation queue
# - Demonstrate optional *simulated merchant ledger* for matching an authorized record
#
# **Out of scope for hackathon:** direct M-Pesa account access; real payment blocking; claims that a genuine-looking SMS means the transaction succeeded; production financial decisions solely from this model.

# %% [markdown]
# ### ▶️ SQLite demo (persists between notebook reruns)
#
# Creates `pesashield_artifacts/pesashield_demo.db`. Records now survive a kernel restart if the directory remains available. **Colab runtime storage is ephemeral:** download the database for long-term retention. The demo intentionally avoids storing raw SMS contents. Never treat this as an approved production datastore.

# %% [markdown]
# ## Phase 2 — Architecture and user journey
# ```text
# User / Merchant
#      │ paste SMS or import demo example
#      ▼
# Frontend (React + Tailwind, planned)
#      │ POST /predict
#      ▼
# FastAPI (planned) ─────► SQLite (MVP) / PostgreSQL (later)
#      │                         │
#      ▼                         ▼
# NLP cleaning + rules      assessment metadata + analyst feedback
#      │
#      ▼
# TF-IDF + Logistic Regression (train below)
#      │
#      ├─► suspicious-text score + warning cues
#      └─► verification_status = not_checked
#                  │
#                  └─► authorized ledger match (optional demo-only)
# ```
#
# **GhostDemand pattern adapted:** Capture → Analyze → Review → Act. Achiek's retail signals become fraud signals; his follow-up queue becomes a fraud investigation queue. Keep the two projects in separate directories unless the team explicitly chooses an integration.

# %% [markdown]
# ## Phase 3 — Data plan and limitations
# **Provided dataset:** `pesashield_synthetic_mpesa_sms.csv` contains 1,200 **synthetic** messages, 750 labeled as suspicious text (`1`) and 450 as normal-looking (`0`). **This does not authenticate any payment.**
#
# **Columns:** `message_id`, `message_text`, `label`, `category`, `template_group`, `data_origin`, `payment_verified`.
#
# **Eight categories** (150 each): `phishing_link`, `credential_request`, `fake_reversal_pressure`, `advance_fee`, `payment_claim_pressure`, `ordinary_account_notice`, `ordinary_payment_notice`, `ordinary_transfer_notice`.
#
# **Labels:** `1` suspicious-text pattern; `0` no modeled suspicious pattern. This is not a true/false claim about an underlying M-Pesa payment.
#
# **Additional data after MVP:** UCI SMS Spam Collection (general SMS spam, different labeling task), permitted real-world Kenyan examples with verified consent and labels, and PaySim only for a *separate* transaction-risk component. Never mislabel PaySim as fake SMS data.
#
# **Key risk:** template variants are highly correlated; a random per-row split inflates performance. We hold out entire `template_group` families. Even this does not measure transfer to real SMS, so an independent real-world, consented test set is essential.

# %% [markdown]
# ## ▶️ Colab Step A — Imports and load dataset
# Upload the CSV when the cell asks, or place it in the Colab working directory. This notebook doesn't need Google Drive access.

# %%
import os, re, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

CSV_NAME = str(DATA_DIR / "pesashield_synthetic_mpesa_sms.csv")
if not Path(CSV_NAME).exists():
    candidates = sorted(Path('.').glob('pesashield_synthetic_mpesa_sms*.csv'))
    if candidates:
        CSV_NAME = str(candidates[0])
    else:
        try:
            from google.colab import files
            print('Upload the PesaShield synthetic CSV:')
            uploaded = files.upload()
            candidates = [p for p in uploaded if p.lower().endswith('.csv')]
            if candidates: CSV_NAME = candidates[0]
        except ImportError:
            print('Place the CSV in your working folder.')
assert Path(CSV_NAME).exists(), f"CSV not found: {CSV_NAME}"
df = pd.read_csv(CSV_NAME)
print('Loaded:', df.shape, 'from', CSV_NAME)
display(df.head(5))

# %%
required = {"message_id", "message_text", "label", "category", "template_group", "data_origin", "payment_verified"}
assert required.issubset(df.columns), f"Missing columns: {required-set(df.columns)}"
assert df["label"].isin([0,1]).all(), "Unexpected label"
print("Missing values: \n", df.isna().sum())
print("Duplicate IDs:", df.message_id.duplicated().sum())
print("Label distribution: \n", df.label.value_counts().sort_index())
print("Template groups:", df.template_group.nunique())
print("Origins:", df.data_origin.unique())
print("Verification statuses:", df.payment_verified.unique())

# %% [markdown]
# ## ▶️ Colab Step B — Exploratory data analysis
# Ask: Are categories balanced? Are messages too repetitive? Which suspicious patterns dominate? Visualize counts without exposing real customers.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
df["label"].map({0:"No modeled red flags",1:"Suspicious pattern"}).value_counts().plot(kind="bar", ax=axes[0], title="Text-pattern labels")
df["category"].value_counts().plot(kind="barh", ax=axes[1], title="Synthetic message categories")
axes[0].set_ylabel("Messages")
axes[1].set_xlabel("Messages")
plt.tight_layout(); show_plot()

df["message_length"] = df.message_text.astype(str).str.len()
display(df.groupby("label").agg(messages=("message_id","count"),mean_length=("message_length","mean")).round(1))

# %%
print('Exact duplicate texts:', df.message_text.duplicated().sum())
print('Distinct messages:', df.message_text.nunique())
print('Template groups per class:')
display(df.groupby('label')['template_group'].nunique())
print('Label counts:', df.label.value_counts().sort_index().to_dict())
print('Duplicate text with conflicting labels:', df.groupby('message_text').label.nunique().gt(1).sum())
# Warning: grouped split protects against same-template leakage, not synthetic shortcuts.
# Keep original records for reporting; deduplicate training only, after splitting.

# %% [markdown]
# ## ▶️ Step C — Group-held-out split + training-only deduplication
#
# Hold out whole `template_group`s so training cannot see the same template family as testing. **Never deduplicate or adjust the test set based on model performance.** After splitting, drop exact duplicates **in training only** to reduce repetition. The held-out test set retains its original rows, so its reported sample count remains explicit. Even this split cannot validate real-world fraud detection.

# %%
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             average_precision_score, precision_recall_curve, precision_score,
                             recall_score, f1_score)

X = df['message_text'].astype(str)
y = df['label'].astype(int)
groups = df['template_group'].astype(str)
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))
train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()
assert set(train_df.template_group).isdisjoint(set(test_df.template_group))
assert set(train_df.label) == {0, 1} and set(test_df.label) == {0, 1}
assert not set(train_df.message_text).intersection(set(test_df.message_text)), 'Text leaked across groups'
train_unique = train_df.drop_duplicates(subset=['message_text']).copy()
X_train = train_unique['message_text'].astype(str)
y_train = train_unique['label'].astype(int)
X_test = test_df['message_text'].astype(str)
y_test = test_df['label'].astype(int)
print(f'Train: {len(train_df)} original, {len(train_unique)} unique, {train_df.template_group.nunique()} templates')
print(f'Test: {len(test_df)} rows, {test_df.template_group.nunique()} unseen templates')
print('Train unique labels:', y_train.value_counts().to_dict())
print('Test labels:', y_test.value_counts().to_dict())

# %% [markdown]
# ## ▶️ Colab Step D — Train a baseline classifier
# Start with TF-IDF + Logistic Regression because it is fast, explainable at the feature-weight level and easy to deploy. The pipeline includes all text transformations inside the training fold (avoiding vocabulary leakage).

# %%
model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2, max_features=15000)),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
])
model.fit(X_train, y_train)
print("Baseline model trained.")

# %% [markdown]
# ## ▶️ Colab Step E — Evaluate model carefully
# Inspect **precision** (how many flagged messages were suspicious in our labels), **recall** (how many suspicious-labeled messages were found), **F1**, **confusion matrix**, and **PR-AUC**. False positives can disrupt legitimate merchants; false negatives can miss scams. Report the internal test performance without treating it as real-world performance.

# %%
pred = model.predict(X_test)
score = model.predict_proba(X_test)[:,1]
print(classification_report(y_test, pred, target_names=['no modeled red flag', 'suspicious pattern'], zero_division=0))
print('Average precision (synthetic only):', round(average_precision_score(y_test, score), 3))
print('Suspicious precision:', round(precision_score(y_test, pred, zero_division=0), 3))
print('Suspicious recall:', round(recall_score(y_test, pred, zero_division=0), 3))
print('False positives:', int(((y_test == 0) & (pred == 1)).sum()))
print('False negatives:', int(((y_test == 1) & (pred == 0)).sum()))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=['no flag', 'suspicious']).plot(cmap='Blues')
plt.title('Synthetic template-held-out test'); show_plot()
precision, recall, thresholds = precision_recall_curve(y_test, score)
plt.figure(figsize=(6,4)); plt.plot(recall, precision)
plt.xlabel('Recall');plt.ylabel('Precision');plt.title('Synthetic precision–recall curve');plt.grid(alpha=.2);show_plot()

# %% [markdown]
# ### How to interpret your results
# - Report **the scores printed by this updated notebook**, not figures from earlier versions.
# - A false positive is an ordinary-looking synthetic SMS incorrectly flagged as suspicious; a false negative is a suspicious synthetic SMS missed by the model.
# - High synthetic recall or average precision is **not real-world fraud accuracy**. Templates can make the task unnaturally easy.
# - **Do not tune the decision threshold on the held-out test set.** If needed, create a separate group-held-out validation set inside training and set the threshold there.
# - `prediction=no_modeled_red_flags` does **not** mean the payment is genuine or safe.

# %% [markdown]
# ## ▶️ Step E.1 — Inspect mistakes before claiming success
#
# Inspect both false positives and false negatives to learn which language patterns the classifier misreads. Avoid relying on the score alone. For privacy, these examples are synthetic.

# %%
errors = test_df[['message_text', 'label', 'category', 'template_group']].copy()
errors['predicted'] = pred
errors['score'] = np.round(score, 4)
false_positives = errors[(errors.label == 0) & (errors.predicted == 1)]
false_negatives = errors[(errors.label == 1) & (errors.predicted == 0)]
print('False positives:', len(false_positives), '| False negatives:', len(false_negatives))
display(false_positives.head(8))
display(false_negatives.head(8))
print('Error counts by category:')
display(errors[errors.label != errors.predicted].groupby(['category', 'label']).size().rename('mistakes').reset_index())

# %% [markdown]
# ### Dataset improvement plan (next iteration, not fabricated data)
# 1. Gather **consented, anonymized, independently labeled** SMS, following legal/privacy requirements.
# 2. Add varied natural phrasings, Kiswahili and code-switching, and difficult ordinary messages mentioning PIN/reversal/urgency.
# 3. Split by real campaign, date, or sender/user **when authorized metadata exists**; never claim this synthetic CSV has those fields.
# 4. Review category-specific errors, label quality and class prevalence. Compare against a simple keyword baseline.
# 5. Record false-positive rates on legitimate messages **before** any genuine deployment.
#
# **Important:** Removing repeated training text helps but cannot fix template-generation bias or provide proof of performance in the wild.

# %% [markdown]
# ## ▶️ Colab Step F — Message parsing, prediction and explanation
# A transparent demonstration function combines **ML score** with independently reported safety cues. These explanations describe matched rules, not necessarily causal reasons for the ML output. **No verified payment records are accessed.**

# %%
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
AMOUNT_RE = re.compile(r"(?:KES|KSh|Ksh)\s*([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE)

def extract_signals(message):
    text = str(message)
    lower = text.lower()
    signals = []
    if URL_RE.search(text): signals.append("Contains a web link: independently check any destination")
    if re.search(r"\b(pin|otp|password|passcode)\b", lower):
        signals.append("Mentions sensitive credentials")
    if re.search(r"\b(urgent|immediately|expire|suspend|release|reverse|reversal)\b", lower):
        signals.append("Uses urgency, release, or reversal language")
    amounts = AMOUNT_RE.findall(text)
    return {"amounts_mentioned": amounts, "signals": signals}

def assess_sms(message, threshold=0.5):
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Please enter a nonempty SMS message")
    p = float(model.predict_proba([message])[0,1])
    signals = extract_signals(message)
    return {
        "prediction": "suspicious_pattern" if p >= threshold else "no_modeled_red_flags",
        "synthetic_model_score": round(p, 4),
        "threshold": threshold,
        "warning_signals": signals["signals"],
        "amounts_mentioned": signals["amounts_mentioned"],
        "verification_status": "not_checked",
        "notice": "SMS content alone cannot verify payment receipt. Model trained on synthetic examples."
    }

samples = [
    "M-PESA ALERT: Your KSh 8500 payment will expire. Click https://check.example.invalid now.",
    "ABC123XYZ4 Confirmed. KSh 1500 received from JANE DOE."
]
for example in samples:
    print("MESSAGE:", example)
    print(json.dumps(assess_sms(example), indent=2));print("-"*60)

# %% [markdown]
# ## ▶️ Colab Step G — Save artifacts for your teammates
# The saved `joblib` pipeline can be loaded by FastAPI. **Only load artifacts created by your trusted team:** pickle/joblib can execute code during loading. The metadata explains data provenance and limitations.

# %%
import joblib
from datetime import datetime, timezone
from pathlib import Path
out = Path("pesashield_artifacts")
out.mkdir(exist_ok=True)
joblib.dump(model, out / "sms_baseline.joblib")
metadata = {
    "project": "PesaShield AI", "dataset": CSV_NAME, "data_origin": "synthetic",
    "target": "suspicious SMS text pattern", "verification_capability": "none",
    "training_rows": len(X_train), "train_original_rows": len(train_df), "test_rows": len(X_test),
    "split_method": "GroupShuffleSplit(template_group, test_size=.25, seed=42)",
    "test_average_precision": float(average_precision_score(y_test, score)),
    "created_at_utc": datetime.now(timezone.utc).isoformat()
}
(out / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
print("Saved:", list(out.iterdir()))

# %% [markdown]
# ## Phase 5 — Backend deliverable: Working prediction API + database
# ### What it means
# A **prediction API** accepts SMS text from the website, loads the approved model, returns an assessment, and persists selected metadata. **Database** means reviewable assessment history and analyst decisions.
#
# **Suggested stack:** FastAPI + Pydantic + SQLite locally; PostgreSQL later if needed.
#
# **API contract (planned)**
# - `GET /health` → service/model readiness
# - `POST /predict` → `{ "message": "..." }` → prediction, score, matched warning signals, `verification_status: "not_checked"`, assessment ID
# - `GET /assessments` → filtered records (authentication required for production)
# - `PATCH /assessments/{id}/review` → update analyst decision, optional supporting evidence
# - Optional `POST /verify-demo` → check an **explicitly simulated** merchant ledger; not M-Pesa
#
# **Security checklist:** input lengths, rate limiting, authentication, secrets in environment, sanitize logs, hash/avoid storing SMS bodies, opt-in and minimum retention, redaction of phone numbers, encrypt persisted data as appropriate, no arbitrary model artifact loading.

# %% [markdown]
# ### ▶️ SQLite demo (persists between notebook reruns)
#
# Creates `pesashield_artifacts/pesashield_demo.db`. Records now survive a kernel restart if the directory remains available. **Colab runtime storage is ephemeral:** download the database for long-term retention. The demo intentionally avoids storing raw SMS contents. Never treat this as an approved production datastore.

# %%
import sqlite3, uuid
from datetime import datetime, timezone
DB_PATH = str(Path('pesashield_artifacts') / 'pesashield_demo.db')
conn = sqlite3.connect(DB_PATH)
conn.execute('''CREATE TABLE IF NOT EXISTS assessments (
    id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
    score REAL NOT NULL, prediction TEXT NOT NULL,
    verification_status TEXT NOT NULL, review_status TEXT NOT NULL
)''')
conn.commit()

def save_assessment(result):
    row = (str(uuid.uuid4()), datetime.now(timezone.utc).isoformat(),
           result['synthetic_model_score'], result['prediction'],
           result['verification_status'], 'unreviewed')
    conn.execute('INSERT INTO assessments VALUES (?,?,?,?,?,?)', row)
    conn.commit()
    return row[0]

assessment_id = save_assessment(assess_sms(samples[0]))
print('Saved assessment ID:', assessment_id, '| persistent database:', DB_PATH)
display(pd.read_sql_query('SELECT * FROM assessments ORDER BY created_at DESC LIMIT 5', conn))

# %% [markdown]
# ### ▶️ Working FastAPI + SQLite starter
#
# The next cell **writes** `pesashield_artifacts/api/main.py`, then a following cell optionally checks it using FastAPI's local `TestClient` (no public server required). The service uses the saved model and a persistent SQLite file with separate connection per request. It provides:
#
# - `GET /health` — confirms the app/model load;
# - `POST /predict` — classifies the input and **saves an assessment**;
# - `GET /assessments` — returns recent assessments without raw SMS text.
#
# **Colab demo only:** there's no authentication, rate limiting, HTTPS gateway, production security review, real payment verification or real data validation. Don't expose it publicly or feed it private customer SMS.

# %% [markdown]
# ## Phase 6 — Frontend dashboard specification
# **Recommended:** React + Tailwind CSS. Dark navy, white, purple accents inspired by the professional GhostDemand-style interface, but not a direct copy.
#
# | Page | User action | Backend data |
# |---|---|---|
# | Overview | See message counts, risk distributions, open reviews | aggregated assessments |
# | Analyze SMS | Paste message, receive score and warning indicators | `POST /predict` |
# | Fraud signal inbox | Filter assessed messages | `GET /assessments` |
# | Investigations | Mark flagged cases reviewed, suspicious, false-positive, unresolved | `PATCH /assessments/{id}/review` |
# | Analytics | Plot category/risk trends, review outcomes | aggregate endpoints |
# | Verification demo | Compare claimed payment to a mocked authorized ledger | **demo-only** endpoint |
#
# **UX safety:** show three separate fields: **Text risk**, **Verification status**, **Analyst outcome**. Never label an SMS as 'payment confirmed' because the classifier scored it low risk.

# %% [markdown]
# ## Phase 7 — Testing, privacy and quality assurance
# **Required tests**
# 1. Empty/long Unicode SMS; Swahili and English mix; punctuation and casing; obfuscated links.
# 2. Negative tests: authentic-looking fake messages and scam-looking benign notifications.
# 3. Hold out unseen templates and report FP/FN cases.
# 4. Never return `payment_verified=true` without an authorized ledger match.
# 5. Verify database stores no unconsented secrets; scrub screenshots, phone numbers, names and transaction codes.
# 6. API request validation, authorization, error handling, CORS restrictions, rate limits, and logging controls.
# 7. Regression test: model load, `/health`, `/predict`, save/review workflow.
#
# **Risks:** synthetic-to-real mismatch, class balance shift, model calibration, adversarial phrasing, privacy, overconfidence. Mitigation: independent evaluation, warnings, human review, provider-approved verification and post-deployment monitoring.

# %% [markdown]
# ## Phase 8 — Seven-day build schedule (hackathon)
# | Day | Work | Owner suggestion | Deliverable |
# |---|---|---|---|
# | 1 | Define scope, categories, Github feature branch, documentation | Product + all | README, HACKATHON, requirements |
# | 2 | Prepare synthetic dataset, data dictionary, EDA | Data lead | CSV + EDA visuals |
# | 3 | Group-split modeling, baseline, error analysis | ML lead | Model + evaluation report |
# | 4 | API endpoints, SQLite and unit tests | Backend lead | `/predict` + assessments storage |
# | 5 | React dashboard, assessment form, inbox and charts | Frontend lead | Clickable UI |
# | 6 | Integrate, review statuses, security & demo ledger | Whole team | End-to-end scenario + test log |
# | 7 | Deploy and demo, 3-minute pitch, limitations | Whole team | Live MVP + repo + slides |
#
# **Team of four or more:** ML/data; backend/database; frontend/UI; integrations/testing/pitch. Everyone reviews and commits work with small PRs.

# %% [markdown]
# ## GitHub collaboration — contribute to Achiek's repository
# **Known repository:** `https://github.com/DengAchiek/hackathon`  
# **Local folder (Windows):** `C:\Users\HP\hackathon`  
# **Existing files:** `app.py`, `demand_ml.py`, `HACKATHON.md`, `ML_ROADMAP.md`, `README.md`, etc.
#
# Your work belongs under `pesashield-ai/` so it won't overwrite existing GhostDemand files.
#
# ```bash
# cd /c/Users/HP/hackathon
# git pull origin main
# git switch -c feature/pesashield-ai
# mkdir -p pesashield-ai/notebooks pesashield-ai/data/raw pesashield-ai/models
# # Copy this .ipynb into pesashield-ai/notebooks/
# # Keep local CSV out of git until you check data rights and team policy.
# git status
# git add pesashield-ai/notebooks/
# git commit -m "Add PesaShield AI roadmap and Colab notebook"
# git push -u origin feature/pesashield-ai
# ```
#
# Then open GitHub → **Compare & pull request** → request review. If your feature branch already exists, use `git switch feature/pesashield-ai` rather than creating it again. Commit datasets and models only if permissions, size and repo policy allow; use `.gitignore` for private/raw SMS data.

# %% [markdown]
# ## Suggested GitHub file layout
# ```text
# hackathon/
# ├── app.py                 # existing GhostDemand
# ├── ML_ROADMAP.md           # existing GhostDemand
# ├── HACKATHON.md            # existing GhostDemand
# └── pesashield-ai/
#     ├── README.md
#     ├── HACKATHON.md
#     ├── ML_ROADMAP.md
#     ├── notebooks/
#     │   └── pesashield_roadmap.ipynb  <-- THIS NOTEBOOK
#     ├── data/raw/           # don't commit personal SMS
#     ├── ml/
#     ├── models/             # generated artifacts
#     ├── backend/
#     ├── frontend/
#     └── tests/
# ```
#
# **Files to document next:** `README.md` (setup and safety), `HACKATHON.md` (problem, story, pitch), `ML_ROADMAP.md` (dataset details, evaluation, future upgrades).

# %% [markdown]
# ## Future upgrades — after baseline MVP
# - Independent dataset of permitted, verified Kenyan mobile-money scam examples; source-specific testing
# - Multi-class SMS scam taxonomy and English–Kiswahili/code-switch examples
# - Screenshot OCR with privacy-preserving preprocessing
# - Calibrated models, error analysis, threshold tuning, drift monitoring
# - Limited authorized merchant transaction verification and reconciliation
# - Separate PaySim-based transaction anomaly/ fraud-risk module (not trained on SMS text)
# - Controlled feedback loop from analyst reviews; don't retrain on unverified user accusations
#
# **Production gate:** independent validation + privacy and security review + legally authorized transaction integration + human escalation policy.

# %% [markdown]
# ## Three-minute hackathon pitch outline
# | Time | Talk track |
# |---|---|
# | 0:00–0:30 | A merchant sees an apparent payment SMS, but the funds may never have arrived. |
# | 0:30–1:00 | Introduce PesaShield AI: *Verify Before You Trust*. |
# | 1:00–1:45 | Paste suspicious example → show analysis, red flags and risk category. |
# | 1:45–2:20 | Demonstrate `verification_status=not_checked`, then a **mock ledger** match or mismatch. |
# | 2:20–2:45 | Show inbox, analyst review, synthetic-test evaluation and limitations. |
# | 2:45–3:00 | Explain future consented data and authorized financial integrations. |
#
# **Credibility matters:** Say "demonstration trained on synthetic data"; do not advertise that it already protects real M-Pesa accounts.

# %% [markdown]
# ## Final deliverables checklist
# - [ ] Problem statement, users, fraud categories and scope documented
# - [ ] Synthetic SMS dataset catalogued; data rights and limitations recorded
# - [ ] EDA and leakage-aware model training successfully run
# - [ ] Precision/recall/F1/PR-AUC plus confusion matrix reported honestly
# - [ ] Saved baseline artifact and reproducible environment
# - [ ] FastAPI `/predict` actually works in a deployed backend
# - [ ] SQLite assessments and analyst review API implemented
# - [ ] Dashboard wired to API; clear verification status
# - [ ] Tests, privacy checks and demo scenarios completed
# - [ ] Git feature branch PR reviewed by Achiek/team
# - [ ] Live demo, three-minute pitch, future roadmap ready
#
# **Learning order:** run cells above → explain EDA → interpret failures → implement real FastAPI + persistence → build UI → integrate and test → push PR.
#
# *End of roadmap. You can rerun the notebook using the companion synthetic CSV file.*

# %% [markdown]
# ## ▶️ Step H — Generate backend project files
#
# This cell produces a real Python backend module in the artifacts folder. Run it **after saving the model**. You can download/copy the whole `pesashield_artifacts` directory into your repository; keep the model path and API module together.

# %%
from pathlib import Path
api_dir = Path('pesashield_artifacts/api')
api_dir.mkdir(parents=True, exist_ok=True)
(api_dir / '__init__.py').write_text('')
api_source = r'''
from pathlib import Path
import sqlite3
import uuid
from datetime import datetime, timezone

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "sms_baseline.joblib"
DB_PATH = ROOT / "pesashield_api.db"
model = joblib.load(MODEL_PATH)  # only load a trusted model produced by your notebook
app = FastAPI(title="PesaShield SMS Demo API", version="0.2")

class SMSInput(BaseModel):
    message: str = Field(min_length=1, max_length=3000)

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

with get_conn() as conn:
    conn.execute("""CREATE TABLE IF NOT EXISTS assessments (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        score REAL NOT NULL,
        prediction TEXT NOT NULL,
        verification_status TEXT NOT NULL,
        review_status TEXT NOT NULL
    )""")

@app.get("/health")
def health():
    return {"status":"running", "model_ready":True, "verification_capability":"none"}

@app.post("/predict")
def predict(payload: SMSInput):
    if not payload.message.strip():
        raise HTTPException(status_code=422, detail="Nonempty SMS required")
    probability = float(model.predict_proba([payload.message])[0, 1])
    label = "suspicious_pattern" if probability >= 0.5 else "no_modeled_red_flags"
    item_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute("INSERT INTO assessments VALUES (?,?,?,?,?,?)",
                     (item_id, created_at, probability, label, "not_checked", "unreviewed"))
    return {"assessment_id":item_id, "prediction":label,
            "synthetic_model_score":round(probability,4), "verification_status":"not_checked",
            "notice":"Synthetic text-pattern classifier. This does not verify payment."}

@app.get("/assessments")
def assessments(limit: int = 20):
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM assessments ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]
'''
(api_dir / 'main.py').write_text(api_source, encoding='utf-8')
print('Created', api_dir / 'main.py')

# %% [markdown]
# ## ▶️ Step I — Test the prediction API locally
#
# Install packages if Colab is missing them (`pip install fastapi httpx uvicorn`). This checks health, prediction, persistent storage, and rejection of blank messages. **The tests simulate HTTP requests inside the notebook**; they don't start a public server.

# %%
try:
    from fastapi.testclient import TestClient
    import importlib.util
    api_file = Path('pesashield_artifacts/api/main.py').resolve()
    spec = importlib.util.spec_from_file_location('pesashield_api_demo', api_file)
    api_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api_module)
    client = TestClient(api_module.app)
    health = client.get('/health')
    assert health.status_code == 200 and health.json()['model_ready']
    response = client.post('/predict', json={'message':samples[0]})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['verification_status'] == 'not_checked'
    saved = client.get('/assessments', params={'limit':5})
    assert saved.status_code == 200
    assert any(x['id'] == result['assessment_id'] for x in saved.json())
    assert client.post('/predict', json={'message':'   '}).status_code == 422
    print('PASS: /health, /predict, /assessments, empty-message validation')
    print('Example response:', json.dumps(result, indent=2))
except ImportError as exc:
    print('Missing API test dependency:', exc)
    print('Install requirements.txt, then rerun this cell.')

# %% [markdown]
# ## ▶️ Step J — Run outside Colab and connect a frontend
#
# 1. Download `pesashield_artifacts/` containing `sms_baseline.joblib` and `api/main.py`.
# 2. Install trusted dependencies: `pip install fastapi uvicorn scikit-learn joblib httpx`.
# 3. From the directory containing `pesashield_artifacts/`, run: `uvicorn pesashield_artifacts.api.main:app --reload`.
# 4. Open `http://127.0.0.1:8000/docs` and test `/predict`.
# 5. Build a small web page: SMS input → **Analyze** → suspicious/no modeled red flags → synthetic score + verification caveat → recent assessments.
#
# **Before any actual deployment:** authentication, CORS restrictions, rate limiting, HTTPS, privacy policy/retention, audit logging, security checks, consent, genuine validation, and authorized payment verification are required.
