# Mteja-Insight: recommended models and training data

Prepared 27 October 2026. These are implementation recommendations for the retail
and e-commerce app. The existing TF-IDF/logistic regression classifier remains
the running model; the transformer, recovery and forecasting models below have
not been installed or trained by this change.

## What to build first

My recommendation is to retain the current baseline, then compare it against
**SetFit with `sentence-transformers/all-MiniLM-L6-v2` for blocker classification**.
Add semantic product matching using the same base encoder next. This concentrates
the hackathon work on understanding requests and identifying related products.
Outcome prediction and forecasting should follow only when appropriate history
exists. These priorities are engineering judgments, not benchmark results.

[SetFit](https://huggingface.co/docs/setfit/en/index) supports fine-tuning sentence
transformers for text classification with limited labeled data.
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
is an English sentence embedding checkpoint with 384-dimensional embeddings,
suitable for semantic similarity and retrieval; its model card lists Apache-2.0.
Neither is already trained on Mteja-Insight's blocker labels.

## Models by operation

| Operation | Algorithm / concrete model | How Mteja-Insight would use it | Training target and data |
| --- | --- | --- | --- |
| Classify the blocker | Current: character TF-IDF + logistic regression. Next candidate: **SetFit + `sentence-transformers/all-MiniLM-L6-v2`**. | Return one of the app's six blocker labels and a score. Preserve low-score fallback and human review. | Verified request-to-reason pairs. Use retail support text as auxiliary data only after checking its actual meaning. |
| Match product wording and find alternatives | **Sentence Transformer embeddings + cosine similarity**, using the same MiniLM checkpoint; compare against a simple lexical/alias baseline. | Retrieve candidate SKUs from the store catalog and show the best matches for review. Preserve brand, size and model constraints. | Query/product relevance pairs from ESCI plus reviewed pairs from the retailer's own catalog. |
| Extract product, quantity, budget, currency and location | **Token classification / named entity recognition**, fine-tuning `distilbert/distilbert-base-uncased` with a new token-classification head. | Extract text spans into the existing editable preview. Keep numeric/currency validation and deterministic parsing. | Customer-request spans annotated for PRODUCT, QUANTITY, BUDGET, CURRENCY and LOCATION. General NER labels are not a substitute for these annotations. |
| Estimate which requests may be recovered | **Gradient-boosted trees: `CatBoostClassifier`**. | Add a separately labeled, calibrated recovery score for prioritizing follow-up, alongside the current transparent priority formula. | First-party requests linked to verified orders or verified non-conversion outcomes within a defined observation window. |
| Optional later: forecast weekly SKU sales | **Gradient-boosted regression: `LightGBM LGBMRegressor`** with lag/calendar features; compare against a seasonal-naive baseline. | Forecast completed sales as a separate feature. Combine with availability history before attempting unmet-demand estimation. | Dated transactions for each SKU. UCI Online Retail II can support a practice experiment; a store pilot needs the store's own history. |

The [DistilBERT checkpoint](https://huggingface.co/distilbert/distilbert-base-uncased)
is a base English model, not a ready-made retail entity extractor. The official
[Transformers token-classification guide](https://huggingface.co/docs/transformers/en/tasks/token_classification)
shows the fine-tuning approach. [CatBoost](https://catboost.ai/docs/en/) provides
gradient-boosted decision trees, and
[LightGBM's regressor documentation](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html)
describes its regression interface. These references support the methods;
performance on this app still needs measurement.

Voice currently uses the browser's speech service. No audio model is trained by
this app. A dedicated speech model would require a separate transcription and
language evaluation; the text datasets below do not train speech recognition.
English checkpoints also need a separate evaluation before claiming Swahili or
mixed-language support.

## Dataset sources and what they can actually teach

### 1. Bitext Retail / eCommerce — customer-language and intent examples

[Publisher dataset card and download files](https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset)

The card describes **44,884 question/answer pairs, 46 intents and 13 categories**
in English, generated as a hybrid synthetic dataset. Fields include instruction,
intent, category, tags and response. The declared license is **CDLA-Sharing-1.0**.
Use instruction text and relevant intent annotations for a retail-language
experiment. Availability and delivery intents need manual interpretation before
mapping to Mteja-Insight's loss reasons: asking whether an item is available does
not prove it was unavailable or that a sale was lost. Keep synthetic origin in
the training manifest and preserve the source's license information.

### 2. Amazon ESCI Shopping Queries — query/product matching

[Amazon Science repository and data instructions](https://github.com/amazon-science/esci-data)

This supplies query/product judgments labeled **Exact, Substitute, Complement,
Irrelevant**, with product metadata. Its larger version contains **130,652 unique
queries and 2,621,738 judgments**, covering English, Japanese and Spanish; the
repository declares **Apache-2.0**. Use it to train or evaluate relevance ranking
and candidate alternatives. Start with an English subset for the proposed
English checkpoint. Relevance is not stock availability, affordability or proof
of a lost sale. Product-level constraints and the store's actual catalog still
need checking.

### 3. UCI Online Retail II — historical sales forecasting practice

[UCI source, metadata and download](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

UCI provides **1,067,371 transaction records** from a UK online retailer across
2009–2011. Fields include stock code, description, quantity, date, unit price,
customer ID and country. The source lists **CC BY 4.0**. Aggregate cleaned
transactions by SKU and week for a forecasting experiment, treating
cancellations, returns and missing values explicitly. This is historical
completed-transaction data, not labeled failed enquiries or observed unmet
demand; it cannot directly train the six blocker labels or a recovery model.

### 4. Mteja-Insight's own reviewed requests — the essential target dataset

The app already has a local synthetic starter set in `training_data.json` and a
workflow for verified labels. The main dataset for the real product should come
from permitted, minimized request records: retail conversations, zero-result
searches, sales notes, unaccepted quotes and stockout requests. Keep a stable
conversation/customer reference for deduplication, using aliases where possible.

Capture at least:

| Field | Purpose |
| --- | --- |
| `text`, `source`, `captured_at`, `conversation_id` | Original request context and grouped/time-based evaluation |
| `reason`, `verified_by`, `verified_at` | Explicit human ground truth for the blocker |
| `product_id`, `variant`, quantity and budget/currency | Reviewed structured fields and catalog alignment |
| Entity character offsets | Span-level supervision for a future NER model |
| `candidate_product_id`, match judgment | Supervision for semantic catalog matching |
| `outcome`, `outcome_at`, observation window, linked order | Recovery labels with a defined conversion window |
| `origin`, `is_synthetic`, dataset/model version | Distinguish real data, generated examples and provenance |

Not all of these fields exist in the current schema. Entity spans, retrieval
judgments, linked orders and observation-window labels are proposed extensions.
The current `closed` status alone does not establish a verified non-conversion,
and an unresolved request must not automatically be used as a negative example.

A practical first collection target is **50–100 reviewed real examples per
blocker** (300–600 total), followed by more examples for the classes and channels
with the largest errors. This is a planning target, not a guarantee or a minimum
required by the algorithms. Keep a representative real evaluation set separate
from development and synthetic starter examples.

## Implementation path in the existing code

1. **Baseline:** keep `ReasonClassifier` in `demand_ml.py` and its persisted
   vocabulary/weights for comparison. The app already supports training from
   verified labels, version tracking and a low-score fallback.
2. **SetFit adapter:** implement the same `predict(text)` response contract:
   `reason`, `confidence`, `used`, `model_version`. Add a separate artifact loader
   and backend setting; the current JSON weight format cannot load a transformer
   checkpoint. Do not simply point `GHOST_ML_PATH` at a Hugging Face directory.
3. **Product retrieval:** extend `catalog_match` in `app.py` to retrieve candidates
   by embedding similarity. Present suggestions with the exact SKU/variant,
   rather than automatically merging products or promising availability.
4. **NER:** insert the fine-tuned token classifier into `draft_signal` before the
   editable preview. Validate quantities and money with existing field checks.
5. **Recovery/forecasting:** add dedicated datasets and evaluation paths before
   adding scores to `intelligence.py`. Keep observed counts, rule-based priority,
   predicted recovery and sales forecasts clearly labeled as different measures.

## How to select a model honestly

- Compare TF-IDF/logistic regression and SetFit on the **same held-out real
  requests**. Report macro F1, each reason's precision/recall, and how often the
  model defers for human review. Choose the new model only if the evidence
  supports its accuracy, latency and resource cost.
- Group all messages from a conversation/customer together when making splits;
  keep duplicates and paraphrases of one generated template in one partition.
  For a real deployment check, also evaluate on a later time period.
- Evaluate matching with Recall@k and wrong-variant errors; evaluate entities
  with span F1 and exact agreement on quantity/budget/currency.
- Evaluate recovery scores on fully observed outcomes using PR-AUC,
  precision among the top-ranked requests and calibration. Features must be
  available at prediction time; recovered amount and future actions are not
  valid predictors for the initial score.
- Backtest sales forecasts chronologically against the seasonal-naive baseline
  using MAE. Sales alone do not identify demand hidden by stockouts.

For the hackathon, the best next experiment is **SetFit for blockers plus MiniLM
candidate product matching**, with the existing baseline and review workflow
retained. More public data does not automatically supply the target labels the
project needs, and no improvement percentage is claimed before evaluation.
