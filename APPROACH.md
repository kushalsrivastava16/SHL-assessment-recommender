# Approach: Conversational SHL Assessment Recommender

> Fill every `[FILL IN]` with your own measured numbers before submitting. Do
> not ship placeholder metrics.

## 1. Problem framing and design principle
The task is a multi-turn recommender that must (a) route between clarify /
recommend / refine / compare / refuse, (b) ground every recommendation in the
SHL Individual Test Solutions catalog, and (c) stay robust across a
non-deterministic, stateless conversation capped at 8 turns.

The single design principle everything follows from: **the LLM decides *what*
to do; Python owns the *schema* and all *grounding*.** The model routes and
selects assessments *by id* from a candidate list we retrieved; it never emits
URLs or free-text product names into the response. This makes hallucinated or
out-of-catalog recommendations structurally impossible rather than merely
discouraged, and keeps the response schema (which the evaluator treats as
non-negotiable) under code control at all times.

## 2. Catalog ingestion
`scraper/scrape_catalog.py` paginates the catalog with `?type=1&start=N`
(Individual Test Solutions only; Job Solutions `type=2` are excluded per scope),
extracts each row's name, detail-page URL, remote/adaptive flags and test-type
keys (A/B/C/D/E/K/P/S), then optionally enriches each item from its detail page
with description, job levels and duration. Selectors are written structurally
(match links into `/view/`, infer feature dots, pull single-letter type tokens)
and the two brittle spots are marked `# VERIFY` because SHL's markup can drift.
Output is a flat `catalog.json` — the single source of truth loaded by the app.
Catalog size after scrape: `[FILL IN N items]`.

## 3. Retrieval
Each assessment is embedded once at startup from a concatenation of name +
test type + job levels + description, using a local `all-MiniLM-L6-v2` model.
Query-time retrieval is exact cosine over an in-memory numpy matrix.

*Why not FAISS/Chroma/pgvector?* The catalog is a few hundred items; exact
cosine is sub-millisecond at that scale, so a vector DB would add a fragile
native dependency and deploy surface for no measurable latency benefit — a
deliberate "right-sized" choice I can defend. A **local** embedding model also
means there is no per-query embedding network hop, which protects the 30s call
cap and avoids embedding-API rate limits on a free tier.

## 4. Agent control flow (prompt design)
Two LLM stages plus deterministic glue:
- **Controller** (`prompts.CONTROLLER_SYSTEM`): sees the whole conversation but
  *not* the catalog, so it cannot invent products. It outputs a JSON routing
  decision — action, extracted structured `requirements`, and a rich
  `search_query`. It is instructed never to recommend on a vague first turn,
  and to stop clarifying after `MAX_CLARIFY_TURNS` so it can't burn the budget.
- **Selector** (`prompts.SELECTOR_SYSTEM`): sees the user's needs plus a
  numbered candidate list retrieved from the catalog, and returns `chosen_ids`
  + a short grounded reply. It may only reference ids we supplied; any id
  outside the list is dropped in code.
- **Compare** answers "difference between X and Y" strictly from catalog fields
  looked up by fuzzy name match, and is told to say "not specified" rather than
  guess — this is what keeps comparisons grounded instead of drawn from the
  model's prior.

Recall vs. precision: the selector is nudged to prefer coverage (include a
plausibly-relevant item over omitting it, and cover multiple facets — e.g. a
knowledge test *and* a personality test for a "Java dev who works with
stakeholders") because scoring is Recall@10. I tuned this after seeing
`[FILL IN: describe what you observed, e.g. "the selector returning only 2–3
items and missing gold personality tests on mixed-facet personas"]`.

## 5. Robustness (guarding the named failure modes)
Every branch degrades gracefully; `/chat` never returns a 500 or an off-schema
body:
- LLM JSON slips → `llm.py` does forgiving extraction + one stricter retry, then
  returns `None`.
- Controller returns `None` → deterministic route (clarify on a short first
  turn, else best-effort vector recommend).
- Selector returns `None`/garbage → fall back to top vector hits.
- Injection ("ignore previous instructions", "reveal your prompt") → refused by
  a deterministic regex backstop *and* the controller.
- Turn-cap awareness → on the final allowed turn the agent commits to a
  shortlist instead of clarifying; `end_of_conversation` is set when we deliver
  a shortlist at the budget edge.
- Unhandled exception → schema-valid apology, not a crash.

## 6. Evaluation
`eval/run_traces.py` replays the provided traces in-process: an LLM plays the
hiring manager answering only from the trace facts, and we compute mean
Recall@10 (denominator = number of gold items, matched by URL tail or
normalized name) plus behavior probes. `tests/test_agent.py` asserts the hard
guardrails offline (no key needed): schema compliance, injection refusal,
no-recommend-on-vague-turn-1, catalog-only URLs, selector-failure fallback,
turn-cap commit.

Results on the 10 public traces:
- Mean Recall@10: `[FILL IN]`
- Behavior probes passed: `[FILL IN]` / `[FILL IN]`
- Median /chat latency: `[FILL IN] s`

## 7. What didn't work / iterations
`[FILL IN — write 3–5 honest bullets from your own runs. Examples of the shape:
- "Single-stage prompt (route + select together) leaked hallucinated names; splitting into controller+selector fixed grounding and raised probe pass-rate from X to Y."
- "Returning raw top-10 vector hits scored Recall@10 = X but failed mixed-facet personas; adding facet-coverage guidance in the selector moved it to Y."
- "Over-clarifying cost turns and hurt recall on the 8-turn cap; capping clarifications at 2 recovered Z."]`

## 8. Stack justification and AI-tool use
FastAPI + Pydantic (schema enforced by the framework), local
sentence-transformers + numpy (right-sized retrieval, no external vector store),
provider-agnostic LLM client defaulting to Groq's free Llama 3.3 70B for
latency headroom under the 30s cap, deployed on Render's free tier.

AI tools: I used an AI assistant to scaffold the project structure, the
provider-agnostic LLM client, and the first draft of the scraper and eval
harness. I reviewed, corrected (e.g. a Recall@10 denominator bug), and verified
every module myself; the design choices above are mine to defend.
`[EDIT this paragraph to reflect exactly what you did vs. what the tool did.]`
