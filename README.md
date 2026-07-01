---
title: SHL Assessment Recommender
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---
# SHL Conversational Assessment Recommender

A stateless FastAPI agent that takes a recruiter from a vague intent
("I'm hiring a Java developer") to a grounded shortlist of SHL **Individual Test
Solutions** through dialogue. It clarifies vague queries, recommends 1â€“10
assessments, refines on constraint changes, compares two assessments from
catalog data, and refuses anything off-topic or any prompt-injection attempt.

## Layout
```
app/         FastAPI service + agent
  main.py       /health, /chat
  agent.py      routing + retrieval + selection + fallbacks
  retrieval.py  local-embedding cosine search over the catalog
  llm.py        provider-agnostic JSON LLM client (groq/gemini/openai/anthropic)
  prompts.py    controller / selector / compare prompts
  catalog.py    catalog loader + grounding lookups
  schemas.py    request/response contract (Pydantic)
  config.py     env-driven settings
scraper/     scrape_catalog.py  -> data/catalog.json
eval/        run_traces.py      -> mean Recall@10 + probes on SHL traces
tests/       test_agent.py      -> offline guardrail tests (no network/key)
data/        catalog.json (you generate) + catalog.sample.json (placeholder)
```

## 1. Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then paste a free API key (Groq recommended)
```

## 2. Build the catalog
```bash
python -m scraper.scrape_catalog --out data/catalog.json --enrich
```
Before a full run, open the catalog page in a browser and confirm the two
selectors marked `# VERIFY` in `scraper/scrape_catalog.py` still match SHL's
markup. To smoke-test without scraping, set `CATALOG_PATH=data/catalog.sample.json`.

## 3. Run
```bash
uvicorn app.main:app --reload
curl localhost:8000/health
curl -X POST localhost:8000/chat -H 'content-type: application/json' -d '{
  "messages": [{"role":"user","content":"Hiring a mid-level Java developer who works with stakeholders"}]
}'
```

## 4. Test (offline, no key needed)
```bash
pytest -q                 # guardrails: schema, injection, vague-clarify, grounding
```

## 5. Evaluate against SHL traces
```bash
python -m eval.run_traces --traces ./traces          # simulated multi-turn
python -m eval.run_traces --traces ./traces --scripted   # fast one-shot recall
```

## 6. Deploy (Render example)
Push to GitHub, create a Blueprint from `render.yaml`, set `LLM_API_KEY` as a
secret. `/health` and `/chat` will be reachable at your service URL. Free
instances cold-start within the 2-minute window the spec allows.

## Endpoints
- `GET /health` â†’ `{"status":"ok"}` (200)
- `POST /chat` â†’ `{"reply": str, "recommendations": [...], "end_of_conversation": bool}`
  - `recommendations` empty while clarifying/refusing; 1â€“10 items on a shortlist.

