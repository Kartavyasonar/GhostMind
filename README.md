# GhostMind 👻🧠

> A self-evolving AI research agent that learns which retrieval strategy works best — and remembers it.

GhostMind fetches real arXiv papers, retrieves relevant documents using one of four strategies, generates grounded answers, then **updates its own memory** to do better next time. No LLM retraining. No manual tuning. Pure episodic reinforcement learning on top of RAG.

---

## What makes it unique

| Feature | Description |
|---|---|
| **MemRL episodic memory** | Stores `intent → strategy → outcome` triplets. Q-values update after every session using TD(0) learning. The agent learns which retrieval strategy works best per topic. |
| **4 distinct retrieval strategies** | `semantic` (cosine similarity), `hybrid` (embedding + keyword re-rank), `graph` (citation graph expansion), `aggressive_rewrite` (domain keyword expansion + LLM rewrite). Each retrieves genuinely different documents. |
| **Cold-start rotation** | First 4 sessions always try each strategy once so the memory table gets populated before exploitation begins. |
| **Epsilon-greedy exploration** | After cold-start, the agent exploits the best known strategy 85% of the time and explores randomly 15% of the time (configurable). |
| **Code-level reward signal** | Q-learning reward is computed from answer length, source coverage, lexical diversity, and retrieval score — not from the LLM evaluator, which can return identical scores every run. |
| **Groq-first multi-provider LLM** | Groq (free, fast) is always tried first. Gemini, OpenAI, and Anthropic are fallbacks. Rate-limited providers are skipped instantly — no sleeping. |
| **Intent fuzzy bucketing** | "What is MemRL?" and "Explain MemRL architecture" hash to the same memory slot, so memory accumulates across paraphrased repeats. |
| **GraphRAG** | Citation graph (NetworkX) enables relationship-aware retrieval for `graph` and `hybrid` strategies. |
| **Self-evaluation** | Every answer is scored for confidence and hallucination rate by the LLM evaluator (for display). |
| **Failure log** | Explicit record of sessions where quality < 0.4 and what strategy was tried next. |

---

## Architecture

```
User Query
    │
    ▼
Fast Intent Classifier (Groq, 20 tokens)
    │  → normalised phrase, e.g. "Explain MemRL framework"
    │  → fuzzy-bucketed to stable topic key, e.g. "memrl_framework"
    ▼
MemRL Strategy Selector ◄──── Q-value lookup from episodic memory
    │  cold-start (sessions 1-4): rotate semantic→hybrid→graph→aggressive_rewrite
    │  exploit (85%): pick highest Q-value strategy for this topic
    │  explore (15%): try an under-explored strategy
    ▼
arXiv Ingestion ──► Embedding (all-MiniLM-L6-v2, local, no API)
    │
    ▼
Strategy-Specific Retriever
    ├── semantic:            top-K cosine similarity
    ├── hybrid:              2× candidates, re-ranked with 60% embedding + 40% keyword
    ├── graph:               semantic seeds + citation graph expansion (2 hops)
    └── aggressive_rewrite:  domain keyword expansion + optional LLM query rewrite
    │
    ▼
Answer Generation (Groq LLaMA 3, with optional memory hint in prompt)
    │
    ▼
Self-Evaluator (LLM) ──► confidence score + hallucination rate (display only)
    │
    ▼
Code-Level Quality Signal ──► answer length + source coverage + retrieval score
    │  this is the actual Q-learning reward (varies per session/strategy)
    ▼
MemRL TD(0) Update
    │  Q_new = Q_old + α × (reward − Q_old)
    │  exploration steps use 0.5× alpha to protect established Q-values
    ▼
Episodic Memory (SQLite) ──► next query uses updated Q-values
    │
    ▼
Failure Log (if quality < 0.4)
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Free [Groq API key](https://console.groq.com) ← **recommended primary provider**
- Optional: [Gemini API key](https://aistudio.google.com/app/apikey) as fallback

### 1. Clone and configure

```bash
cp backend/.env.example backend/.env
# Edit backend/.env — at minimum set GROQ_API_KEY
```

### 2. Run (Windows — PowerShell)

```powershell
.\start.ps1
```

If you get an execution policy error, run once:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Run (Mac/Linux)

```bash
chmod +x start.sh && ./start.sh
```

### 2. Run (Windows — Command Prompt)

```cmd
.\start.bat
```

### 3. Manual setup (if scripts fail)

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Environment Variables

See `.env.example` for the full annotated file. Key variables:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(recommended)* | Free at console.groq.com — 14,400 req/day, no daily quota |
| `GEMINI_API_KEY` | *(optional fallback)* | Free at aistudio.google.com — 1,500 req/day per key |
| `GEMINI_API_KEY_1` … `_10` | *(optional)* | Add multiple keys to increase Gemini daily capacity |
| `OPENAI_API_KEY` | *(optional, paid)* | Fallback if Groq and Gemini both fail |
| `DATABASE_URL` | `sqlite+aiosqlite:///./ghostmind.db` | SQLite default; PostgreSQL supported |
| `ARXIV_MAX_RESULTS` | `15` | Papers fetched per query from arXiv |
| `MEMRL_EPSILON` | `0.15` | Exploration rate — 0.0 = pure exploit, 1.0 = pure random |
| `MEMRL_ALPHA` | `0.15` | Q-value learning rate — lower = slower but more stable |
| `MEMRL_GAMMA` | `0.9` | Discount factor (not used in current single-step TD) |
| `LLM_TEMPERATURE` | `0.2` | Generation temperature |
| `LLM_MAX_TOKENS` | `2048` | Max tokens per answer |

### Recommended demo config (`.env`)

```env
GROQ_API_KEY=your_groq_key_here
MEMRL_EPSILON=0.15     # explore 15% of the time
MEMRL_ALPHA=0.15       # stable learning rate
```

After ~20 sessions on the same topic, set `MEMRL_EPSILON=0.0` to show pure exploitation in your demo.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/query` | Run a research query through the full pipeline |
| `POST` | `/api/v1/feedback` | Submit a score (0-1) on a session |
| `GET` | `/api/v1/sessions` | List all sessions |
| `GET` | `/api/v1/sessions/{id}` | Full session detail including answer and sources |
| `GET` | `/api/v1/benchmarks` | Per-session quality metrics for charting |
| `GET` | `/api/v1/memory` | MemRL state — Q-values per strategy, failure log |
| `GET` | `/api/v1/stats` | System-wide totals |
| `GET` | `/health` | Health check |

Interactive API docs: **http://localhost:8000/docs**

### Example query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is MemRL and how does it work?"}'
```

The response includes a `memrl_debug` block showing exactly what memory knew before and after the session:

```json
{
  "retrieval_strategy": "hybrid",
  "outcome_quality": 0.887,
  "q_value_after": 0.891,
  "memrl_debug": {
    "was_exploring": false,
    "intent_bucket": "memrl_framework",
    "memory_before": {
      "best_strategy": "semantic",
      "best_q": 0.874,
      "q_values": {
        "semantic":  {"q": 0.874, "visits": 3},
        "hybrid":    {"q": 0.838, "visits": 2},
        "graph":     {"q": 0.755, "visits": 1},
        "aggressive_rewrite": {"q": 0.845, "visits": 1}
      }
    },
    "q_delta": 0.012
  }
}
```

---

## How the memory works

After every query, GhostMind:

1. **Classifies** the query into a normalised intent phrase (e.g. `"Explain MemRL framework"`)
2. **Buckets** it to a stable topic key (e.g. `"memrl_framework"`) so paraphrased repeats share memory
3. **Selects** a retrieval strategy via epsilon-greedy on stored Q-values
4. **Retrieves** documents using that strategy (each strategy retrieves differently)
5. **Generates** an answer, injecting a memory hint if prior quality was high
6. **Evaluates** quality via a code-level signal (answer length + source coverage + retrieval score)
7. **Updates** the Q-value for this topic+strategy pair:

```
Q_new = Q_old + α × (reward − Q_old)
```

8. **Next time** a similar query arrives, the agent exploits the strategy with the highest Q-value

### Why Q-values might decrease

This is correct and expected behaviour. If strategy X scored 0.90 once, but scores 0.85 on the next visit, the Q-value moves toward the true average (0.87). Q-values converge to the mean reward for each strategy — whichever strategy has the highest mean wins.

---

## Demonstrating learning (for interviews/demos)

**Step 1 — Cold start (sessions 1-4):** Run the same query 4 times. Each session uses a different strategy. Watch the Sessions page show `semantic → hybrid → graph → aggressive_rewrite`.

**Step 2 — Exploitation kicks in (sessions 5+):** The agent starts picking the strategy with the highest Q-value. Watch the Memory page — Q-values differ across strategies, and the best one is highlighted.

**Step 3 — Show the memory page:** All 4 strategies have Q-values. The bar chart shows which one the system has learned is best for this topic.

**Step 4 — Try a different topic:** Ask a completely different question. Memory starts fresh for that topic (cold-start again), then converges to its own best strategy independently.

**Step 5 — Pure exploitation:** Set `MEMRL_EPSILON=0.0` in `.env` and restart. Now every session exploits memory 100% — strategy never changes because memory has converged.

---

## Tech Stack

| Layer | Tech |
|---|---|
| LLM | Groq (LLaMA 3.3 70B, free) — Gemini / OpenAI / Anthropic as fallbacks |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers (local, no API key needed) |
| Backend | FastAPI + SQLAlchemy async |
| Database | SQLite (default) / PostgreSQL |
| Knowledge graph | NetworkX |
| Paper ingestion | arXiv Python SDK |
| Frontend | React + Vite + Recharts |
| MemRL | Custom TD(0) Q-learning, CPU-only |

---

## Project structure

```
ghostmind/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── config.py        # all env vars and settings
│   │   │   ├── database.py      # async SQLAlchemy session
│   │   │   ├── models.py        # ORM: sessions, triplets, benchmarks, failures
│   │   │   ├── llm.py           # Groq-first multi-provider client
│   │   │   └── embeddings.py    # MiniLM local embedder
│   │   ├── retrieval/
│   │   │   ├── ingestion.py     # arXiv fetch + embed + store
│   │   │   └── retriever.py     # 4 strategy-specific retrieval functions
│   │   ├── graph/
│   │   │   └── knowledge_graph.py  # NetworkX citation graph + expansion
│   │   ├── memory/
│   │   │   └── memrl.py         # TD(0) Q-learning, fuzzy bucketing, strategy selection
│   │   ├── evaluation/
│   │   │   └── self_eval.py     # LLM-based confidence + hallucination scoring
│   │   ├── api/
│   │   │   └── routes.py        # FastAPI endpoints
│   │   └── agent.py             # main pipeline orchestrator
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── pages/               # Research, Benchmarks, Memory, Sessions
│       └── utils/api.js
├── start.sh / start.ps1 / start.bat
└── docker-compose.yml
```

---

## Troubleshooting

**Queries taking 3-5 minutes:** Your Gemini keys are rate-limited. Set `GROQ_API_KEY` in `.env` — Groq is free and fast (~5 seconds per query).

**"All LLM providers unavailable":** All your API keys have hit their limits. Set a Groq key — it has no daily quota.

**Confidence stuck at 80%/20%:** This is the LLM evaluator being consistent. It's display-only and doesn't affect learning. The `code_quality` signal in `memrl_debug` is what the Q-learner actually uses, and that varies per session.

**Q-values slowly decreasing:** Expected. Q-values converge to the true mean reward per strategy. If the best strategy is genuinely better, its Q-value will be highest at convergence.

**Strategy never changes from "semantic":** Delete `ghostmind.db` and restart — the cold-start rotation will ensure all 4 strategies are tried before exploitation begins.

---

## Roadmap

- [ ] PostgreSQL + pgvector for production-scale embeddings
- [ ] PDF upload and ingestion alongside arXiv
- [ ] Semantic Scholar integration
- [ ] Export / import memory snapshots
- [ ] Multi-hop reasoning benchmark
- [ ] Strategy performance comparison chart in UI

---

## License

MIT