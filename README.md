# GhostMind 👻🧠

> A self-evolving research agent that remembers how it was wrong.

GhostMind is an AI research agent that reads arXiv papers, retrieves relevant documents, generates answers, and — most importantly — **learns from its own failures** using reinforcement signals. It gets measurably better over sessions without retraining the LLM.

---

## What makes it unique

| Feature | Description |
|---|---|
| **MemRL episodic memory** | Stores intent → strategy → outcome triplets. Q-values update after each session using TD learning. |
| **Agentic RAG + self-correction** | Retriever grades its own results and rewrites queries if relevance is too low. |
| **GraphRAG** | Citation graph (NetworkX) enables relationship-aware retrieval beyond keyword matching. |
| **Self-evaluation** | Every answer is scored for confidence and hallucination rate before being returned. |
| **Failure log** | Explicit, human-readable record of what went wrong and what strategy was tried next. |
| **Multi-provider LLM** | Round-robin across all configured Gemini keys + OpenAI/Anthropic fallbacks with automatic quota handling. |

---

## Architecture

```
User Query
    │
    ▼
Intent Classifier (Gemini)
    │
    ▼
MemRL Strategy Selector ◄──── Q-value lookup from episodic memory
    │
    ▼
arXiv Ingestion ──► Embedding (all-MiniLM-L6-v2, local)
    │
    ▼
Agentic Retriever ──► Grade relevance ──► Rewrite query if needed (loop)
    │
    ▼
GraphRAG Expansion (if strategy = graph|hybrid)
    │
    ▼
Answer Generation (Gemini 2.0 Flash / OpenAI / Anthropic)
    │
    ▼
Self-Evaluator ──► Confidence score + Hallucination rate
    │
    ▼
MemRL Update ──► TD Q-value update ──► Write to episodic memory
    │
    ▼
Failure Log (if quality < 0.4)
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- A free [Gemini API key](https://aistudio.google.com/app/apikey)

### 1. Clone and configure

```bash
# Copy env file
cp backend/.env.example backend/.env

# Edit backend/.env and add your key:
# GEMINI_API_KEY=your_key_here
```

### 2. Run (Mac/Linux)

```bash
chmod +x start.sh
./start.sh
```

### 2. Run (Windows — PowerShell) ✅ Recommended

```powershell
.\start.ps1
```

If you get an execution policy error, run this first (one time):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Run (Windows — Command Prompt)

```cmd
.\start.bat
```

> **Note:** In PowerShell, always use `.\` prefix: `.\start.ps1` or `.\start.bat`. Do NOT use `start.bat` without the prefix — Windows PowerShell won't find it.

### 3. Manual setup (if scripts fail)

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
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

### LLM Providers

GhostMind uses a **multi-provider round-robin** strategy. It distributes calls across all configured providers, skips any that hit quota limits, and falls back gracefully.

Each Gemini key is registered with **both** `gemini-2.0-flash` and `gemini-1.5-flash`, effectively doubling your free-tier capacity per key (1 key = 3,000 free requests/day).

```
# Add as many Gemini keys as you have Google accounts
GEMINI_API_KEY=AIzaSy...your_first_key_here
GEMINI_API_KEY_2=AIzaSy...second_key      # optional
GEMINI_API_KEY_3=AIzaSy...third_key       # optional
# ... up to GEMINI_API_KEY_10

# Paid fallbacks (only used when all Gemini keys are exhausted)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Full Variable Reference

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Free at aistudio.google.com |
| `GEMINI_API_KEY_1` … `GEMINI_API_KEY_10` | *(optional)* | Extra Gemini keys for more free-tier capacity |
| `OPENAI_API_KEY` | *(optional)* | Paid fallback provider |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use |
| `ANTHROPIC_API_KEY` | *(optional)* | Paid fallback provider |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Anthropic model to use |
| `LLM_BACKEND` | `gemini` | Legacy single-backend override |
| `LLM_MODEL` | `gemini-2.0-flash` | Legacy model override |
| `DATABASE_URL` | `sqlite+aiosqlite:///./ghostmind.db` | SQLite by default, PostgreSQL supported |
| `ARXIV_MAX_RESULTS` | `15` | Papers fetched per query |
| `ARXIV_CATEGORIES` | `["cs.AI","cs.LG","cs.CL","cs.IR"]` | arXiv category filters |
| `MEMRL_EPSILON` | `0.15` | Exploration rate (0=exploit, 1=explore) |
| `MEMRL_ALPHA` | `0.1` | Learning rate for Q-value updates |
| `MEMRL_GAMMA` | `0.9` | Discount factor |
| `MEMORY_MAX_TRIPLETS` | `10000` | Max episodic triplets to store |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/query` | Run a research query |
| `POST` | `/api/v1/feedback` | Submit thumbs up/down on a session |
| `GET` | `/api/v1/sessions` | List all sessions |
| `GET` | `/api/v1/sessions/{id}` | Get full session detail |
| `GET` | `/api/v1/benchmarks` | Per-session benchmark metrics |
| `GET` | `/api/v1/memory` | MemRL state + failure log |
| `GET` | `/api/v1/stats` | System-wide stats |
| `GET` | `/health` | Health check |

Interactive docs: **http://localhost:8000/docs**

---

## Tech Stack

| Layer | Tech |
|---|---|
| LLM (primary) | Google Gemini 2.0 Flash + Gemini 1.5 Flash (free tier, round-robin) |
| LLM (fallback) | OpenAI gpt-4o-mini / Anthropic claude-haiku-4-5-20251001 (paid, optional) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers (local, no API) |
| Backend | FastAPI + SQLAlchemy async |
| Database | SQLite (default) / PostgreSQL |
| Knowledge graph | NetworkX |
| Paper ingestion | arXiv Python SDK |
| Frontend | React 18 + Vite + Recharts + lucide-react |
| MemRL | Custom TD learning on CPU |

---

## How the memory works

After every query, GhostMind:

1. Hashes the detected **intent** (e.g. "reduce hallucination in RAG")
2. Records which **strategy** was used (semantic / graph / hybrid / aggressive_rewrite)
3. Computes an **outcome quality** score: `confidence × (1 − hallucination_rate)`
4. Updates the **Q-value** for that intent+strategy pair using TD learning:

```
Q_new = Q_old + α × (outcome + γ × outcome − Q_old)
```

Next time a similar intent is detected, the agent picks the strategy with the highest Q-value (with ε-greedy exploration).

---

## Project structure

```
ghostmind/
├── backend/
│   ├── app/
│   │   ├── core/           # config, database, models, LLM, embeddings
│   │   ├── retrieval/      # arXiv ingestion + agentic retriever
│   │   ├── graph/          # GraphRAG knowledge graph
│   │   ├── memory/         # MemRL Q-value engine
│   │   ├── evaluation/     # self-eval confidence + hallucination scoring
│   │   ├── api/            # FastAPI routes
│   │   └── agent.py        # main orchestrator
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── pages/          # Query, Dashboard, Memory, Sessions
│       └── utils/api.js
├── start.sh                # Mac/Linux one-command run
├── start.ps1               # Windows PowerShell one-command run
├── start.bat               # Windows CMD one-command run
└── docker-compose.yml
```

---

## Roadmap

- [ ] PostgreSQL + pgvector for production embeddings
- [ ] PDF upload and ingestion
- [ ] Multi-hop reasoning evaluation benchmark
- [ ] Export memory snapshots
- [ ] Semantic Scholar integration alongside arXiv

---

## License

MIT
