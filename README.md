# Agen1 Research

A local-first, autonomous research runner for Linux/macOS terminals. Give it a question and a time budget. A planner delegates to parallel researchers, a supervisor detects stalled or repetitive work, and a writer creates a cited Markdown paper draft from the gathered evidence.

This is an initial working implementation. The offline workflow and automated tests are verified; real model quality, live provider access, and multi-hour runs still need validation on your machine. Papers are research drafts, not independently verified scientific results.

## Quick start

Python **3.11+** is required. The core application has no third-party Python dependencies. From this repository:

```bash
python3 -m agen1 run "How can retrieval-augmented generation reduce factual errors?" \
  --demo --hours 0.01 --run-dir runs/demo
```

The demo runs completely offline using **synthetic evidence**. It exercises planning, parallel workers, retrieval, notes, logs, and paper writing. It is not actual research.

For an installed command, optionally use:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
agen1 --help
```

To extract text from downloaded PDFs, install the optional extra:

```bash
python -m pip install -e '.[pdf]'
```

## Connect a local LLM

**Already running vLLM?** Follow [the local vLLM setup and test guide](docs/VLLM.md), including 60-, 180-, and 600-second research runs, proposal attachments, and longer sessions.

Use an OpenAI-compatible server, such as vLLM, Ollama's `/v1` endpoint, or LM Studio. Start your chosen server and load a model before running Agen1. The model must follow JSON instructions; native tool calling is not required.

For example, with Ollama installed:

```bash
ollama pull qwen2.5:14b
ollama serve  # Only if Ollama is not already running
```

In another terminal, export your credentials securely. `.env.example` documents their names; `.env` files are **not** automatically loaded. Do not commit keys.

```bash
export SEMANTIC_SCHOLAR_API_KEY='your-key'
export BRAVE_SEARCH_API_KEY='your-key'
# Optional, if your local model server requires authentication:
# export LOCAL_LLM_API_KEY='your-local-server-key'

python3 -m agen1 doctor --base-url http://127.0.0.1:11434/v1 --model qwen2.5:14b
python3 -m agen1 run "Compare recent methods for evaluating factuality in RAG" \
  --hours 4 --workers 3 --llm-concurrency 2 \
  --base-url http://127.0.0.1:11434/v1 --model qwen2.5:14b \
  --max-tokens 500000 --max-steps 1000 --run-dir runs/rag-review --verbose
```

For LM Studio, use its configured model ID and a base URL such as `http://127.0.0.1:1234/v1`. If your GPU/server only handles one generation at once, set `--llm-concurrency 1`. Researchers still run concurrently while requests queue. Increase `--request-timeout` and `--stall-seconds` for slow models; the stall window should cover request queueing and generation time.

The cloud machine cannot reach your laptop's `127.0.0.1`. Run Agen1 on the same machine as the model, or configure a reachable server address and appropriate network access.

## Supply a proposal or reference material

Attach local files when starting a run. Repeat either option to supply more than one file:

```bash
python3 -m agen1 run "How might a lid affect the cooling rate of a cup of water?" \
  --proposal examples/cooling-proposal.md \
  --material examples/cooling-data.csv \
  --material /path/to/reference-paper.pdf \
  --hours 1 --run-dir runs/cooling --verbose
```

`--proposal` supplies research goals, scope, and constraints. `--material` supplies supporting context such as notes, papers, or datasets. Both become source records with IDs that agents can cite. The planner, researchers, supervisor, and writer receive bounded excerpts; researchers can use `read_material` to read additional sections of an imported text snapshot. This does not require `--full-access`.

Supported files are UTF-8 text (`.txt`, `.md`, `.markdown`, `.rst`), tables (`.csv`, `.tsv`), structured text (`.json`, `.jsonl`), HTML (`.html`, `.htm`), PDF, and DOCX. PDF extraction requires the optional `pdf` extra; DOCX extraction uses the standard library. Tables and structured files are supplied as text, not automatically interpreted as validated datasets. Images, audio, scanned PDFs requiring OCR, and other binary formats are not supported.

Import happens **before the research time budget starts**. Each input is limited to 20 MB, with up to 200,000 extracted characters retained. PDF extraction covers at most the first 30 pages and does not perform OCR. Extracted text snapshots are stored under `materials/`, with import metadata in `materials.json`. Original files are not modified or copied as raw binaries. Configured secrets are redacted using the same logging rules. Imports and truncation are recorded in the activity log.

Identical content with the same role is deduplicated. Reimporting changed content creates another source version. Saved snapshots remain available on resume even if the original files have moved or been deleted. Supplied material is unverified context: a proposal can guide research scope, but its claims are not automatically treated as established findings, and document contents do not authorize system actions.

To add files to an existing run, stop it first, import the files, then resume:

```bash
python3 -m agen1 add-material --run-dir runs/cooling \
  --proposal /path/to/revised-proposal.md --material /path/to/additional-notes.txt
python3 -m agen1 run --run-dir runs/cooling --hours 1 --model qwen2.5:14b
```

Imports use the run's process lock. Adding material while the run is active is rejected; there is no live attachment reload in this version.

For a short offline check with the included **non-scientific fixtures**:

```bash
python3 -m agen1 run "How might a lid affect the cooling rate of a cup of water?" \
  --proposal examples/cooling-proposal.md --material examples/cooling-data.csv \
  --demo --seconds 5 --run-dir runs/cooling-demo
```

`--seconds` and `--hours` are mutually exclusive duration options. Very short budgets can end before useful findings are recorded; the saved fallback paper should state that limitation. The demo's model responses and retrieved evidence are synthetic, so its output cannot answer the scientific question reliably.

## What runs automatically

1. **Planner:** splits the question into distinct assignments based on the selected duration and any saved evidence.
2. **Researchers:** use Semantic Scholar, Brave, page retrieval, and optional PDF extraction. They share sources and source-linked notes. Each model decision produces one visible tool call.
3. **Watchdog supervisor:** checks every five seconds. It cancels agents that make no evidence progress within `--stall-seconds`. Three identical calls, eight unproductive actions, ten actions without a saved finding, four consecutive failures, step limits, and a shared token budget also stop unproductive work.
4. **Supervisor reviewer:** every 30 seconds, a separate LLM role reviews each active worker's recent calls, outcomes, errors, and shared evidence. It can redirect an assignment (up to twice per worker) or stop it. The watchdog remains active if the reviewer fails.
5. **Restarts:** if a worker fails, or a round ends with no findings, research restarts while time remains, up to `--max-restarts` times (default 5). The methods alternate: `agents-simplified` (fresh conversations, reasoning off, half the context) and `direct` (the engine chooses the searches itself, fetches pages, and asks the model about one source at a time). If the model is unavailable, `direct` saves relevant sentences labelled as verbatim excerpts, so evidence is still collected. A round also starts when agents finish cleanly but retrieved sources remain unread, so that `direct` can read them; pass `--max-restarts 0` to disable all of this. `summary.json` lists each round under `rounds`.
6. **Writer:** uses the evidence and notes to produce `paper.md`, including retrieved references. Unknown source-ID citations trigger an evidence-only fallback. This checks citation IDs, not whether every claim is scientifically supported.

`--hours` (or `--seconds`) sets a **maximum session duration**, not a promise to keep busy for that entire period. Writing reserves 20% of the duration, capped at five minutes. Research can finish early when assignments finish or limits are reached. If synthesis is unavailable, the program still saves an honest fallback report. File persistence and process cleanup can take a little additional time.

The default token budget is 250,000 per session, with 20% reserved for synthesis. Usage comes from the server when supplied; otherwise it is estimated from text length. Per-request input is bounded with recent history and compact shared evidence, so long runs do not grow the model context indefinitely. Tune the budgets to your hardware and research duration.

## Recovery and persistence

- Transient HTTP failures and rate limits receive up to three attempts, honoring numeric `Retry-After` values up to 60 seconds. Requests have hard timeouts and response size limits.
- Cut-off or invalid JSON responses count as transient failures and are retried. Model replies are streamed, which keeps slow responses alive on networks that stall silent connections, and a draft cut off partway is kept and labelled rather than lost.
- Model replies are parsed leniently: code fences, surrounding prose, arguments placed beside `"tool"`, and arrays of actions (up to five per reply) all work. Replies that are still unusable, and tool failures, are returned to the researcher so it can change approach.
- An empty reply usually means a reasoning model used its whole output budget. The client retries with `chat_template_kwargs: {"enable_thinking": false}` and more output tokens, and keeps reasoning off for the rest of the session. `events.jsonl` records the `finish_reason`.
- `search_papers` works without a Semantic Scholar key, at the shared public rate limit, and falls back to OpenAlex when Semantic Scholar is unavailable.
- If the planner fails on resume, the run reuses the assignments saved in `plan.json`.
- Recovery is bounded by the time budget, the token budget, and `--max-restarts`. Missing credentials, invalid claims, or broken dependencies cannot always be repaired autonomously. These failures remain visible in the logs.
- Sources and notes are checkpointed after changes. Resume saved evidence with:

```bash
python3 -m agen1 run --run-dir runs/rag-review --hours 2 --model qwen2.5:14b
```

Resume starts a **new session budget and plan**; it does not restore exact in-flight conversations or previously running commands. Reapply desired model, worker, and permission options. Use separate directories for different topics. A process lock prevents two runs from writing to the same directory.

`Ctrl+C` or `SIGTERM` stops the workers and attempts final synthesis. Model unavailability falls back to the saved evidence. Abrupt termination preserves the last completed checkpoint but cannot guarantee a final paper.

## Terminal display and artifacts

Every model request/response, action, result, retry, error, and supervisor intervention appears as a terminal event. `--verbose` also shows full message contents and tool results, within documented tool output limits. There is no graphical UI in this version.

| File | Contents |
| --- | --- |
| `activity.md` | Human-readable chronological activity, including queries and action summaries |
| `events.jsonl` | Structured conversations, tool arguments/results, errors, and interventions |
| `plan.json` | Assignments and time allocation |
| `state.json` | Checkpointed topic, sources, notes, session count |
| `sources.json` | Source metadata, retrieved/imported excerpts, evidence type, and IDs |
| `materials.json` | Imported-file roles, source IDs, provenance, and extraction limits |
| `materials/` | Redacted extracted-text snapshots of supplied files |
| `status.json` | Live worker status, then the final run summary |
| `summary.json` | Completed run metrics and reasons workers stopped |
| `paper.md` | Generated research draft or explicit fallback report |
| `workspace/` | Downloaded PDFs and agent-created files |

```bash
python3 -m agen1 status runs/rag-review
tail -f runs/rag-review/activity.md
```

Known secret values from environment variable names containing `API_KEY`, `TOKEN`, `SECRET`, or `PASSWORD`, plus bearer tokens, are redacted from persisted activity. Logs can still contain research material and sensitive file contents. Redaction cannot identify every possible secret, especially credentials introduced after startup. Treat run directories accordingly.

## System access

Default tools read and write within the run's `workspace/`; shell execution is disabled. To authorize unattended system actions:

```bash
python3 -m agen1 run "Your research question" --hours 4 \
  --allow-shell --full-access --run-dir runs/system-research --verbose
```

- `--allow-shell` lets the model execute Bash commands with the current OS user's permissions. Its working directory is the run workspace, but **this is not a shell sandbox**: commands can access other files and the network.
- `--full-access` additionally permits the file tools to use paths outside the workspace. It requires `--allow-shell`.
- Commands have bounded output and execution time (at most 60 seconds per call); process groups are terminated on timeout/cancellation. They cannot gain permissions the OS user does not have.
- Run with a dedicated user/container if you want OS-level isolation. Do not run as root merely to enable research. Retrieved web pages and papers are untrusted input; prompt instructions reduce, but do not eliminate, prompt-injection risk.

## Retrieval details and limits

Semantic Scholar uses `api.semanticscholar.org/graph/v1/paper/search` with `SEMANTIC_SCHOLAR_API_KEY`. Calls are serialized at roughly one per second. Brave uses `api.search.brave.com/res/v1/web/search` with `BRAVE_SEARCH_API_KEY`. Your provider plan may impose additional quotas.

Paper search retrieves metadata, abstracts, and open-access PDF links. PDF downloads are limited to 20 MB. Optional text extraction covers the first 30 pages and retains an excerpt; it is not OCR. Page retrieval returns bounded HTML text, not JavaScript-rendered pages. All stored sources indicate whether evidence came from an abstract, search snippet, page text, or PDF excerpt. Paywall bypass and fabricated experiments are not supported.

Allow outbound HTTPS to the two API hosts and to the publisher/web destinations required by your topic. Existing HTTPS proxies and CA trust are preserved. Keys are not needed for the offline demo. Missing keys disable the corresponding provider; other tools can still operate.

## Development

```bash
python3 -m unittest discover -s tests -v
```

Tests cover supplied materials, short time budgets, parallel execution, provider request formatting, JSON decisions, error recovery, repeated calls, watchdog cancellation, budgets, checkpoint restore, secret redaction, citation validation, shell limits, and real HTTP against a temporary local server. Live API/LLM tests are intentionally separate from deterministic tests.

Core modules: `engine.py` schedules agents; `tools.py` implements capabilities; `network.py` connects the local model and manages cancellable HTTP helpers; `storage.py` persists evidence/logs; `cli.py` provides terminal commands. `demo.py` is only a synthetic test transport.

Exit codes: `0` means the session recorded source-linked notes; `2` means it completed without usable findings; `1` means a configuration or execution error. Exit code zero does not certify paper quality.
