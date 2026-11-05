# Run Agen1 with your existing vLLM server

Run these commands on your own Linux machine (or WSL2), from the extracted `agen1-research` source directory. Agen1 uses vLLM's OpenAI-compatible HTTP API; it does not need vLLM installed in its Python environment. Your existing model server can keep running separately.

## 1. Install the research application

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[pdf]'
python -m agen1 --help
```

Python 3.11+ is required. The `pdf` extra enables reading supplied PDFs and extracting downloaded paper text. TXT, Markdown, CSV, TSV, JSON, JSONL, HTML, and DOCX need no additional packages. Native Windows is not supported; use WSL2 for Bash, process signals, and file locking.

## 2. Set your endpoint, model ID, and keys

vLLM commonly serves at port 8000. Replace the URL if yours differs. The URL must include `/v1` and must not end in `/chat/completions`.

```bash
export AGEN1_BASE_URL='http://127.0.0.1:8000/v1'
export AGEN1_MODEL='YOUR_SERVED_MODEL_ID'
```

`AGEN1_BASE_URL` and `AGEN1_MODEL` are conveniences used by the commands below; the application reads them through the explicit CLI arguments. The application defaults still target Ollama unless you pass these arguments.

Use the exact model `id` returned by `GET /v1/models`, typically your model path/repository name or the alias selected with vLLM's `--served-model-name`. An existing local model is enough; no OpenAI account/key is needed.

If your vLLM server requires authentication, set its key as `LOCAL_LLM_API_KEY`. For Bash, these prompts keep keys out of shell history and terminal output:

```bash
read -r -s -p 'Semantic Scholar API key: ' SEMANTIC_SCHOLAR_API_KEY; printf '\n'
export SEMANTIC_SCHOLAR_API_KEY
read -r -s -p 'Brave Search API key: ' BRAVE_SEARCH_API_KEY; printf '\n'
export BRAVE_SEARCH_API_KEY
# Only if your vLLM endpoint requires a key:
# read -r -s -p 'vLLM API key: ' LOCAL_LLM_API_KEY; printf '\n'
# export LOCAL_LLM_API_KEY
```

`.env.example` lists the credential names. The application does not automatically load `.env` files. Do not put keys in proposals or commit them to Git.

Check connectivity and the served model name:

```bash
python -m agen1 doctor --base-url "$AGEN1_BASE_URL" --model "$AGEN1_MODEL"
```

This lists the model IDs even if the requested name is wrong. Update `AGEN1_MODEL` to match and rerun. `doctor` checks the model-list endpoint and variable presence; it does not run a generation or validate the research API keys remotely.

## 3. First live research run

```bash
python -m agen1 run "Why does ice float in liquid water?" \
  --base-url "$AGEN1_BASE_URL" --model "$AGEN1_MODEL" \
  --seconds 180 --workers 3 --llm-concurrency 1 \
  --request-timeout 120 --stall-seconds 600 \
  --context-chars 16000 --max-output-tokens 1024 \
  --run-dir runs/vllm-first --verbose
```

This command uses your real model and research providers: it does **not** use `--demo`. Each agent's conversation, tool call, result, and errors are displayed in the terminal. One concurrent generation is a conservative starting point for GPU memory use; the three researchers still run concurrently and queue model requests. Increase `--llm-concurrency` to 2 if your server handles the load comfortably.

Inspect the result:

```bash
python -m agen1 status runs/vllm-first
cat runs/vllm-first/paper.md
cat runs/vllm-first/activity.md
```

Successful connectivity alone does not establish research quality. Review citations, evidence types, errors, and the paper's limitations. A short run may finish before collecting findings; it still saves a fallback paper and logs.

## 4. Compare a few time limits

Run the same question in separate directories so each test starts without earlier evidence:

```bash
AGEN1_TEST_TAG=$(date -u +%Y%m%dT%H%M%SZ)
for seconds in 60 180 600; do
  python -m agen1 run "Why does ice float in liquid water?" \
    --base-url "$AGEN1_BASE_URL" --model "$AGEN1_MODEL" \
    --seconds "$seconds" --workers 3 --llm-concurrency 1 \
    --request-timeout 120 --stall-seconds 600 \
    --context-chars 16000 --max-output-tokens 1024 \
    --run-dir "runs/vllm-${AGEN1_TEST_TAG}-${seconds}s" --verbose
  result=$?
  printf 'Budget: %ss; exit code: %s\n' "$seconds" "$result"
done
```

Exit code `0` means source-linked notes were recorded, `2` means no usable findings were recorded, and `1` means a setup/execution error. None certifies scientific correctness. The time limit is a maximum: agents can finish early. Writing reserves 20% of the duration, capped at five minutes; a slow model may need a longer run to complete synthesis. Imports/startup occur before the timed research session, and final file writes/process cleanup can add a little overhead.

## 5. Your proposal, documents, and a longer run

Replace the example paths with existing local files. Repeat `--proposal` and `--material` for additional files, or omit them if no documents are needed.

```bash
python -m agen1 run "Your research question" \
  --base-url "$AGEN1_BASE_URL" --model "$AGEN1_MODEL" \
  --proposal /path/to/proposal.md \
  --material /path/to/reference.pdf \
  --material /path/to/observations.csv \
  --hours 4 --workers 3 --llm-concurrency 1 \
  --request-timeout 120 --stall-seconds 600 \
  --context-chars 16000 --max-output-tokens 2048 \
  --max-steps 1000 \
  --run-dir runs/my-research --verbose
```

Set `--context-limit` to the `max_model_len` vLLM serves (default 100000). Agen1 keeps every request, prompt and output together, under that limit and compacts older context to do so. `--context-chars` is a softer character budget for the evidence and history in each researcher request; increase it when researchers need more context per step, at the cost of slower steps. `--max-tokens` is separate: an optional total across the whole session, off by default.

`Ctrl+C` asks the runner to stop work and write a final draft. To add documents later, stop the run first:

```bash
python -m agen1 add-material --run-dir runs/my-research --material /path/to/more-notes.txt
python -m agen1 run --run-dir runs/my-research --hours 2 \
  --base-url "$AGEN1_BASE_URL" --model "$AGEN1_MODEL" \
  --workers 3 --llm-concurrency 1 --request-timeout 120 --stall-seconds 600 \
  --context-chars 16000 --max-output-tokens 2048 --verbose
```

Resume preserves sources, documents, and notes, and starts a new plan/time budget. Repeat the settings you want to use; model and permission options are not automatically restored from a prior command.

To authorize autonomous Bash commands with your OS user's permissions, add `--allow-shell`. Add `--full-access` too if the file tools should access paths outside the run workspace. An enabled shell is not sandboxed. These options are not needed merely to import explicitly supplied documents.

## vLLM troubleshooting

| Symptom | Check/action |
| --- | --- |
| Connection refused | Confirm vLLM is running and use its actual host/port. `127.0.0.1` refers to the machine running Agen1; a remote server needs its reachable address. |
| HTTP 404 | Base URL should end in `/v1`; use a model ID from `/v1/models`. |
| HTTP 401/403 from vLLM | Set `LOCAL_LLM_API_KEY` if the server requires it, and check access configuration. |
| Chat-template error | Use an instruct/chat model with a working chat template configured in vLLM. Agen1 calls `/v1/chat/completions`. |
| Invalid JSON decisions | Use a model that follows JSON instructions. The client consumes ordinary assistant `content` containing one JSON action; it does not consume native `tool_calls` or `reasoning_content`. Configure thinking/reasoning behavior appropriately for your model. |
| `Model returned empty/non-text content` | A reasoning model probably used its whole output budget: check `reasoning_chars` and `finish_reason` in the `llm_retry` events. Agen1 retries with reasoning disabled through `chat_template_kwargs`, and also does so after a timeout. Pass `--no-reasoning` to start that way. If your model cannot disable reasoning, raise `--max-output-tokens` and `--request-timeout`. |
| Context-length error | Agen1 retries shorter and adopts the limit vLLM reports, without turning off streaming. If `compaction` events recur in `events.jsonl`, set `--context-limit` to the served `max_model_len`. |
| Workers stop with `LLM token budget reached` | `--max-tokens` was set and 80% of it is spent. This is the session total, not the model's window. Raise it, or omit it to let the time budget decide. |
| Slow generation or GPU exhaustion | Keep `--llm-concurrency 1`; lower output/context sizes. Increase request/stall timeouts and the run duration for slow hardware. |
| Search errors | Check the relevant key, network access, quota, and the error in `events.jsonl`. Semantic Scholar and Brave require separate provider credentials. |
| Paper is a fallback | Inspect `summary.json` and `events.jsonl`: insufficient evidence, an exhausted time budget or `--max-tokens` cap, invalid citations, or model errors can trigger a fallback. |

Agen1 sends standard streaming chat requests with `model`, `messages`, `temperature`, `max_tokens`, and `stream: true`, and falls back to non-streaming if the server rejects that. Native tool-call parsing flags in vLLM are not required by this client; its JSON action protocol is handled by Agen1.

## Verified before handoff

The automated suite passed 33 tests, including actual HTTP against a temporary local server, PDF extraction, material import/resume, parallel workers, error recovery, supervision, and deadline cancellation. Offline runs with supplied materials at 0.01, 1, 3, and 6 seconds produced the expected drafts or explicit fallback. Your live vLLM model, API credentials, and long-duration research quality remain for you to validate.

```bash
python -m unittest discover -s tests -v
```
