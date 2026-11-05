import asyncio
import json
import os
import re
from urllib.parse import urlsplit


class HTTPError(RuntimeError):
    def __init__(self, message, retryable=False, retry_after=0):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


class HTTP:
    """Cancellable HTTP with system/proxy CA trust and a hard request deadline."""
    def __init__(self, timeout=60):
        self.timeout = timeout

    async def request(self, url, *, headers=None, body=None, max_bytes=2_000_000):
        if urlsplit(url).scheme not in ("http", "https"):
            raise ValueError("Only HTTP(S) URLs are supported")
        import base64
        import sys
        from pathlib import Path
        # Child isolation lets cancellation terminate even slow-drip socket reads.
        proc = await asyncio.create_subprocess_exec(sys.executable, str(Path(__file__).with_name("http_worker.py")),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            payload = json.dumps({"url": url, "headers": headers or {}, "body": body,
                                  "timeout": self.timeout, "max_bytes": max_bytes}).encode()
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(payload), timeout=self.timeout)
            except TimeoutError as e:
                raise HTTPError(f"Request to {urlsplit(url).hostname} exceeded {self.timeout}s", True) from e
            if proc.returncode:
                raise HTTPError("HTTP helper failed; check URL and configured CA trust", False)
            result = json.loads(stdout)
            if "error" in result:
                raise HTTPError(result["error"], result.get("retryable", False), result.get("retry_after", 0))
            return base64.b64decode(result["data"])
        finally:
            if proc.returncode is None:
                proc.kill()
            await proc.wait()

    async def json(self, url, **kwargs):
        try:
            return json.loads(await self.request(url, **kwargs))
        except json.JSONDecodeError as e:
            # Usually a cut-off body or a proxy error page, so a retry can succeed.
            raise HTTPError(f"Invalid JSON from {urlsplit(url).hostname}: {e}", True) from e


# Servers word this differently; vLLM says "maximum context length is N tokens".
OVERFLOW = re.compile(r"context.{0,20}(length|size|window)|max_model_len|reduce the length|too many tokens|prompt is too long", re.I)
REPORTED_LIMIT = re.compile(r"maximum context length is (\d+)", re.I)
REPORTED_PROMPT = re.compile(r"(\d+)(?: in the messages| input tokens)", re.I)
# Stay a little under the window: token counts here are estimates.
WINDOW_USE = .95
CUT_MARK = "\n...[cut to fit the model's context limit]"


class LLM:
    def __init__(self, config, journal, http):
        self.config, self.journal, self.http = config, journal, http
        self.gate = asyncio.Semaphore(config.llm_concurrency)
        self.tokens = 0
        self.reserved = 0
        self.context_limit = config.context_limit
        # Cautious, because JSON, URLs, and IDs tokenize worse than prose.
        self.chars_per_token = 3.0
        self.peak_prompt_tokens = 0
        self.trims = 0
        # Reasoning models can spend the whole output budget before answering.
        # Once seen, ask the server to skip reasoning for the rest of the session.
        self.disable_thinking = config.no_reasoning
        self.template_kwargs_supported = True
        # Streaming keeps bytes flowing, because some network paths stall silent
        # responses after a few seconds, and it preserves partial text.
        self.streaming = True

    def prompt_tokens(self, messages):
        return int(sum(len(m["content"]) for m in messages) / self.chars_per_token) + 4 * len(messages)

    def output_budget(self, final):
        # A paper needs more room than a single tool decision.
        wanted = max(self.config.max_output_tokens, 4096) if final else self.config.max_output_tokens
        return min(wanted, self.context_limit // 2)

    def session_ceiling(self, final):
        # Keep the last 20% of a session cap for synthesis.
        return self.config.max_tokens if final else int(self.config.max_tokens * .8)

    def prompt_chars(self, final=False):
        """Prompt characters one request can carry within the window and any session cap."""
        chars = (int(self.context_limit * WINDOW_USE) - self.output_budget(final)) * self.chars_per_token
        if self.config.max_tokens:
            left = self.session_ceiling(final) - self.tokens - self.reserved - self.config.max_output_tokens
            chars = min(chars, left * 4)
        return max(0, int(chars))

    def fit(self, agent, messages, output_tokens):
        """Last line of defense: cut the longest messages so the request fits the window."""
        excess = self.prompt_tokens(messages) + output_tokens - int(self.context_limit * WINDOW_USE)
        if excess <= 0:
            return messages
        fitted = [dict(m) for m in messages]
        cut = int(excess * self.chars_per_token) + 1
        before = cut
        while cut > 0:
            longest = max((m for m in fitted if m["role"] != "system"), key=lambda m: len(m["content"]), default=None)
            # Keep enough of every message for it to still make sense.
            removable = len(longest["content"]) - 200 if longest else 0
            if removable <= 0:
                break
            taken = min(cut, removable)
            longest["content"] = longest["content"][:-(taken + len(CUT_MARK))] + CUT_MARK
            cut -= taken
        self.trims += 1
        self.journal.event(agent, "compaction", f"Cut {before - cut} prompt characters to fit the {self.context_limit}-token window")
        return fitted

    def tighten(self, error, messages, output_tokens):
        """The server measured this prompt; trust that over the estimate from now on."""
        text = str(error)
        if limit := REPORTED_LIMIT.search(text):
            self.context_limit = min(self.context_limit, int(limit[1]))
        if measured := REPORTED_PROMPT.search(text):
            ratio = sum(len(m["content"]) for m in messages) / max(1, int(measured[1]))
            self.chars_per_token = max(1, min(self.chars_per_token, ratio))
        else:
            # The size is unknown, so assume this request has to shrink by a quarter.
            shrunk = int((self.prompt_tokens(messages) + output_tokens) * .75)
            self.context_limit = max(1000, min(self.context_limit, shrunk))

    async def chat(self, agent, messages, *, final=False):
        async with self.gate:
            messages = self.fit(agent, messages, self.output_budget(final))
            estimate = sum(len(m["content"]) for m in messages) // 4 + self.config.max_output_tokens
            if self.config.max_tokens and self.tokens + self.reserved + estimate > self.session_ceiling(final):
                raise RuntimeError("LLM token budget exhausted; end research and synthesize")
            self.reserved += estimate
            try:
                self.journal.event(agent, "llm_request", f"Requesting {self.config.model}", messages=messages)
                return await self.generate(agent, messages, final)
            finally:
                self.reserved -= estimate

    async def generate(self, agent, messages, final):
        base_output = output_tokens = self.output_budget(final)
        diagnosis = "no response"
        partial = ""
        for attempt in range(3):
            # Retries raise the output budget, which must not push past the window.
            room = int(self.context_limit * WINDOW_USE) - self.prompt_tokens(messages)
            body = {"model": self.config.model, "messages": messages, "temperature": .2,
                    "max_tokens": max(1, min(output_tokens, room)), "stream": self.streaming}
            if self.streaming:
                body["stream_options"] = {"include_usage": True}
            if self.disable_thinking and self.template_kwargs_supported:
                body["chat_template_kwargs"] = {"enable_thinking": False}
            try:
                data = await self.http.json(self.config.base_url.rstrip("/") + "/chat/completions",
                    headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + os.environ["LOCAL_LLM_API_KEY"]} if os.getenv("LOCAL_LLM_API_KEY") else {})},
                    body=body)
            except HTTPError as e:
                self.journal.event(agent, "retry", str(e), attempt=attempt + 1)
                # An overlong prompt is also an HTTP 400, and removing request features
                # would not cure it, so check for it before dropping any of them.
                overflow = OVERFLOW.search(str(e))
                if overflow and attempt < 2:
                    self.tighten(e, messages, base_output)
                    # A smaller window can also shrink what is left for the reply.
                    base_output = output_tokens = min(output_tokens, self.output_budget(final))
                    messages = self.fit(agent, messages, base_output)
                    continue
                # Drop optional request features one at a time if the server rejects them.
                if not overflow and "HTTP 400" in str(e) and "chat_template_kwargs" in body:
                    self.template_kwargs_supported = False
                    continue
                if not overflow and "HTTP 400" in str(e) and self.streaming:
                    self.streaming = False
                    continue
                # Reasoning can outlast the request timeout, so retry without it.
                if "exceeded" in str(e) or "TimeoutError" in str(e):
                    self.disable_thinking = True
                if not e.retryable or attempt == 2:
                    if final and partial:
                        return self.accept(agent, partial, "interrupted", final)
                    raise
                await asyncio.sleep(max(e.retry_after, 2 ** attempt))
                continue
            choice = data["choices"][0]
            message, finish_reason = choice["message"], choice.get("finish_reason")
            content = strip_reasoning(message.get("content"))
            usage = (data.get("usage") or {}).get("total_tokens")
            # Rejected replies still consumed server tokens, so count them too.
            self.tokens += int(usage) if isinstance(usage, (int, float)) and usage > 0 else sum(len(m["content"]) for m in messages) // 4 + len(content) // 4
            prompt_used = (data.get("usage") or {}).get("prompt_tokens")
            self.peak_prompt_tokens = max(self.peak_prompt_tokens, int(prompt_used) if isinstance(prompt_used, (int, float)) and prompt_used > 0 else self.prompt_tokens(messages))
            if content and finish_reason == "interrupted" and attempt < 2:
                partial = max(partial, content, key=len)
                self.journal.event(agent, "retry", f"Reply was cut off after {len(content)} characters", attempt=attempt + 1)
                continue
            if content:
                return self.accept(agent, max(partial, content, key=len), finish_reason, final)
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            diagnosis = f"finish_reason={finish_reason}, reasoning_chars={len(reasoning)}, max_tokens={output_tokens}"
            self.journal.event(agent, "llm_retry", f"Unusable model reply ({diagnosis}); retrying without reasoning and with more output tokens", attempt=attempt + 1)
            self.disable_thinking = True
            output_tokens *= 2
        if final and partial:
            return self.accept(agent, partial, "interrupted", final)
        raise ValueError(f"Model returned empty/non-text content ({diagnosis})")

    def accept(self, agent, content, finish_reason, final):
        if final and finish_reason in ("length", "interrupted"):
            content += "\n\n> **This draft was cut off before the model finished.** Increase --max-output-tokens or --request-timeout and resume to regenerate it."
        self.journal.event(agent, "llm_response", "Received model response", content=content, finish_reason=finish_reason, total_tokens=self.tokens)
        return content


JSON_START = re.compile(r"[\[{]")


def strip_reasoning(content):
    """Remove inline reasoning, including a block cut off by the output limit."""
    if not isinstance(content, str):
        return ""
    return re.sub(r"<think>.*?(</think>|\Z)", "", content, flags=re.S | re.I).strip()


def extract_json(text, accept=lambda value: isinstance(value, dict)):
    """Return the first acceptable JSON value, tolerating fences and surrounding prose."""
    text = strip_reasoning(text)
    decoder = json.JSONDecoder()
    position = 0
    while match := JSON_START.search(text, position):
        try:
            value, position = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            position = match.start() + 1
            continue
        if accept(value):
            return value
    raise ValueError("Reply contained no usable JSON")


ACTION_FORMAT = 'Reply with one JSON object, for example {"tool":"search_web","args":{"query":"your search terms"}}'


def is_action(value):
    return isinstance(value, dict) and ("tool" in value or "name" in value)


def normalize_action(item):
    """Accept the action shapes models really emit; arguments may sit beside "tool"."""
    if not is_action(item):
        raise ValueError(ACTION_FORMAT)
    fields = dict(item)
    tool = fields.pop("tool", None) or fields.pop("name", None)
    args = fields.pop("args", None)
    if args is None:
        args = fields.pop("arguments", None)
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            raise ValueError(ACTION_FORMAT) from None
    args = {} if args is None else args
    if not isinstance(tool, str) or not isinstance(args, dict):
        raise ValueError(ACTION_FORMAT)
    return {"tool": tool, "args": {**fields, **args}}


def parse_actions(text):
    """Return every action in a reply: a single object or an array of them."""
    try:
        value = extract_json(text, lambda v: is_action(v) or (isinstance(v, list) and any(is_action(x) for x in v)))
    except ValueError:
        raise ValueError(ACTION_FORMAT) from None
    return [normalize_action(item) for item in (value if isinstance(value, list) else [value])]


def parse_action(text):
    return parse_actions(text)[0]
