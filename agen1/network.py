import asyncio
import json
import os
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
        return json.loads(await self.request(url, **kwargs))


class LLM:
    def __init__(self, config, journal, http):
        self.config, self.journal, self.http = config, journal, http
        self.gate = asyncio.Semaphore(config.llm_concurrency)
        self.tokens = 0
        self.reserved = 0

    async def chat(self, agent, messages, *, final=False):
        async with self.gate:
            estimate = sum(len(m["content"]) for m in messages) // 4 + self.config.max_output_tokens
            # Keep the last 20% of the token budget for synthesis.
            ceiling = self.config.max_tokens if final else int(self.config.max_tokens * .8)
            if self.tokens + self.reserved + estimate > ceiling:
                raise RuntimeError("LLM token budget exhausted; end research and synthesize")
            self.reserved += estimate
            try:
                self.journal.event(agent, "llm_request", f"Requesting {self.config.model}", messages=messages)
                for attempt in range(3):
                    try:
                        data = await self.http.json(self.config.base_url.rstrip("/") + "/chat/completions",
                            headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + os.environ["LOCAL_LLM_API_KEY"]} if os.getenv("LOCAL_LLM_API_KEY") else {})},
                            body={"model": self.config.model, "messages": messages, "temperature": .2,
                                  "max_tokens": self.config.max_output_tokens, "stream": False})
                        content = data["choices"][0]["message"]["content"]
                        if not isinstance(content, str) or not content.strip():
                            raise ValueError("Model returned empty/non-text content")
                        usage = data.get("usage", {}).get("total_tokens")
                        self.tokens += int(usage) if isinstance(usage, (int, float)) and usage > 0 else sum(len(m["content"]) for m in messages) // 4 + len(content) // 4
                        self.journal.event(agent, "llm_response", "Received model response", content=content, total_tokens=self.tokens)
                        return content
                    except HTTPError as e:
                        self.journal.event(agent, "retry", str(e), attempt=attempt + 1)
                        if not e.retryable or attempt == 2:
                            raise
                        await asyncio.sleep(max(e.retry_after, 2 ** attempt))
            finally:
                self.reserved -= estimate


def parse_action(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    result = json.loads(text)
    if not isinstance(result, dict) or not isinstance(result.get("tool"), str) or not isinstance(result.get("args", {}), dict):
        raise ValueError('Expected {"tool": "name", "args": {...}}')
    return result
