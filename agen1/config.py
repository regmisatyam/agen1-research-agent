from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
import math


@dataclass
class Config:
    topic: str
    run_dir: Path
    hours: float = 1
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "qwen2.5:14b"
    workers: int = 3
    llm_concurrency: int = 2
    request_timeout: float = 60
    max_steps: int = 500
    max_tokens: int = 250_000
    max_output_tokens: int = 2048
    context_chars: int = 40_000
    supervisor_interval: float = 5
    stall_seconds: float = 180
    max_restarts: int = 5
    no_reasoning: bool = False
    allow_shell: bool = False
    full_access: bool = False
    verbose: bool = False
    demo: bool = False
    proposal: list[Path] = field(default_factory=list)
    material: list[Path] = field(default_factory=list)

    def validate(self):
        if not self.topic.strip():
            raise ValueError("A research topic is required")
        for name in ("hours", "request_timeout", "supervisor_interval", "stall_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("workers", "llm_concurrency", "max_steps", "max_tokens", "max_output_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_restarts < 0:
            raise ValueError("max_restarts must not be negative")
        if self.context_chars < 8000:
            raise ValueError("context_chars must be at least 8000")
        if self.workers > 16:
            raise ValueError("At most 16 workers are supported")
        if urlsplit(self.base_url).scheme not in ("http", "https"):
            raise ValueError("LLM URL must use http or https")
        if self.full_access and not self.allow_shell:
            raise ValueError("--full-access requires --allow-shell")
