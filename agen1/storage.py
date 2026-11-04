import datetime as dt
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import re


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


class Journal:
    def __init__(self, directory, verbose=False, quiet=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.verbose, self.quiet = verbose, quiet
        self.recent = deque(maxlen=100)
        # Very short values are placeholders such as "none"; replacing every
        # occurrence of one would corrupt evidence text and the JSON logs.
        self.secrets = [v for k, v in os.environ.items()
                        if any(x in k.upper() for x in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")) and len(v) >= 8]
        self.path = self.directory / "events.jsonl"
        self.activity = self.directory / "activity.md"
        if not self.activity.exists():
            self.activity.write_text("# Research activity\n\nChronological, human-readable record of actions and outcomes.\n", encoding="utf-8")

    def clean(self, value):
        text = str(value)
        for secret in self.secrets:
            text = text.replace(secret, "[REDACTED]")
        text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]+", r"\1[REDACTED]", text)
        return text

    def event(self, agent, kind, summary, **data):
        now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        safe = json.loads(self.clean(json.dumps({"time": now, "agent": agent, "kind": kind,
                                               "summary": summary, "data": data}, ensure_ascii=False)))
        self.recent.append(safe)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe, ensure_ascii=False) + "\n")
        with self.activity.open("a", encoding="utf-8") as f:
            f.write(f"\n- **{now} · {agent} · {kind}** — {safe['summary']}\n")
        if not self.quiet:
            # Prevent tool/model output from injecting terminal escape sequences.
            line = f"[{now}] {agent:14} {kind:15} {safe['summary']}"
            if self.verbose and data:
                line += "\n" + json.dumps(safe["data"], indent=2, ensure_ascii=False)
            print(re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "?", line), flush=True)


class Store:
    def __init__(self, directory, topic, journal, new_session=True):
        self.directory = Path(directory)
        self.journal = journal
        self.path = self.directory / "state.json"
        self.state = {"topic": topic, "sources": {}, "notes": [], "sessions": 0}
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
            if self.state["topic"] != topic:
                raise ValueError("Run directory belongs to a different topic")
        if new_session:
            self.state["sessions"] += 1
        self.save()

    def save(self):
        atomic_json(self.path, self.state)
        atomic_json(self.directory / "sources.json", self.state["sources"])

    def source(self, title, url, text, provider, **metadata):
        identity = metadata.get("paper_id") or url or title
        sid = "S" + hashlib.sha256(identity.encode()).hexdigest()[:10]
        is_new = sid not in self.state["sources"]
        incoming = json.loads(self.journal.clean(json.dumps({
            "id": sid, "title": title, "url": url, "text": text[:16000], "provider": provider, **metadata})))
        old = self.state["sources"].get(sid)
        ranks = {"search_snippet": 0, "abstract": 1, "page_text": 2, "pdf_excerpt": 3}
        old_rank = ranks.get(old.get("evidence_type"), 0) if old else -1
        new_rank = ranks.get(incoming.get("evidence_type"), 0)
        if old_rank > new_rank:
            incoming["text"], incoming["evidence_type"] = old["text"], old["evidence_type"]
        # Fetched pages are titled by URL; keep the real title from the search result.
        if old and title == url:
            incoming["title"] = old["title"]
        self.state["sources"][sid] = incoming
        self.save()
        # Stronger evidence for a known source is progress, just like a new source.
        return sid, is_new or new_rank > old_rank

    def note(self, text, citations, agent):
        unknown = set(citations) - self.state["sources"].keys()
        if unknown:
            raise ValueError(f"Unknown citations: {sorted(unknown)}")
        if not text.strip() or not citations:
            raise ValueError("Notes require nonempty text and at least one retrieved source ID")
        normalized = " ".join(text.lower().split())
        if any(" ".join(n["text"].lower().split()) == normalized for n in self.state["notes"]):
            return False
        self.state["notes"].append({"text": self.journal.clean(text[:8000]), "citations": citations, "agent": agent})
        self.save()
        return True

    def mark_read(self, source_id):
        """Record that the model read a source, even if it found nothing relevant.

        Verbatim excerpts only stand in for that reading, so they are removed."""
        self.state["notes"] = [n for n in self.state["notes"] if not (n.get("agent") == "direct-excerpt" and source_id in n["citations"])]
        self.state.setdefault("read_sources", []).append(source_id)
        self.save()

    def evidence(self, limit=24000, max_notes=20):
        """Bounded context with space reserved for supplied proposal/reference excerpts."""
        all_sources = list(self.state["sources"].values())
        materials = sorted((s for s in all_sources if s.get("provider") == "user"),
                           key=lambda s: s.get("material_role") != "proposal")
        compact = []
        budget = min(limit // 2, 8000)
        for source in materials:
            item = {k: source[k] for k in ("id", "title", "material_role", "evidence_type", "content_chars", "truncated")}
            item["text"] = source["text"][:min(2000, max(200, budget // max(1, len(materials)) - 250))]
            if len(json.dumps(compact + [item])) > budget:
                break
            compact.append(item)
        notes = self.state["notes"] if max_notes is None else self.state["notes"][-max_notes:]
        result = {"notes": notes, "materials": compact,
                  "materials_total": len(materials), "materials_omitted": len(materials) - len(compact),
                  "sources": [{k: v for k, v in s.items() if k != "text"} | {"text": s["text"][:1200]}
                              for s in all_sources if s.get("provider") != "user"]}
        while len(json.dumps(result)) > limit and result["sources"]:
            result["sources"].pop()
        while len(json.dumps(result)) > limit and result["notes"]:
            result["notes"].pop(0)
        return result
