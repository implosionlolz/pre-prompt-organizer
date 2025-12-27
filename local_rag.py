from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss  # type: ignore
import numpy as np


ROLE_HEADER_RE = re.compile(r"^###\s+(system|user|assistant)\s+·\s+(.+?)\s*$", re.MULTILINE)
SESSION_ID_RE = re.compile(r"^- session_id:\s+`([^`]+)`\s*$", re.MULTILINE)
TITLE_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _looks_like_export_json_blob(text: str) -> bool:
    s = text.lstrip()
    if len(s) < 400:
        return False
    if not (s.startswith("{") or s.startswith("[")):
        return False
    # Heuristics for Chatbox export blobs: lots of quoted keys and timestamps.
    markers = ('"history"', '"models"', '"timestamp"', '"childrenIds"', '"modelName"', '"session_id"')
    return sum(1 for m in markers if m in s) >= 2


@dataclass(frozen=True)
class ParsedMessage:
    role: str
    timestamp: str
    content: str


def parse_md_session(md_text: str) -> Tuple[Dict[str, Any], List[ParsedMessage]]:
    """
    Parse a Chatbox-exported Markdown session produced by chatbox_export_to_md.py.

    Returns:
      - header metadata (best-effort)
      - ordered messages with role/timestamp/content
    """
    meta: Dict[str, Any] = {}
    m = SESSION_ID_RE.search(md_text)
    if m:
        meta["session_id"] = m.group(1)

    title_match = TITLE_RE.search(md_text)
    if title_match:
        meta["title"] = title_match.group(1).strip()

    matches = list(ROLE_HEADER_RE.finditer(md_text))
    messages: List[ParsedMessage] = []
    for idx, match in enumerate(matches):
        role = match.group(1).strip()
        timestamp = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(md_text)
        content = md_text[start:end].strip()
        if not content:
            continue
        # Keep as-is; downstream filters decide what to index.
        messages.append(ParsedMessage(role=role, timestamp=timestamp, content=content))

    return meta, messages


def to_rounds(messages: List[ParsedMessage]) -> List[Dict[str, Any]]:
    """
    Convert message stream to "rounds" (roughly: user + assistant answer(s)).
    System messages are excluded.
    """
    rounds: List[Dict[str, Any]] = []
    current_user: Optional[ParsedMessage] = None
    current_assistant: List[ParsedMessage] = []

    def flush_round() -> None:
        nonlocal current_user, current_assistant
        if current_user is None and not current_assistant:
            return
        user_text = "" if current_user is None else current_user.content.strip()
        assistant_text = "\n\n".join(m.content.strip() for m in current_assistant if m.content.strip()).strip()
        if user_text and _looks_like_export_json_blob(user_text):
            # Drop giant export payloads; they are noise for retrieval.
            user_text = ""
        if not user_text and not assistant_text:
            current_user = None
            current_assistant = []
            return
        rounds.append(
            {
                "user": user_text,
                "assistant": assistant_text,
                "user_ts": None if current_user is None else current_user.timestamp,
                "assistant_ts": None if not current_assistant else current_assistant[-1].timestamp,
            }
        )
        current_user = None
        current_assistant = []

    for msg in messages:
        if msg.role == "system":
            continue
        if msg.role == "user":
            flush_round()
            current_user = msg
            current_assistant = []
        elif msg.role == "assistant":
            if current_user is None:
                # Orphan assistant message; keep it as its own round.
                rounds.append({"user": "", "assistant": msg.content.strip(), "user_ts": None, "assistant_ts": msg.timestamp})
            else:
                if msg.content.strip():
                    current_assistant.append(msg)
        else:
            # Unknown role: ignore.
            continue

    flush_round()
    return rounds


def chunk_rounds(
    rounds: List[Dict[str, Any]],
    *,
    chunk_rounds: int = 10,
    overlap_rounds: int = 2,
) -> List[Dict[str, Any]]:
    if chunk_rounds <= 0:
        raise ValueError("chunk_rounds must be > 0")
    if overlap_rounds < 0:
        raise ValueError("overlap_rounds must be >= 0")
    if overlap_rounds >= chunk_rounds:
        raise ValueError("overlap_rounds must be < chunk_rounds")

    step = max(1, chunk_rounds - overlap_rounds)
    chunks: List[Dict[str, Any]] = []
    for start in range(0, len(rounds), step):
        end = min(len(rounds), start + chunk_rounds)
        window = rounds[start:end]
        lines: List[str] = []
        for r in window:
            user = (r.get("user") or "").strip()
            assistant = (r.get("assistant") or "").strip()
            if user:
                lines.append(f"你：{user}")
            if assistant:
                lines.append(f"助理：{assistant}")
        text = "\n\n".join(lines).strip()
        if not text:
            continue
        chunks.append(
            {
                "round_start": start + 1,
                "round_end": end,
                "text": text,
            }
        )
        if end >= len(rounds):
            break
    return chunks


class EmbeddingCache:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
              text_hash TEXT NOT NULL,
              model TEXT NOT NULL,
              dim INTEGER NOT NULL,
              vec BLOB NOT NULL,
              created_at REAL NOT NULL,
              PRIMARY KEY (text_hash, model)
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def get_many(self, keys: List[str], *, model: str) -> Dict[str, np.ndarray]:
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        rows = self._conn.execute(
            f"SELECT text_hash, dim, vec FROM embeddings WHERE model = ? AND text_hash IN ({placeholders})",
            [model, *keys],
        ).fetchall()
        out: Dict[str, np.ndarray] = {}
        for text_hash, dim, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.shape[0] != int(dim):
                continue
            out[str(text_hash)] = vec
        return out

    def put_many(self, items: List[Tuple[str, str, np.ndarray]]) -> None:
        if not items:
            return
        now = time.time()
        payload = []
        for text_hash, model, vec in items:
            v = np.asarray(vec, dtype=np.float32)
            payload.append((text_hash, model, int(v.shape[0]), v.tobytes(), now))
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings (text_hash, model, dim, vec, created_at) VALUES (?, ?, ?, ?, ?)",
            payload,
        )
        self._conn.commit()


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


@dataclass(frozen=True)
class RetrievalHit:
    id: int
    score: float
    meta: Dict[str, Any]


class LocalRagStore:
    def __init__(self, index: Any, meta: List[Dict[str, Any]], config: Dict[str, Any]):
        self.index = index
        self.meta = meta
        self.config = config

    @classmethod
    def load(cls, store_dir: Path) -> "LocalRagStore":
        index_path = store_dir / "index.faiss"
        meta_path = store_dir / "meta.jsonl"
        config_path = store_dir / "config.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata: {meta_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config: {config_path}")

        index = faiss.read_index(str(index_path))
        meta: List[Dict[str, Any]] = []
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                meta.append(json.loads(line))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(index=index, meta=meta, config=config)

    def search(self, query_vec: np.ndarray, *, top_k: int = 6) -> List[RetrievalHit]:
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        if self.config.get("normalize", True):
            q = l2_normalize(q)
        scores, ids = self.index.search(q, top_k)
        out: List[RetrievalHit] = []
        for rank in range(ids.shape[1]):
            idx = int(ids[0, rank])
            if idx < 0:
                continue
            score = float(scores[0, rank])
            meta = self.meta[idx] if 0 <= idx < len(self.meta) else {"id": idx}
            out.append(RetrievalHit(id=idx, score=score, meta=meta))
        return out

