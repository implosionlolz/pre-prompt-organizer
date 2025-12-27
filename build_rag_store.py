from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss  # type: ignore
import hashlib
import numpy as np

from local_rag import EmbeddingCache, chunk_rounds, l2_normalize, parse_md_session, to_rounds
from embedding_client import create_client, embed_texts


def split_text(text: str, *, max_chars: int, overlap_chars: int) -> List[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= max_chars:
        overlap_chars = max_chars // 5

    t = text.strip()
    if len(t) <= max_chars:
        return [t]

    parts: List[str] = []
    step = max(1, max_chars - overlap_chars)
    start = 0
    while start < len(t):
        end = min(len(t), start + max_chars)
        # Prefer splitting at a newline near the end for readability.
        if end < len(t):
            window_start = max(start + 1, end - 400)
            nl = t.rfind("\n", window_start, end)
            if nl != -1 and (nl - start) > (max_chars // 2):
                end = nl
        part = t[start:end].strip()
        if part:
            parts.append(part)
        if end >= len(t):
            break
        start = end - overlap_chars
        if start < 0:
            start = 0
    return parts


def iter_md_files(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")
    files = sorted([p for p in input_dir.glob("*.md") if p.is_file() and p.name.lower() != "index.md"])
    return files


def build_store(
    *,
    input_dir: Path,
    out_dir: Path,
    embedding_model: str,
    chunk_size_rounds: int,
    overlap_rounds: int,
    max_files: int | None,
) -> None:
    t0 = time.time()
    client = create_client()

    out_dir.mkdir(parents=True, exist_ok=True)
    cache = EmbeddingCache(out_dir / "embeddings.sqlite3")

    try:
        md_files = iter_md_files(input_dir)
        if max_files is not None:
            md_files = md_files[: max_files]

        max_chunk_chars = int(os.getenv("RAG_MAX_CHUNK_CHARS") or "6000")
        chunk_overlap_chars = int(os.getenv("RAG_CHUNK_OVERLAP_CHARS") or "300")

        all_chunks: List[Dict[str, Any]] = []
        for md_path in md_files:
            raw = md_path.read_text(encoding="utf-8", errors="replace")
            header_meta, messages = parse_md_session(raw)
            rounds = to_rounds(messages)
            chunks = chunk_rounds(rounds, chunk_rounds=chunk_size_rounds, overlap_rounds=overlap_rounds)
            rel = str(md_path.relative_to(input_dir))
            for c in chunks:
                text = str(c["text"])
                parts = split_text(text, max_chars=max_chunk_chars, overlap_chars=chunk_overlap_chars)
                for part_idx, part in enumerate(parts, start=1):
                    all_chunks.append(
                        {
                            "source_path": rel,
                            "title": header_meta.get("title") or md_path.stem,
                            "session_id": header_meta.get("session_id"),
                            "round_start": int(c["round_start"]),
                            "round_end": int(c["round_end"]),
                            "subchunk_index": part_idx if len(parts) > 1 else 1,
                            "subchunk_total": len(parts),
                            "text": part,
                        }
                    )

        if not all_chunks:
            raise RuntimeError("No chunks produced. Check your input directory and parser.")

        texts = [c["text"] for c in all_chunks]
        hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]

        cached = cache.get_many(hashes, model=embedding_model)
        vectors: List[Optional[np.ndarray]] = []
        missing: List[Tuple[int, str]] = []
        for i, h in enumerate(hashes):
            vec = cached.get(h)
            if vec is None:
                missing.append((i, h))
                vectors.append(None)
            else:
                vectors.append(vec)

        # Some providers enforce a max total tokens per request; batch by total characters to avoid 400 errors.
        max_items_per_request = 64
        max_chars_per_request = int(os.getenv("RAG_EMBEDDING_MAX_CHARS") or "120000")

        cursor = 0
        while cursor < len(missing):
            batch: List[Tuple[int, str]] = []
            total_chars = 0
            while cursor < len(missing) and len(batch) < max_items_per_request:
                i, h = missing[cursor]
                t = texts[i]
                if batch and (total_chars + len(t)) > max_chars_per_request:
                    break
                batch.append((i, h))
                total_chars += len(t)
                cursor += 1

            batch_texts = [texts[i] for i, _ in batch]
            batch_vecs = embed_texts(client, batch_texts, model=embedding_model)
            to_cache: List[Tuple[str, str, np.ndarray]] = []
            for (i, h), vec in zip(batch, batch_vecs):
                vectors[i] = vec
                to_cache.append((h, embedding_model, vec))
            cache.put_many(to_cache)

        if any(v is None for v in vectors):
            raise RuntimeError("Embedding build failed: missing vectors remain after embedding.")
        mat = np.vstack([np.asarray(v, dtype=np.float32) for v in vectors if v is not None])
        mat = l2_normalize(mat)
        dim = int(mat.shape[1])

        index = faiss.IndexFlatIP(dim)
        index.add(mat)

        faiss.write_index(index, str(out_dir / "index.faiss"))

        meta_path = out_dir / "meta.jsonl"
        with meta_path.open("w", encoding="utf-8") as f:
            for idx, chunk in enumerate(all_chunks):
                rec = {
                    "id": idx,
                    **chunk,
                    "char_len": len(chunk["text"]),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        config = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_dir": str(input_dir),
            "file_count": len(md_files),
            "chunk_count": len(all_chunks),
            "chunk_size_rounds": chunk_size_rounds,
            "overlap_rounds": overlap_rounds,
            "max_chunk_chars": max_chunk_chars,
            "chunk_overlap_chars": chunk_overlap_chars,
            "embedding_model": embedding_model,
            "dim": dim,
            "normalize": True,
            "faiss_index": "IndexFlatIP",
        }
        (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        elapsed = round(time.time() - t0, 2)
        print(f"✅ 建庫完成：{out_dir}")
        print(f"- 檔案數：{len(md_files)}")
        print(f"- chunk 數：{len(all_chunks)}")
        print(f"- embedding 模型：{embedding_model}")
        print(f"- 向量維度：{dim}")
        print(f"- 耗時：{elapsed}s")
    finally:
        cache.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local Markdown RAG store (FAISS + metadata).")
    parser.add_argument("--input-dir", default="past_chat_MD/md_sessions", help="Directory containing Markdown sessions.")
    parser.add_argument("--out-dir", default="rag_store", help="Output directory to write the store.")
    parser.add_argument("--embedding-model", default=os.getenv("RAG_EMBEDDING_MODEL") or "text-embedding-3-large")
    parser.add_argument("--chunk-rounds", type=int, default=10, help="Rounds per chunk (user+assistant).")
    parser.add_argument("--overlap-rounds", type=int, default=2, help="Overlap rounds between chunks.")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files (0 = no limit).")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    max_files = None if int(args.max_files) <= 0 else int(args.max_files)

    build_store(
        input_dir=input_dir,
        out_dir=out_dir,
        embedding_model=str(args.embedding_model),
        chunk_size_rounds=int(args.chunk_rounds),
        overlap_rounds=int(args.overlap_rounds),
        max_files=max_files,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

