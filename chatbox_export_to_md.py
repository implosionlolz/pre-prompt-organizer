"""
Chatbox 匯出 JSON → Markdown（每個 session 一個檔案 + index）

為什麼要做這件事：你多半不是只想「轉格式」，而是想把對話變成可讀、可搜尋、可被後續工具再利用的資產。
所以這支工具刻意做成「零第三方依賴」並保留結構（role / timestamp / tool-call / OCR）。

用法（在專案根目錄）：

  python3 chatbox_export_to_md.py \
    --input past_chat_MD/chatbox-exported-data-2025-12-20.json \
    --output-dir past_chat_MD/md_sessions

輸出：
  - <output-dir>/*.md      每個 session 一個 Markdown
  - <output-dir>/index.md  總索引
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


INVALID_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00]')
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _safe_filename(name: str, max_len: int = 120) -> str:
    name = (name or "").strip()
    if not name:
        return "untitled"
    name = INVALID_FILENAME_RE.sub(" ", name)
    name = CONTROL_RE.sub("", name)
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) > max_len:
        name = name[: max_len - 1].rstrip() + "…"
    return name


def _format_ts(value: Any) -> Optional[str]:
    """把常見 timestamp 轉成 ISO8601（秒級）。"""
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            # ms epoch vs s epoch
            if value > 1e12:
                dt = datetime.fromtimestamp(value / 1000, tz=timezone.utc)
            elif value > 1e9:
                dt = datetime.fromtimestamp(value, tz=timezone.utc)
            else:
                return str(value)
            return dt.isoformat(timespec="seconds").replace("+00:00", "Z")
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            if s.endswith("Z"):
                dt = datetime.fromisoformat(s[:-1] + "+00:00")
                return dt.isoformat(timespec="seconds").replace("+00:00", "Z")
            dt = datetime.fromisoformat(s)
            return dt.isoformat(timespec="seconds")
    except Exception:
        return str(value)
    return str(value)


def _json_block(data: Any) -> str:
    return "```json\n" + json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n```"


def _quote_block(text: str, prefix: str = "> ") -> str:
    lines = (text or "").splitlines() or [""]
    return "\n".join(prefix + line for line in lines)


def _render_content_part(
    part: Dict[str, Any],
    *,
    include_images: bool,
    include_ocr: bool,
    include_tool_calls: bool,
) -> str:
    ptype = part.get("type")

    if ptype == "text":
        return str(part.get("text") or "")

    if ptype == "info":
        text = str(part.get("text") or "")
        return _quote_block(f"[info] {text}".strip())

    if ptype == "image":
        if not include_images:
            return ""
        storage_key = str(part.get("storageKey") or "").strip()
        ocr = str(part.get("ocrResult") or "").strip()

        chunks: List[str] = []
        chunks.append(_quote_block(f"[image] {storage_key}".strip() if storage_key else "[image]"))

        if include_ocr and ocr:
            chunks.append(
                "<details>\n"
                "<summary>OCR</summary>\n\n"
                "```text\n"
                f"{ocr}\n"
                "```\n\n"
                "</details>"
            )
        return "\n\n".join(chunks)

    if ptype == "tool-call":
        if not include_tool_calls:
            return ""

        tool_name = str(part.get("toolName") or "tool").strip()
        state = str(part.get("state") or "").strip()
        tool_call_id = str(part.get("toolCallId") or "").strip()
        args = part.get("args")
        result = part.get("result")

        summary_bits: List[str] = [tool_name]
        if state:
            summary_bits.append(state)
        if tool_call_id:
            summary_bits.append(tool_call_id)
        summary = " · ".join(summary_bits)

        body: List[str] = []
        if args is not None:
            body.append("**args**\n\n" + _json_block(args))
        if result is not None:
            body.append("**result**\n\n" + _json_block(result))

        inner = "\n\n".join(body) if body else "_(no payload)_"
        return (
            "<details>\n"
            f"<summary>tool-call: {summary}</summary>\n\n"
            f"{inner}\n\n"
            "</details>"
        )

    # 其他/未知：保留最小可追查資訊
    return _quote_block("[unknown_part] " + json.dumps(part, ensure_ascii=False)[:2000])


def _render_message(
    msg: Dict[str, Any],
    *,
    include_system: bool,
    include_images: bool,
    include_ocr: bool,
    include_tool_calls: bool,
) -> Optional[str]:
    role = str(msg.get("role") or "unknown").strip()
    if role == "system" and not include_system:
        return None

    ts = _format_ts(msg.get("timestamp"))
    header = f"### {role}" + (f" · {ts}" if ts else "")

    parts: List[str] = []
    cps = msg.get("contentParts")
    if isinstance(cps, list):
        for part in cps:
            if isinstance(part, dict):
                rendered = _render_content_part(
                    part,
                    include_images=include_images,
                    include_ocr=include_ocr,
                    include_tool_calls=include_tool_calls,
                ).strip()
                if rendered:
                    parts.append(rendered)
            else:
                parts.append(_quote_block("[non_dict_part] " + str(part)))

    if not parts:
        parts.append("_（此則訊息沒有可用內容）_")

    return header + "\n\n" + "\n\n".join(parts)


@dataclass(frozen=True)
class SessionMeta:
    session_id: str
    name: str
    type: Optional[str]
    starred: Optional[bool]


def _collect_sessions(root: Dict[str, Any]) -> List[SessionMeta]:
    sessions_list = root.get("chat-sessions-list")
    metas: List[SessionMeta] = []
    if isinstance(sessions_list, list):
        for item in sessions_list:
            if not isinstance(item, dict):
                continue
            sid = item.get("id")
            if not isinstance(sid, str) or not sid:
                continue
            metas.append(
                SessionMeta(
                    session_id=sid,
                    name=str(item.get("name") or sid),
                    type=str(item.get("type")) if item.get("type") is not None else None,
                    starred=bool(item["starred"]) if "starred" in item and item["starred"] is not None else None,
                )
            )
        return metas

    # fallback：沒有 sessions list 的話，就從 session:* keys 推
    for k in sorted(root.keys()):
        if isinstance(k, str) and k.startswith("session:"):
            sid = k.split("session:", 1)[1]
            metas.append(SessionMeta(session_id=sid, name=sid, type=None, starred=None))
    return metas


def _session_object(root: Dict[str, Any], session_id: str) -> Optional[Dict[str, Any]]:
    obj = root.get(f"session:{session_id}")
    return obj if isinstance(obj, dict) else None


def _session_time_range(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    stamps: List[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        s = _format_ts(m.get("timestamp"))
        if s:
            stamps.append(s)
    if not stamps:
        return None, None
    stamps.sort()
    return stamps[0], stamps[-1]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def convert(
    *,
    input_path: Path,
    output_dir: Path,
    include_system: bool,
    include_images: bool,
    include_ocr: bool,
    include_tool_calls: bool,
    limit: Optional[int],
) -> List[Tuple[SessionMeta, Path, int]]:
    root = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(root, dict):
        raise ValueError("Unexpected JSON shape: top-level is not an object.")

    exported_at = _format_ts(root.get("__exported_at"))
    metas = _collect_sessions(root)

    written: List[Tuple[SessionMeta, Path, int]] = []
    used_names: Dict[str, int] = {}

    for idx, meta in enumerate(metas, start=1):
        if limit is not None and idx > limit:
            break

        sess = _session_object(root, meta.session_id)
        if sess is None:
            continue

        messages_raw = sess.get("messages")
        messages: List[Dict[str, Any]] = [m for m in messages_raw if isinstance(m, dict)] if isinstance(messages_raw, list) else []

        provider = None
        model_id = None
        max_ctx = None
        settings = sess.get("settings")
        if isinstance(settings, dict):
            provider = settings.get("provider")
            model_id = settings.get("modelId")
            max_ctx = settings.get("maxContextMessageCount")

        t0, t1 = _session_time_range(messages)

        safe_title = _safe_filename(meta.name)
        base = f"{idx:03d}-{safe_title}-{meta.session_id[:8]}"
        n = used_names.get(base, 0)
        used_names[base] = n + 1
        if n:
            base = f"{base}-{n+1}"

        out_path = output_dir / f"{base}.md"

        lines: List[str] = []
        lines.append(f"## {meta.name}")
        lines.append("")
        lines.append(f"- session_id: `{meta.session_id}`")
        if exported_at:
            lines.append(f"- exported_at: `{exported_at}`")
        if meta.type is not None:
            lines.append(f"- type: `{meta.type}`")
        if meta.starred is not None:
            lines.append(f"- starred: `{meta.starred}`")
        if provider is not None or model_id is not None:
            lines.append(f"- model: `{provider or ''}` / `{model_id or ''}`".rstrip())
        if max_ctx is not None:
            lines.append(f"- maxContextMessageCount: `{max_ctx}`")
        if t0 or t1:
            lines.append(f"- time_range: `{t0 or ''}` → `{t1 or ''}`")
        lines.append("")
        lines.append("## Conversation")
        lines.append("")

        for m in messages:
            rendered = _render_message(
                m,
                include_system=include_system,
                include_images=include_images,
                include_ocr=include_ocr,
                include_tool_calls=include_tool_calls,
            )
            if rendered:
                lines.append(rendered)
                lines.append("")

        _write_text(out_path, "\n".join(lines).rstrip() + "\n")
        written.append((meta, out_path, len(messages)))

    # index
    idx_lines: List[str] = ["## Chatbox sessions index", ""]
    if exported_at:
        idx_lines.append(f"exported_at: `{exported_at}`")
        idx_lines.append("")
    idx_lines.append(f"total_sessions: `{len(written)}`")
    idx_lines.append("")
    for meta, path, msg_count in written:
        star = " ★" if meta.starred else ""
        idx_lines.append(f"- [{meta.name}](./{path.name}){star}  （{msg_count} messages）")
    _write_text(output_dir / "index.md", "\n".join(idx_lines).rstrip() + "\n")

    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert Chatbox exported JSON into Markdown files.")
    parser.add_argument("--input", required=True, type=Path, help="Path to Chatbox exported JSON")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write Markdown sessions into")
    parser.add_argument("--exclude-system", action="store_true", help="Drop system messages")
    parser.add_argument("--exclude-images", action="store_true", help="Drop image content parts")
    parser.add_argument("--exclude-ocr", action="store_true", help="Drop OCR text for images")
    parser.add_argument("--exclude-tool-calls", action="store_true", help="Drop tool-call content parts")
    parser.add_argument("--limit", type=int, default=None, help="Only export the first N sessions (quick verify)")

    args = parser.parse_args(argv)

    written = convert(
        input_path=args.input,
        output_dir=args.output_dir,
        include_system=not args.exclude_system,
        include_images=not args.exclude_images,
        include_ocr=not args.exclude_ocr,
        include_tool_calls=not args.exclude_tool_calls,
        limit=args.limit,
    )

    print(f"Wrote {len(written)} session markdown files to: {args.output_dir}")
    print(f"Index: {args.output_dir / 'index.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

