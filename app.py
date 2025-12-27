from __future__ import annotations

import copy
import json
import logging
import os
import re
import secrets
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core import prepare
from embedding_client import create_client


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
LOG_DIR = ROOT / "logs"
SESSION_DEBUG_LOG_NAME = "debug.log"


def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("preprompt")
    logger.setLevel(logging.INFO)

    # 永遠不要寫全域 logs/debug.log（避免 session 混在一起）
    # 這裡保留一個 console handler，確保非 /api/prepare 的錯誤仍可在終端看到。
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
        logger.addHandler(sh)

    return logger


logger = _setup_logging()


def _iso_utc_z(dt_utc: datetime) -> str:
    """
    把 UTC datetime 輸出成 ISO8601 + Z（與既有 log 格式相容）。
    """
    s = dt_utc.isoformat(timespec="seconds")
    if s.endswith("+00:00"):
        s = s[: -6] + "Z"
    return s


def _log_filename_tz() -> str:
    """
    控制 session JSON 檔名時間戳用哪個時區。
    - utc（預設）：與既有行為一致，檔名/created_at 都是 UTC（Z）。
    - local：檔名使用本機時區（方便肉眼對照），JSON 仍會同時寫 UTC 與 local。
    """
    return (os.getenv("RAG_LOG_FILENAME_TZ") or "utc").strip().lower()


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    s = v.strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _redact_prepare_log_record(*, payload_dump: dict, result_dump: dict) -> tuple[dict, dict]:
    """
    開源/共享時避免「無意間把私密內容落盤」：
    - 不記錄 question / prompt / retrieved text / compressed summary / expand raw JSON / expand queries
    - 仍保留可用於除錯的結構、分數、來源、參數與統計資訊
    """
    pd = copy.deepcopy(payload_dump or {})
    if "question" in pd:
        q = pd.get("question") or ""
        pd["question_chars"] = len(q)
        pd["question"] = "<redacted>"

    rd = copy.deepcopy(result_dump or {})

    # result 內通常也會帶 question / prompt
    if "question" in rd:
        q = rd.get("question") or ""
        rd["question_chars"] = len(q)
        rd["question"] = "<redacted>"

    if "prompt" in rd:
        p = rd.get("prompt") or ""
        rd["prompt_chars"] = len(p)
        rd["prompt"] = "<redacted>"

    # 檢索片段：保留 meta（title/path/score），移除 text
    items = rd.get("retrieved_items")
    if isinstance(items, list):
        new_items = []
        for it in items:
            if not isinstance(it, dict):
                new_items.append(it)
                continue
            it2 = dict(it)
            if "text" in it2:
                t = it2.get("text") or ""
                it2["text_chars"] = len(t)
                it2["text"] = "<redacted>"
            new_items.append(it2)
        rd["retrieved_items"] = new_items

    # 壓縮後摘要與擴寫輸出：很容易包含敏感內容
    for k in [
        "retrieved_summary",
        "expand_raw_json",
        "expand_queries",
        "retrieval_queries_used",
    ]:
        if k in rd:
            rd[k] = "<redacted>"

    # tags 也可能泄漏題目意圖；保守起見一併打碼（仍保留有無啟用）
    if "expand_tags" in rd:
        rd["expand_tags"] = "<redacted>"

    return pd, rd


def _format_log_ts(dt: datetime) -> str:
    """
    檔名用的時間戳：YYYYmmdd-HHMMSS-ffffff，若帶時區偏移則附加一段可攜 suffix。
    注意：檔名避免使用 ":" 以確保跨平台相容。
    """
    base = dt.strftime("%Y%m%d-%H%M%S-%f")
    off = dt.strftime("%z")  # e.g. +0800
    if off and off not in {"+0000", "-0000"}:
        # 避免檔名含 "+" "-" 在某些工具顯示時造成誤讀：改成 p/m 前綴
        safe = off.replace("+", "p").replace("-", "m")
        return f"{base}-{safe}"
    return base


class PrepareRequest(BaseModel):
    question: str = Field(..., min_length=1, description="要送去外部模型的問題")
    style: str = Field(default="auto", description="回答模板/情境：auto/general/academic/life/quick")
    session_id: str = Field(default="default", description="用於分組每次轉換 log（前端維持即可）")
    store_dir: str = Field(default="rag_store")
    profile: str = Field(default_factory=lambda: os.getenv("RAG_PROFILE_PATH", "profile.md"))
    max_profile_facts: int = Field(default=4, ge=0, le=12)
    min_profile_score: float = Field(default=0.25, ge=-1.0, le=1.0)
    top_k: int = Field(default=8, ge=1, le=50)
    max_retrieved_chars: int = Field(default=3200, ge=200, le=20000)
    min_score: float = Field(default=0.22, ge=-1.0, le=1.0)
    max_items: int = Field(default=4, ge=0, le=12)
    # relevance gates：寧可不放，也不要放錯（避免把不相干背景塞進 prompt 造成答案偏航）
    min_top_score: float = Field(default=0.0, ge=-1.0, le=1.0, description="若 top hit 分數低於此值，視為無足夠相關片段")
    use_relative_threshold: bool = Field(default=True, description="是否啟用相對 top hit 的動態門檻")
    rel_score_ratio: float = Field(default=0.90, ge=0.0, le=1.0, description="保留分數 >= top_score * ratio 的片段")
    rel_score_drop: float = Field(default=0.08, ge=0.0, le=1.0, description="保留分數 >= top_score - drop 的片段")
    keyword_gate: bool = Field(default=True, description="是否啟用題目關鍵詞閘門（片段須含題目關鍵詞，除非分數很高）")
    min_keyword_hits: int = Field(default=1, ge=1, le=6, description="片段至少命中多少個關鍵詞才算通過")
    keyword_force_keep_score: float = Field(default=0.45, ge=-1.0, le=1.0, description="若分數高於此值，即使沒命中關鍵詞仍可保留")
    lexical_gate: bool = Field(default=True, description="是否啟用字元 n-gram 覆蓋率閘門（擋完全不同主題）")
    min_lexical_overlap: float = Field(default=0.06, ge=0.0, le=1.0, description="n-gram 覆蓋率的最小門檻（越大越嚴格）")
    embedding_model: str = Field(default="")


app = FastAPI(title="Pre-prompt info organizer", version="0.1.0")

_SAFE_SESSION_RE = re.compile(r"[^a-zA-Z0-9._-]+")

@app.middleware("http")
async def _no_cache_static_assets(request: Request, call_next):
    """
    這個專案主要是本機工具，前端檔案常在開發時調整。
    避免瀏覽器一直吃到舊版 CSS/JS（導致看起來「完全沒改」），對 /web/* 一律禁用快取。
    """
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/web/"):
        # no-store：不要存；no-cache/must-revalidate：就算有舊快取也要重新驗證
        # pragma/expires：給舊代理/舊瀏覽器一個明確訊號
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp


def _safe_session_id(s: str) -> str:
    sid = (s or "").strip() or "default"
    sid = _SAFE_SESSION_RE.sub("_", sid)
    sid = sid.strip("._-") or "default"
    if len(sid) > 80:
        sid = sid[:80]
    return sid


def _session_dir(session_id: str) -> Path:
    sid = _safe_session_id(session_id)
    out_dir = LOG_DIR / "sessions" / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@contextmanager
def _session_debug_log_handler(*, session_id: str) -> Iterator[Optional[str]]:
    """
    在「單次請求期間」把 logger 輸出同步寫到 logs/sessions/<session_id>/debug.log，
    讓你排查某個 session 時不必翻全域 debug.log。
    """
    enable_session = os.getenv("RAG_SESSION_DEBUG_LOG", "1").strip() not in {"0", "false", "False"}
    if not enable_session:
        yield None
        return

    out_dir = _session_dir(session_id)
    debug_path = out_dir / SESSION_DEBUG_LOG_NAME
    fh = logging.FileHandler(str(debug_path), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    logger.addHandler(fh)
    try:
        yield str(debug_path)
    finally:
        # 重要：避免 handler 累積（熱重載/多次請求）
        try:
            logger.removeHandler(fh)
        finally:
            try:
                fh.close()
            except Exception:
                pass


def _write_prepare_log(*, request_id: str, session_id: str, payload: PrepareRequest, result: dict) -> str:
    """
    每次轉換寫一個獨立 log 檔（JSON）。
    注意：log 會包含使用者問題、檢索片段（含清理後文本）、以及最後輸出 prompt，僅建議本機除錯使用。
    """
    if os.getenv("RAG_LOG_EACH_REQUEST", "1").strip() in {"0", "false", "False"}:
        return ""
    sid = _safe_session_id(session_id)
    dt_utc = datetime.now(timezone.utc)
    dt_local = dt_utc.astimezone()
    tz_mode = _log_filename_tz()
    dt_for_name = dt_local if tz_mode in {"local", "localtime"} else dt_utc
    ts = _format_log_ts(dt_for_name)
    out_dir = _session_dir(sid)
    path = out_dir / f"{ts}-{request_id}.json"

    payload_dump = payload.model_dump()
    result_dump = result

    include_content = _env_bool("RAG_LOG_INCLUDE_CONTENT", default=False)
    if not include_content:
        payload_dump, result_dump = _redact_prepare_log_record(payload_dump=payload_dump, result_dump=result_dump)

    record = {
        "request_id": request_id,
        "session_id": sid,
        # created_at: 保持既有格式（UTC + Z），避免破壞既有解析/習慣
        "created_at": _iso_utc_z(dt_utc),
        # 額外提供 local 時間，方便肉眼對照（含 offset）
        "created_at_local": dt_local.isoformat(timespec="seconds"),
        "log_filename_tz": tz_mode,
        "payload": payload_dump,
        "result": result_dump,
        "log_include_content": include_content,
    }

    max_chars = int(os.getenv("RAG_LOG_MAX_CHARS") or "250000")
    text = json.dumps(record, ensure_ascii=False, indent=2)
    if len(text) > max_chars:
        # 太大就先縮成單行；仍太大就硬截斷尾巴，避免產生超巨檔案
        text = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        if len(text) > max_chars:
            text = text[: max_chars - 18] + '"__TRUNCATED__"}'
    path.write_text(text, encoding="utf-8")
    return str(path)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # 常見的設定錯誤：缺少 API key。這不屬於伺服器壞掉，回 400 並給清楚指引。
    if isinstance(exc, RuntimeError) and "找不到 API key" in str(exc):
        return JSONResponse(
            status_code=400,
            content={
                "error": "缺少 API key：請在專案根目錄建立 `.env`，設定 SUPER_MIND_API_KEY 或 OPENAI_API_KEY（兩者擇一），或改用系統環境變數後重啟後端。",
                "detail": str(exc),
            },
        )
    # 壓縮背景失敗：同樣是設定/權限問題，回 400 讓使用者能立即修。
    if isinstance(exc, RuntimeError) and "壓縮摘要未產生" in str(exc):
        return JSONResponse(
            status_code=400,
            content={
                "error": "背景摘要產生失敗：請設定可用的 RAG_COMPRESS_MODEL，並確認 API key 具有 chat/response 模型權限；必要時也可設 RAG_COMPRESS_API=responses。",
                "detail": str(exc),
            },
        )

    # 把完整堆疊寫入 log 檔（不要把敏感資訊回傳到前端）
    trace_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
    logger.exception("trace_id=%s path=%s error=%s", trace_id, request.url.path, repr(exc))
    return JSONResponse(
        status_code=500,
        content={
            "error": "後端發生錯誤（請查看該 session 的 debug log；或查看後端終端輸出）。",
            "trace_id": trace_id,
        },
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        str(WEB_DIR / "index.html"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.post("/api/prepare")
def api_prepare(payload: PrepareRequest) -> Any:
    request_id = secrets.token_hex(8)
    sid = _safe_session_id(payload.session_id)
    with _session_debug_log_handler(session_id=sid) as session_debug_log:
        logger.info("prepare_start session_id=%s request_id=%s", sid, request_id)
        try:
            result_full = prepare(
                question=payload.question,
                style=payload.style,
                store_dir=Path(payload.store_dir),
                profile_path=Path(payload.profile),
                embedding_model=payload.embedding_model,
                max_profile_facts=payload.max_profile_facts,
                min_profile_score=payload.min_profile_score,
                top_k=payload.top_k,
                max_retrieved_chars=payload.max_retrieved_chars,
                min_score=payload.min_score,
                max_items=payload.max_items,
                min_top_score=payload.min_top_score,
                use_relative_threshold=payload.use_relative_threshold,
                rel_score_ratio=payload.rel_score_ratio,
                rel_score_drop=payload.rel_score_drop,
                keyword_gate=payload.keyword_gate,
                min_keyword_hits=payload.min_keyword_hits,
                keyword_force_keep_score=payload.keyword_force_keep_score,
                lexical_gate=payload.lexical_gate,
                min_lexical_overlap=payload.min_lexical_overlap,
                debug=True,
            )
            log_file = _write_prepare_log(request_id=request_id, session_id=sid, payload=payload, result=result_full)

            # 回前端維持精簡；debug 資料只落盤
            resp = dict(result_full)
            resp.pop("debug", None)
            resp["request_id"] = request_id
            resp["log_file"] = log_file
            if session_debug_log:
                resp["session_debug_log_file"] = session_debug_log
            logger.info("prepare_done session_id=%s request_id=%s log_file=%s", sid, request_id, log_file)
            return resp

        except Exception as exc:
            # 讓「該 session」底下也一定有可追的紀錄（JSON + debug.log）
            trace_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
            logger.exception("prepare_failed session_id=%s request_id=%s trace_id=%s error=%r", sid, request_id, trace_id, exc)

            status_code = 500
            user_error = "後端發生錯誤（已寫入 debug log）。"
            # 常見設定錯誤：缺少 API key / 壓縮模型不可用
            if isinstance(exc, RuntimeError) and "找不到 API key" in str(exc):
                status_code = 400
                user_error = "缺少 API key：請在專案根目錄建立 `.env`，設定 SUPER_MIND_API_KEY 或 OPENAI_API_KEY（兩者擇一），或改用系統環境變數後重啟後端。"
            elif isinstance(exc, RuntimeError) and "壓縮摘要未產生" in str(exc):
                status_code = 400
                user_error = "背景摘要產生失敗：請設定可用的 RAG_COMPRESS_MODEL，並確認 API key 具有 chat/response 模型權限；必要時也可設 RAG_COMPRESS_API=responses。"

            err_payload = {
                "error": user_error,
                "detail": str(exc),
                "trace_id": trace_id,
            }
            if session_debug_log:
                err_payload["debug_log_file"] = session_debug_log

            # 仍寫一份 request json，讓同一個 session 資料夾可以一鍵還原當下參數
            req_log = _write_prepare_log(request_id=request_id, session_id=sid, payload=payload, result={"error": user_error, "detail": str(exc), "trace_id": trace_id})
            if req_log:
                err_payload["log_file"] = req_log
            return JSONResponse(status_code=status_code, content=err_payload)


@app.get("/api/models")
def api_models() -> dict:
    """
    從目前設定的 provider/gateway 取得可用模型清單。
    這能避免「用錯 model 名稱」導致壓縮摘要永遠產不出來。
    """
    client = create_client(root_dir=ROOT)
    resp = client.models.list()
    data = getattr(resp, "data", None) or []
    ids = []
    for m in data:
        mid = getattr(m, "id", None)
        if mid:
            ids.append(str(mid))
    # 讓前端比較好用：排序、去重
    ids = sorted(set(ids))
    return {"models": ids, "count": len(ids)}


# serve static assets
app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")


