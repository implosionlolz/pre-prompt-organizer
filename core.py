from __future__ import annotations

import hashlib
import json
import os
import re
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import httpx

from embedding_client import create_client, embed_query, embed_texts
from local_rag import LocalRagStore, RetrievalHit


_WS_RE = re.compile(r"[ \t]+")
_DETAILS_BLOCK_RE = re.compile(r"<details>[\s\S]*?</details>", re.IGNORECASE)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_LOG = logging.getLogger("preprompt")
_CJK_SEQ_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_ALNUM_SEQ_RE = re.compile(r"[a-zA-Z0-9]{3,}")
_LEX_KEEP_RE = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff]+")

# 讓壓縮模型「第一次成功後就固定」：避免每次都走 fallback 試一輪，造成噪音與不穩定。
# key 只用 base_url（不記 key），因為這個專案一般是一個後端對一個 provider/gateway。
_COMPRESS_LOCK: Dict[str, Tuple[str, str]] = {}  # base_url -> (model, api: "chat"|"responses")


def _extract_query_keywords(question: str) -> List[str]:
    """
    非分詞的輕量關鍵詞抽取：用於「閘門式」排除明顯不相干片段。
    設計原則：保守、不要求完美，只要能擋掉噪音即可。
    """
    q = (question or "").strip()
    if not q:
        return []

    # 先抓「中文連續字串」與「英數 token」
    raw: List[str] = []
    raw.extend(_CJK_SEQ_RE.findall(q))
    raw.extend(_ALNUM_SEQ_RE.findall(q.lower()))

    # 去重（保留順序）
    seen: set[str] = set()
    uniq: List[str] = []
    for t in raw:
        s = t.strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)

    # 移除高度泛用詞（避免「營養/健康」這種字讓閘門失效）
    stop = {
        "請問",
        "怎麼",
        "如何",
        "要",
        "需要",
        "可以",
        "是否",
        "一天",
        "每天",
        "大概",
        "多少",
        "幾",
        "才夠",
        "夠",
        "建議",
        "推薦",
        "健康",
        "營養",
        "飲食",
        "運動",
        "訓練",
        "增肌",  # 增肌很重要，但常見同義詞多；讓語意分數去扛比較安全
    }
    out: List[str] = []
    for t in uniq:
        if t in stop:
            continue
        # 太短或過泛的詞不拿來當 gate key
        if len(t) < 2:
            continue
        # 避免把整句話當成 keyword（中文沒有空白時很常發生）
        if len(t) > 8:
            continue
        out.append(t)

    # 常見同義/縮寫補強（只做少量高價值的）
    expanded: List[str] = []
    for t in out:
        expanded.append(t)
        if t == "蛋白":
            expanded.append("蛋白質")
        if t == "蛋白質":
            expanded.append("protein")
        if t == "咖啡因":
            expanded.append("caffeine")
    # 再次去重
    seen2: set[str] = set()
    final: List[str] = []
    for t in expanded:
        if t in seen2:
            continue
        seen2.add(t)
        final.append(t)
    return final


def _count_keyword_hits(text: str, keywords: List[str]) -> int:
    if not text or not keywords:
        return 0
    t = text.lower()
    return sum(1 for k in keywords if k and (k.lower() in t))


def _normalize_for_lex_gate(text: str) -> str:
    # 只保留中英數，避免標點/空白干擾 n-gram
    s = (text or "").lower()
    s = _LEX_KEEP_RE.sub("", s)
    return s


def _normalize_query_for_lex_gate(question: str) -> str:
    """
    為了讓 lexical gate 不被「一天/多少」這類泛用詞誤導，對 query 做更激進的去噪。
    這不是分詞，只是把常見疑問/量詞/口語助詞刪掉，保留較具主題性的字串。
    """
    q = _normalize_for_lex_gate(question)
    if not q:
        return ""
    # 先刪多字 stop terms（避免拆成 n-gram 後仍留下大量噪音）
    stop_terms = [
        "請問",
        "怎麼",
        "如何",
        "需要",
        "可以",
        "一天",
        "每天",
        "多少",
        "大概",
        "才夠",
    ]
    for t in stop_terms:
        q = q.replace(t, "")
    # 再刪單字口語/助詞（保守，只刪最常見、且對主題區分度低的）
    stop_chars = set("我你他她它要吃喝嗎呢啊吧了的嗎")
    q = "".join(ch for ch in q if ch not in stop_chars)
    return q.strip()


def _char_ngrams(text: str, n: int) -> set[str]:
    if not text or n <= 0:
        return set()
    if len(text) < n:
        return set()
    return {text[i : i + n] for i in range(0, len(text) - n + 1)}


def _ngram_overlap_ratio(query: str, candidate: str) -> float:
    """
    語言無關的「字元 n-gram 覆蓋率」：
    - 用 query 的 n-gram 做分母，衡量 candidate 有覆蓋到多少 query 的表面訊號
    - 目的：擋掉「完全不同主題」但 embedding 分數仍偏高的片段
    """
    q = _normalize_query_for_lex_gate(query)
    c = _normalize_for_lex_gate(candidate)
    if not q or not c:
        return 0.0
    # 同時計算 2-gram/3-gram，取較大者，兼顧中英文
    q2 = _char_ngrams(q, 2)
    q3 = _char_ngrams(q, 3)
    if not q2 and not q3:
        return 0.0
    c2 = _char_ngrams(c, 2) if q2 else set()
    c3 = _char_ngrams(c, 3) if q3 else set()
    r2 = (len(q2 & c2) / max(1, len(q2))) if q2 else 0.0
    r3 = (len(q3 & c3) / max(1, len(q3))) if q3 else 0.0
    return float(max(r2, r3))


class CompressResult:
    __slots__ = ("text", "used_api", "model", "error_detail")

    def __init__(self, *, text: str, used_api: str, model: str, error_detail: str = "") -> None:
        self.text = text
        self.used_api = used_api
        self.model = model
        self.error_detail = error_detail


class ExpandResult:
    __slots__ = ("queries", "tags", "raw_json_text", "used_api", "model", "error_detail", "system_prompt_used")

    def __init__(
        self,
        *,
        queries: List[str],
        tags: List[str],
        raw_json_text: str,
        used_api: str,
        model: str,
        error_detail: str = "",
        system_prompt_used: str = "",
    ) -> None:
        self.queries = queries
        self.tags = tags
        self.raw_json_text = raw_json_text
        self.used_api = used_api
        self.model = model
        self.error_detail = error_detail
        self.system_prompt_used = system_prompt_used


def _read_text_file_best_effort(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _load_expand_system_prompt() -> str:
    """
    Query 擴寫用的 system prompt：
    - 優先讀 RAG_EXPAND_PROMPT_PATH（方便放多行）
    - 再讀 RAG_EXPAND_PROMPT（單行 env）
    - 最後退回內建預設（強制 JSON only）
    """
    p = (os.getenv("RAG_EXPAND_PROMPT_PATH") or "").strip()
    if p:
        pp = Path(p)
        if not pp.is_absolute():
            pp = (Path(__file__).resolve().parent / pp).resolve()
        text = _read_text_file_best_effort(pp).strip()
        if text:
            return text

    inline = (os.getenv("RAG_EXPAND_PROMPT") or "").strip()
    if inline:
        return inline

    return (
        "你是一個 Query 擴寫器。你的任務不是回答問題，而是為了 RAG 檢索產生多個檢索線索。\n"
        "請以繁體中文為主（可混少量英文關鍵字），產生與問題高度相關、但角度不同的 queries 與 tags。\n"
        "硬性輸出規則：\n"
        "1) 只能輸出 JSON（不要 Markdown、不要 code fence、不要任何解釋文字）。\n"
        "2) JSON schema：\n"
        '   {"queries":[...], "tags":[...]}\n'
        "3) queries：每個元素是可直接拿去做向量檢索的短句（避免太長；避免重複）。\n"
        "4) tags：每個元素是關鍵詞（短、可命中；避免泛用詞）。\n"
    )


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    # 常見：```json ... ```
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_json_candidate(text: str) -> str:
    """
    從 LLM 回覆中抓出最像 JSON 的片段（避免 gateway 包了一層說明文字）。
    先找第一個 { 到最後一個 }；找不到就回傳原文。
    """
    t = _strip_code_fences(text)
    if not t:
        return ""
    i = t.find("{")
    j = t.rfind("}")
    if 0 <= i < j:
        return t[i : j + 1].strip()
    return t.strip()


def _coerce_str_list(v: object, *, max_items: int) -> List[str]:
    out: List[str] = []
    if isinstance(v, list):
        for x in v:
            if isinstance(x, str):
                s = x.strip()
            else:
                s = str(x).strip() if x is not None else ""
            if not s:
                continue
            out.append(s)
            if len(out) >= max_items:
                break
    elif isinstance(v, str):
        s = v.strip()
        if s:
            out.append(s)
    return out


def _expand_queries_via_llm(*, client: object, question: str) -> ExpandResult:
    """
    在 RAG 前，用外部 LLM 產生 tags/queries（JSON only），以提升檢索品質。
    失敗時回傳空 lists；是否要中止由上層的 strict 開關決定。
    """
    enabled = os.getenv("RAG_EXPAND_ENABLE", "0").strip() in {"1", "true", "True"}
    if not enabled:
        return ExpandResult(
            queries=[],
            tags=[],
            raw_json_text="",
            used_api="disabled",
            model="",
            error_detail="",
            system_prompt_used="",
        )

    # 允許用逗號提供多個模型作為 fallback（跟 compress 一致），例如：
    #   RAG_EXPAND_MODEL=supermind-agent-v1,deepseek,gpt-5
    raw_models = (os.getenv("RAG_EXPAND_MODEL") or "").strip()
    if not raw_models:
        # 若使用 supermind gateway，預設優先用它自己的 agent；否則退回 LLM_MODEL（相容 GPT_MODEL）
        raw_models = (
            "supermind-agent-v1"
            if (os.getenv("SUPER_MIND_API_KEY") or "").strip()
            else (
                (os.getenv("LLM_MODEL") or "").strip()
                or (os.getenv("GPT_MODEL") or "").strip()
                or "gpt-5.2"
            )
        )
    models_to_try = [m.strip() for m in raw_models.split(",") if m and m.strip()]
    # 最後保險：避免空清單
    if not models_to_try:
        models_to_try = ["gpt-5.2"]
    prefer_api = (os.getenv("RAG_EXPAND_API") or "auto").strip().lower()
    max_tokens = int(os.getenv("RAG_EXPAND_MAX_TOKENS") or "350")
    temperature = float(os.getenv("RAG_EXPAND_TEMPERATURE") or "0.2")
    n_queries = int(os.getenv("RAG_EXPAND_NUM_QUERIES") or "6")
    n_queries = max(1, min(24, n_queries))

    user_system = _load_expand_system_prompt()
    # 重要：即使使用者給的 prompt 很「通用」，這裡仍要硬性鎖死輸出規格，避免擴寫失效或格式漂移。
    system = (
        user_system.strip()
        + "\n\n"
        + "你目前的任務是 Query 擴寫（用於 RAG 檢索），不是回答問題。\n"
        + "硬性輸出規則（必須遵守）：\n"
        + "1) 只能輸出 JSON（不要 Markdown、不要 code fence、不要任何解釋文字）。\n"
        + '2) JSON schema 必須為：{"queries":[...], "tags":[...]}。\n'
        + f"3) queries 必須產生 {n_queries} 個元素；每個元素是可用於檢索的短句（避免冗長、避免重複）。\n"
        + "4) tags 是關鍵詞清單（短、可命中；避免泛用詞）。\n"
    ).strip()
    user = (
        "目的：用來做 RAG 檢索，不是要你回答問題。\n"
        f"問題：{(question or '').strip()}\n"
        "請輸出 JSON："
    )

    sdk_has_responses = hasattr(client, "responses")
    base_url, api_key = _resolve_llm_http_config()
    official_openai = _is_official_openai_base_url(base_url)
    # AI Builders / supermind gateway 的 OpenAI 相容層通常只有 /chat/completions（見你貼的 OpenAPI），
    # auto 模式下不再嘗試 /responses，避免 404 把最後狀態覆蓋成「像是 responses 出錯」。
    looks_like_ai_builders = "/backend/v1" in (base_url or "")

    # 若使用者只給了一個 model，但該 model（例如某些 gateway 的 gpt-5）會回空 content，
    # 這裡在 auto 模式下自動把 gateway 的可用模型追加為 fallback，避免整條鏈路退化成永遠靠本機 fallback。
    if prefer_api == "auto" and len(models_to_try) == 1:
        try:
            resp_models = client.models.list()
            data = getattr(resp_models, "data", None) or []
            ids = []
            for m in data:
                mid = getattr(m, "id", None)
                if mid:
                    ids.append(str(mid))
            # 這些模型較常能穩定輸出可解析的文字（JSON）：
            preferred = ["supermind-agent-v1", "deepseek", "gemini-2.5-pro", "grok-4-fast", "gpt-5"]
            for mid in preferred:
                if mid in ids and mid not in models_to_try:
                    models_to_try.append(mid)
        except Exception:
            pass

    last_err = ""
    last_api = "none"
    last_model = models_to_try[-1] if models_to_try else ""
    raw_responses_supported = True

    # helper：決定本輪要嘗試哪些 API
    def _attempts_for(*, prefer_api: str) -> List[str]:
        if prefer_api == "chat":
            return ["chat"]
        if prefer_api == "responses":
            return ["responses"]
        # auto：官方 OpenAI 優先 responses；但 AI Builders/supermind 這類 gateway 優先 chat，且多半不支援 responses
        if looks_like_ai_builders and not official_openai:
            return ["chat"]
        return ["chat", "responses"] if not official_openai else ["responses", "chat"]

    for model in models_to_try:
        last_model = model
        attempts = _attempts_for(prefer_api=prefer_api)
        # 若 SDK 不支援 responses 且也沒有可用 raw http key/base_url，避免嘗試造成噪音
        if not sdk_has_responses and not (base_url and api_key):
            attempts = [a for a in attempts if a != "responses"]
        # 若已經確認 raw /responses 在此 base_url 不存在，就別再試
        if not raw_responses_supported:
            attempts = [a for a in attempts if a != "responses"]
        if not attempts:
            continue

        for api in attempts:
            last_api = api
            try:
                if api == "responses":
                    if sdk_has_responses:
                        resp = client.responses.create(
                            model=model,
                            input=[
                                {"role": "system", "content": system},
                                {"role": "user", "content": user},
                            ],
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                        )
                        out = _extract_response_text(resp).strip()
                        if not out:
                            last_err = "responses returned empty output_text"
                            continue
                        cand = _extract_json_candidate(out)
                        data = json.loads(cand)
                    else:
                        if not (base_url and api_key) or not raw_responses_supported:
                            last_err = "responses not supported by installed OpenAI SDK"
                            continue
                        status, data = _raw_post_json(
                            f"{base_url}/responses",
                            api_key=api_key,
                            payload={
                                "model": model,
                                "input": [
                                    {"role": "system", "content": system},
                                    {"role": "user", "content": user},
                                ],
                                "temperature": temperature,
                                "max_output_tokens": max_tokens,
                            },
                        )
                        if int(status) == 404:
                            raw_responses_supported = False
                            last_err = f"responses(raw) endpoint not found (status={status})"
                            continue
                        if int(status) >= 400:
                            last_err = f"responses(raw) http error (status={status})"
                            continue
                        raw_text = _extract_responses_text_from_json(data).strip()
                        if not raw_text:
                            last_err = f"responses(raw) returned empty (status={status})"
                            continue
                        data = json.loads(_extract_json_candidate(raw_text))

                else:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    out = ""
                    if getattr(resp, "choices", None):
                        msg = resp.choices[0].message
                        out = (msg.content or "").strip()
                    if not out:
                        _LOG.warning("expand_queries_empty: model=%s api=%s resp=%s", model, api, _safe_dump(resp))
                    if not out and base_url and api_key:
                        status, raw = _raw_post_json(
                            f"{base_url}/chat/completions",
                            api_key=api_key,
                            payload={
                                "model": model,
                                "messages": [
                                    {"role": "system", "content": system},
                                    {"role": "user", "content": user},
                                ],
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                            },
                        )
                        if int(status) >= 400:
                            last_err = f"chat(raw) http error (status={status})"
                            continue
                        out = _extract_chat_content_from_json(raw).strip()
                        if not out:
                            _LOG.warning(
                                "expand_queries_empty: model=%s api=%s status=%s resp=%s",
                                model,
                                "chat(raw)",
                                status,
                                _safe_dump(raw),
                            )

                    if not out:
                        last_err = "chat returned empty message.content"
                        continue

                    raw_json_text = _extract_json_candidate(out)
                    data = json.loads(raw_json_text)

                # 兼容 keys 命名：queries / prompts
                max_q = max(1, min(24, n_queries))
                queries = _coerce_str_list(data.get("queries") if isinstance(data, dict) else None, max_items=max_q)
                if not queries and isinstance(data, dict):
                    queries = _coerce_str_list(data.get("prompts"), max_items=max_q)
                tags = _coerce_str_list(data.get("tags") if isinstance(data, dict) else None, max_items=40)
                raw_json_text = json.dumps(data, ensure_ascii=False)
                return ExpandResult(
                    queries=queries,
                    tags=tags,
                    raw_json_text=raw_json_text,
                    used_api=api,
                    model=model,
                    error_detail="",
                    system_prompt_used=system,
                )

            except Exception as e:
                last_err = repr(e)
                _LOG.warning("expand_queries_failed: model=%s api=%s err=%r", model, api, e)

    return ExpandResult(
        queries=[],
        tags=[],
        raw_json_text="",
        used_api=last_api,
        model=last_model,
        error_detail=last_err,
        system_prompt_used=system,
    )


def _normalize_style(style: str) -> str:
    s = (style or "").strip().lower()
    # allow a few friendly aliases
    aliases = {
        "default": "general",
        "general": "general",
        "auto": "auto",
        "academic": "academic",
        "paper": "academic",
        "research": "academic",
        "life": "life",
        "zen": "life",
        "philosophy": "life",
        "quick": "quick",
        "concise": "quick",
    }
    return aliases.get(s, "auto")


def _infer_style_from_question(question: str) -> str:
    """
    MVP：先用保守的規則推斷情境，避免多一次 LLM 呼叫。
    之後若你要「讓大模型選模板」，可在這裡改成 tiny prompt router。
    """
    q = (question or "").strip().lower()
    if not q:
        return "general"
    academic_markers = [
        "論文",
        "研究",
        "文獻",
        "引用",
        "method",
        "methodology",
        "related work",
        "ablation",
        "baseline",
        "p-value",
        "假設",
        "證明",
        "定理",
        "推導",
        "嚴謹",
    ]
    life_markers = [
        "人生",
        "意義",
        "焦慮",
        "迷惘",
        "痛苦",
        "價值觀",
        "選擇",
        "關係",
        "自我",
        "禪",
        "冥想",
        "哲學",
        "維根斯坦",
    ]
    quick_markers = [
        "怎麼做",
        "步驟",
        "快速",
        "懶人包",
        "精簡",
        "結論",
        "直接",
        "我只要",
        "tl;dr",
    ]
    if any(m.lower() in q for m in academic_markers):
        return "academic"
    if any(m.lower() in q for m in life_markers):
        return "life"
    if any(m.lower() in q for m in quick_markers):
        return "quick"
    return "general"


def _style_rules(style: str) -> Tuple[str, str]:
    """
    回傳：(rule_block, tone_block)
    - rule_block：不可違背的回答規則（跨模板共通，但可調整呈現方式）
    - tone_block：口吻/結構偏好（模板差異點）
    """
    common = (
        "1) 只能使用背景中明確給出的資訊作為事實；不要把推測寫成事實。\n"
        "2) 如果缺少關鍵資料，先提出最多 3 個釐清問題；等我補充後再給完整方案。\n"
        "3) 對涉及健康/安全/法律等高風險情境，先列出風險與何時該尋求專業協助。\n"
        "4) 回覆以可執行為主：給出步驟、檢查清單，並說明每一步的目的。\n"
        "5) 避免冗長重述背景；若需要引用背景，請標記是引用哪個相關片段編號。\n"
    )
    s = _normalize_style(style)
    if s == "academic":
        tone = (
            "回覆風格：學術/結構化/邏輯嚴謹。\n"
            "建議結構：\n"
            "- 問題重述（1 句，界定範圍）\n"
            "- 核心假設與限制（若背景不足就列釐清問題）\n"
            "- 分點推理（以因果/定義/比較為主）\n"
            "- 可驗證的結論與下一步（包含如何驗證/反例）\n"
        )
        return common, tone
    if s == "life":
        tone = (
            "回覆風格：寬泛、帶一點哲學/反思的語言，但仍要落地可行。\n"
            "建議結構：\n"
            "- 先同理與框架化：你卡住的是什麼（不要替我下結論）\n"
            "- 提供 2–3 個視角（價值/時間/關係/能力圈等）\n"
            "- 給一個很小但可做的行動（今天就能試）\n"
        )
        return common, tone
    if s == "quick":
        tone = (
            "回覆風格：極簡、直接、以完成任務為導向。\n"
            "硬性格式：\n"
            "- 先給一段 TL;DR（3 行內）\n"
            "- 再給 3–7 個步驟的清單（每步一句話）\n"
        )
        return common, tone
    # general
    tone = "回覆風格：清楚、務實、條理分明；必要時提供兩種方案供選擇。"
    return common, tone


def _safe_dump(obj: object, *, max_chars: int = 2000) -> str:
    """
    把回應物件轉成可寫入 log 的字串（避免過大）。
    注意：不要 dump 使用者輸入（retrieved_text/question）以免把私密資料塞進 log。
    """
    try:
        if hasattr(obj, "model_dump"):
            s = str(obj.model_dump())
        else:
            s = repr(obj)
    except Exception:
        s = repr(obj)
    if len(s) > max_chars:
        return s[: max_chars - 3] + "..."
    return s


def _extract_response_text(resp: object) -> str:
    """
    OpenAI responses API 的輸出萃取（不同 gateway 實作可能只部分相容）。
    優先用 resp.output_text；否則退回遍歷 output。
    """
    try:
        t = str(getattr(resp, "output_text", "") or "").strip()
        if t:
            return t
    except Exception:
        pass
    out = getattr(resp, "output", None)
    if not out:
        return ""
    parts: List[str] = []
    try:
        for item in out:
            content = getattr(item, "content", None) or []
            for c in content:
                if getattr(c, "type", None) == "output_text":
                    txt = str(getattr(c, "text", "") or "").strip()
                    if txt:
                        parts.append(txt)
    except Exception:
        return ""
    return "\n".join(parts).strip()


def _resolve_llm_http_config() -> Tuple[str, str]:
    """
    供 raw HTTP fallback 使用（避免 gateway 回傳格式與 OpenAI SDK 型別不合，導致 content 變空字串）。
    注意：不要把 key 寫進 log。
    """
    supermind_key = (os.getenv("SUPER_MIND_API_KEY") or "").strip()
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    api_key = supermind_key or openai_key
    if not api_key:
        return "", ""
    # base_url 決策與 embedding_client.create_client 一致
    base_url = ""
    if supermind_key:
        base_url = (os.getenv("SUPER_MIND_BASE_URL") or "").strip()
        if not base_url:
            base_url = "https://space.ai-builders.com/backend/v1"
    else:
        base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
        if not base_url:
            base_url = "https://api.openai.com/v1"
    return base_url.rstrip("/"), api_key


def _is_official_openai_base_url(base_url: str) -> bool:
    b = (base_url or "").rstrip("/")
    return b.startswith("https://api.openai.com/v1")


def _normalize_used_api_for_lock(used_api: str) -> str:
    """
    將 used_api 正規化成 chat/responses，供 lock 使用。
    """
    a = (used_api or "").strip().lower()
    if a.startswith("responses"):
        return "responses"
    if a.startswith("chat"):
        return "chat"
    return ""


def _raw_post_json(url: str, *, api_key: str, payload: dict) -> Tuple[int, object]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=40.0) as c:
        r = c.post(url, headers=headers, json=payload)
    try:
        return int(r.status_code), r.json()
    except Exception:
        return int(r.status_code), r.text


def _extract_chat_content_from_json(data: object) -> str:
    if not isinstance(data, dict):
        return ""
    # --- OpenAI chat.completions 風格 ---
    choices = data.get("choices") or []
    if isinstance(choices, list) and choices:
        choice0 = choices[0] or {}
        if isinstance(choice0, dict):
            # 兼容舊式：choices[0].text
            t0 = choice0.get("text")
            if isinstance(t0, str) and t0.strip():
                return t0.strip()

            msg = choice0.get("message") or {}
            if isinstance(msg, dict):
                # 常見：message.content 是字串
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()

                # 變體：message.text / message.output_text
                mt = msg.get("text") or msg.get("output_text")
                if isinstance(mt, str) and mt.strip():
                    return mt.strip()

                # 變體：message.content 是 list parts（OpenAI responses-style 或 gateway 自訂）
                if isinstance(content, list):
                    parts: List[str] = []
                    for p in content:
                        if not isinstance(p, dict):
                            continue
                        t = p.get("text") or p.get("content") or p.get("value") or ""
                        if isinstance(t, str) and t.strip():
                            parts.append(t.strip())
                    if parts:
                        return "\n".join(parts).strip()

                # 變體：message.parts（部分 Gemini/其他相容層會這樣放）
                mparts = msg.get("parts")
                if isinstance(mparts, list):
                    parts2: List[str] = []
                    for p in mparts:
                        if not isinstance(p, dict):
                            continue
                        t = p.get("text") or p.get("content") or p.get("value") or ""
                        if isinstance(t, str) and t.strip():
                            parts2.append(t.strip())
                    if parts2:
                        return "\n".join(parts2).strip()

    # --- Gemini 原生 candidates 風格（某些 gateway 可能直接透傳） ---
    cands = data.get("candidates")
    if isinstance(cands, list) and cands:
        c0 = cands[0] or {}
        if isinstance(c0, dict):
            content = c0.get("content") or {}
            if isinstance(content, dict):
                parts = content.get("parts") or []
                if isinstance(parts, list):
                    out: List[str] = []
                    for p in parts:
                        if not isinstance(p, dict):
                            continue
                        t = p.get("text") or ""
                        if isinstance(t, str) and t.strip():
                            out.append(t.strip())
                    if out:
                        return "\n".join(out).strip()

    # --- 其他常見相容變體：頂層 content blocks ---
    top_content = data.get("content")
    if isinstance(top_content, list):
        out2: List[str] = []
        for b in top_content:
            if not isinstance(b, dict):
                continue
            t = b.get("text") or b.get("content") or b.get("value") or ""
            if isinstance(t, str) and t.strip():
                out2.append(t.strip())
        if out2:
            return "\n".join(out2).strip()

    return ""


def _extract_responses_text_from_json(data: object) -> str:
    if not isinstance(data, dict):
        return ""
    t = data.get("output_text")
    if isinstance(t, str) and t.strip():
        return t.strip()
    out = data.get("output") or []
    if not isinstance(out, list):
        return ""
    parts: List[str] = []
    for item in out:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or []
        if not isinstance(content, list):
            continue
        for c in content:
            if not isinstance(c, dict):
                continue
            if c.get("type") in {"output_text", "text"}:
                txt = c.get("text")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt.strip())
    return "\n".join(parts).strip()


def _normalize_for_dedupe(text: str) -> str:
    t = text.strip()
    t = _WS_RE.sub(" ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strip_details_blocks(text: str) -> str:
    """
    chatbox_export_to_md.py 會把 tool-call / OCR 包在 <details> 裡。
    對「要貼給外部模型的最小必要背景」來說，這通常是噪音（尤其 web_search 結果會帶一堆連結）。
    """
    return _DETAILS_BLOCK_RE.sub("", text or "")


def _strip_urls_and_md_links(text: str) -> str:
    # 先把 markdown link 轉成純文字（保留 anchor，丟掉 URL）
    t = _MD_LINK_RE.sub(r"\1", text or "")
    # 再移除裸 URL（避免殘留）
    t = _URL_RE.sub("", t)
    # 清掉因移除造成的多餘空白
    t = re.sub(r"[ \t]+", " ", t)
    return t


def _extract_user_only(text: str) -> str:
    """
    build_rag_store.py 產生的 chunk 會長得像：
      你：...
      助理：...
    並且內容可能跨多行。這裡用狀態機保留「你」的內容，丟掉「助理」的內容，
    以符合「只要我個人的資訊」的使用情境。

    若日後你想保留助理內容，可把環境變數 RAG_PROMPT_USER_ONLY=0。
    """
    if not text:
        return ""
    lines = text.splitlines()
    keep = True  # 預設保留，直到遇到明確的 "助理："
    out: List[str] = []
    for line in lines:
        if line.startswith("你："):
            keep = True
            out.append(line[len("你：") :].lstrip())
            continue
        if line.startswith("助理："):
            keep = False
            continue
        if keep:
            out.append(line)
    return "\n".join(out).strip()


def _dedupe_lines_preserve_order(text: str) -> str:
    seen: set[str] = set()
    out: List[str] = []
    for line in (text or "").splitlines():
        raw = line.rstrip()
        if not raw.strip():
            # 保留空行，但避免連續多個空行
            if out and out[-1] != "":
                out.append("")
            continue
        key = _WS_RE.sub(" ", raw.strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(raw)
    # 收尾：去掉頭尾空行、壓成最多一個空行
    t = "\n".join(out).strip()
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t


_FRAGMENT_SPLIT_RE = re.compile(r"(?=【片段\s+\d+】)")


def _fallback_compress_retrieved_text(retrieved_text: str) -> str:
    """
    壓縮摘要的本機 fallback（不使用 LLM）：
    - 只做去重/裁切/重排，不新增任何事實
    - 目的：避免上游模型回空字串時（content=""），在 require_summary 模式下整個流程直接 400
    """
    src = (retrieved_text or "").strip()
    if not src:
        return ""
    max_chars = int(os.getenv("RAG_COMPRESS_FALLBACK_MAX_CHARS") or "2200")
    per_fragment_max = int(os.getenv("RAG_COMPRESS_FALLBACK_PER_FRAGMENT_MAX_CHARS") or "900")
    max_chars = max(400, min(12000, max_chars))
    per_fragment_max = max(200, min(6000, per_fragment_max))

    parts = [p.strip() for p in _FRAGMENT_SPLIT_RE.split(src) if (p or "").strip()]
    out: List[str] = []
    used = 0
    for p in parts:
        block = _dedupe_lines_preserve_order(_normalize_for_dedupe(p)).strip()
        if not block:
            continue
        if len(block) > per_fragment_max:
            block = block[:per_fragment_max].rstrip() + "\n（以上為截斷）"
        add = len(block) + (2 if out else 0)
        if used + add > max_chars:
            break
        out.append(block)
        used += add
    return "\n\n".join(out).strip()


def _organize_personal_background(text: str) -> str:
    """
    把使用者背景做輕量整理：不是摘要（不生成新事實），而是去重 + 去噪。
    移除寫死的領域特定分類規則。
    """
    if not text.strip():
        return ""
    # 去掉常見「評分/抱怨上一版輸出」的 meta 段落：這些通常不是問題背景，而是對回答品質的評論。
    # （保守做法：只要命中明確關鍵詞才移除，不做語意判斷。）
    meta_markers = [
        "完成度",
        "滿分",
        "模型的輸出",
        "為什麼會這麼多link",
        "我不需要link",
        "我不要link",
        "不要有網頁",
        "你不要查",
        "你不需要幫我",
    ]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text.strip()) if p.strip()]
    paragraphs = [p for p in paragraphs if not any(m in p for m in meta_markers)]
    if not paragraphs:
        return ""

    # 移除寫死的領域特定分類，只做基本的去重和整理
    # 保留原始段落結構，不做硬編碼的分類
    out: List[str] = []
    for para in paragraphs:
        body = _dedupe_lines_preserve_order(para).strip()
        if body:
            out.append(body)
    
    return "\n\n".join(out).strip()


def _clean_chunk_for_prompt(raw: str) -> str:
    t = str(raw or "")
    t = _strip_details_blocks(t)
    t = _strip_urls_and_md_links(t)
    if os.getenv("RAG_PROMPT_USER_ONLY", "1").strip() not in {"0", "false", "False"}:
        t = _extract_user_only(t)
    t = _normalize_for_dedupe(t)
    t = _dedupe_lines_preserve_order(t)
    t = _organize_personal_background(t)
    return t.strip()


def _compress_background_via_llm(*, client: object, question: str, retrieved_text: str) -> CompressResult:
    """
    進一步把「已清理過的檢索片段」壓縮成更精簡的背景。
    目標是減少冗長與重複，但嚴格不新增事實（只重寫/重排/去重）。
    """
    if os.getenv("RAG_COMPRESS_CONTEXT", "1").strip() in {"0", "false", "False"}:
        return CompressResult(text="", used_api="disabled", model="", error_detail="")
    src = (retrieved_text or "").strip()
    if not src:
        return CompressResult(text="", used_api="empty", model="", error_detail="")

    # 讓使用者可指定想要的壓縮模型；支援用逗號提供 fallback 清單，例如：
    #   RAG_COMPRESS_MODEL=gemini-2.5-pro,supermind-agent-v1
    raw_models = (os.getenv("RAG_COMPRESS_MODEL") or "").strip()
    if not raw_models:
        # Super Mind / Superlinear gateway 預設提供的 model 名稱通常不是 OpenAI 的命名。
        raw_models = "supermind-agent-v1" if (os.getenv("SUPER_MIND_API_KEY") or "").strip() else "gpt-5.2"
    models_to_try = [m.strip() for m in raw_models.split(",") if m.strip()]
    max_tokens = int(os.getenv("RAG_COMPRESS_MAX_TOKENS") or "650")
    temperature = float(os.getenv("RAG_COMPRESS_TEMPERATURE") or "0.2")
    prefer_api = (os.getenv("RAG_COMPRESS_API") or "auto").strip().lower()

    system = (
        "你是一位嚴謹的編輯。你的任務是把「素材」壓縮成可直接貼給外部模型的最小必要背景。\n"
        "硬性規則：\n"
        "1) 只能使用素材中明確出現的資訊；不允許補充、推測、延伸。\n"
        "2) 移除重複句與贅字，但保留所有數值、座標、限制條件、目標描述。\n"
        "3) 不能輸出任何 URL 或外部連結。\n"
        "4) 每個事實句尾都要標註來源片段編號（例如：來源：片段 1；若多個就用逗號）。\n"
        "5) 不要輸出「片段/round」之類的標題行；只保留句尾的來源標註即可。\n"
        "輸出格式：短段落（優先），可用少量短句；務必精簡、去重、可直接貼用。\n"
        "輸出語言：繁體中文。"
    )
    user = (
        f"問題：{question.strip()}\n\n"
        "素材（只能用此處作為事實）：\n"
        f"{src}\n\n"
        "請輸出「最小必要背景（已去重整理）」："
    )

    base_url, api_key = _resolve_llm_http_config()
    sdk_has_responses = hasattr(client, "responses")
    official_openai = _is_official_openai_base_url(base_url)
    # AI Builders / supermind gateway 的 OpenAI 相容層通常只有 /chat/completions；
    # auto 模式下不要嘗試 /responses，避免 404 + 噪音 + 浪費一次請求。
    looks_like_ai_builders = "/backend/v1" in (base_url or "")

    lock_after_success = os.getenv("RAG_COMPRESS_LOCK_AFTER_SUCCESS", "1").strip() not in {"0", "false", "False"}
    locked = _COMPRESS_LOCK.get(base_url) if lock_after_success and base_url else None
    if locked:
        locked_model, _locked_api = locked
        # 只把「模型」往前提：API 偏好由後面 attempts 邏輯決定（避免違反使用者顯式 prefer_api）
        if locked_model in models_to_try:
            models_to_try = [locked_model] + [m for m in models_to_try if m != locked_model]

    # 若使用者只給了一個 model，但該 model（例如某些 gateway 的 gpt-5）會回空 content，
    # 這裡在 auto 模式下自動把 gateway 的可用模型追加為 fallback，避免整個流程卡死。
    if prefer_api == "auto" and len(models_to_try) == 1:
        try:
            resp_models = client.models.list()
            data = getattr(resp_models, "data", None) or []
            ids = []
            for m in data:
                mid = getattr(m, "id", None)
                if mid:
                    ids.append(str(mid))
            preferred = ["supermind-agent-v1", "gemini-2.5-pro", "grok-4-fast", "deepseek"]
            for mid in preferred:
                if mid in ids and mid not in models_to_try:
                    models_to_try.append(mid)
        except Exception:
            pass

    last_err = ""
    last_api = "none"
    last_model = models_to_try[-1] if models_to_try else ""
    # raw HTTP fallback 中，某些 base_url 並不支援 /responses；遇到 404 後在同一輪就不再嘗試，避免噪音。
    raw_responses_supported = True

    for model in models_to_try:
        # 有些 gateway 對同一個 model 只支援其中一種 API（chat 或 responses）。
        # auto：對 gpt-5 這類新模型，responses 成功率通常更高。
        if prefer_api == "chat":
            attempts = ["chat"]
        elif prefer_api == "responses":
            attempts = ["responses"]
        else:
            # auto
            # - supermind/AI Builders 這類 gateway：通常只有 /chat/completions
            # - 其他非官方 base_url：chat 較常見且穩定；responses 可能 404
            # - 官方 OpenAI：gpt-5 類模型 responses 成功率較高
            if looks_like_ai_builders and not official_openai:
                attempts = ["chat"]
            elif not official_openai:
                attempts = ["chat", "responses"]
            else:
                attempts = ["responses", "chat"] if model.startswith("gpt-5") else ["chat", "responses"]

            # 若先前已在此 base_url 成功過，就把成功的 api 放在第一順位，讓後續請求穩定不抖動。
            if locked and lock_after_success:
                _locked_model, locked_api = locked
                if model == _locked_model and locked_api in {"chat", "responses"} and locked_api in attempts:
                    attempts = [locked_api] + [a for a in attempts if a != locked_api]

        # 若 SDK 不支援 responses，仍嘗試 raw /responses；但若沒有 key/base_url，就直接跳過避免噪音。
        if not sdk_has_responses and not (base_url and api_key):
            attempts = [a for a in attempts if a != "responses"]
        # 若已經確認 raw /responses 在此 base_url 不存在，就別再試
        if not raw_responses_supported:
            attempts = [a for a in attempts if a != "responses"]
        if not attempts:
            continue

        for api in attempts:
            last_api = api
            last_model = model
            try:
                if api == "responses":
                    if sdk_has_responses:
                        resp = client.responses.create(
                            model=model,
                            input=[
                                {"role": "system", "content": system},
                                {"role": "user", "content": user},
                            ],
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                        )
                        out = _extract_response_text(resp).strip()
                        if out:
                            if lock_after_success and base_url:
                                _COMPRESS_LOCK[base_url] = (model, "responses")
                            return CompressResult(text=out, used_api="responses", model=model, error_detail="")
                        _LOG.warning("compress_background_empty: model=%s api=%s resp=%s", model, api, _safe_dump(resp))
                        last_err = "responses returned empty output_text"
                        continue

                    if base_url and api_key:
                        status, data = _raw_post_json(
                            f"{base_url}/responses",
                            api_key=api_key,
                            payload={
                                "model": model,
                                "input": [
                                    {"role": "system", "content": system},
                                    {"role": "user", "content": user},
                                ],
                                "temperature": temperature,
                                "max_output_tokens": max_tokens,
                            },
                        )
                        # 這不是「模型回空」，而是「端點不存在/不支援」
                        if int(status) == 404:
                            raw_responses_supported = False
                            _LOG.info(
                                "compress_background_responses_unsupported: model=%s api=%s status=%s resp=%s",
                                model,
                                "responses(raw)",
                                status,
                                _safe_dump(data),
                            )
                            last_err = f"responses(raw) endpoint not found (status={status})"
                            continue
                        if int(status) >= 400:
                            _LOG.warning(
                                "compress_background_http_error: model=%s api=%s status=%s resp=%s",
                                model,
                                "responses(raw)",
                                status,
                                _safe_dump(data),
                            )
                            last_err = f"responses(raw) http error (status={status})"
                            continue
                        out = _extract_responses_text_from_json(data).strip()
                        if out:
                            if lock_after_success and base_url:
                                _COMPRESS_LOCK[base_url] = (model, "responses")
                            return CompressResult(text=out, used_api="responses(raw)", model=model, error_detail="")
                        _LOG.warning(
                            "compress_background_empty: model=%s api=%s status=%s resp=%s",
                            model,
                            "responses(raw)",
                            status,
                            _safe_dump(data),
                        )
                        last_err = f"responses(raw) returned empty (status={status})"
                        continue

                    last_err = "responses not supported by installed OpenAI SDK"
                    continue

                # chat
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                out = ""
                if getattr(resp, "choices", None):
                    msg = resp.choices[0].message
                    out = (msg.content or "").strip()
                out = out.strip()
                if out:
                    if lock_after_success and base_url:
                        _COMPRESS_LOCK[base_url] = (model, "chat")
                    return CompressResult(text=out, used_api="chat", model=model, error_detail="")

                # 重要：SDK 可能把 gateway 回傳的非標準欄位吞掉；用 raw 再看一次。
                if base_url and api_key:
                    status, data = _raw_post_json(
                        f"{base_url}/chat/completions",
                        api_key=api_key,
                        payload={
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system},
                                {"role": "user", "content": user},
                            ],
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                    )
                    if int(status) >= 400:
                        _LOG.warning(
                            "compress_background_http_error: model=%s api=%s status=%s resp=%s",
                            model,
                            "chat(raw)",
                            status,
                            _safe_dump(data),
                        )
                        last_err = f"chat(raw) http error (status={status})"
                        continue
                    raw_out = _extract_chat_content_from_json(data).strip()
                    if raw_out:
                        if lock_after_success and base_url:
                            _COMPRESS_LOCK[base_url] = (model, "chat")
                        return CompressResult(text=raw_out, used_api="chat(raw)", model=model, error_detail="")
                    _LOG.warning(
                        "compress_background_empty: model=%s api=%s status=%s resp=%s",
                        model,
                        "chat(raw)",
                        status,
                        _safe_dump(data),
                    )
                    last_err = f"chat returned empty message.content (status={status})"
                    continue

                _LOG.warning("compress_background_empty: model=%s api=%s resp=%s", model, api, _safe_dump(resp))
                last_err = "chat returned empty message.content"
                continue

            except Exception as e:
                last_err = repr(e)
                _LOG.warning("compress_background_failed: model=%s api=%s err=%r", model, api, e)

    return CompressResult(text="", used_api=last_api, model=last_model, error_detail=last_err)


def extract_filled_profile_facts(profile_md: str) -> List[str]:
    """
    `profile.md` 只需要一張表，這裡只提取表格中「已填值」的列，避免噪音。
    """
    lines = (profile_md or "").splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("|") and "欄位" in line and "值" in line:
            start = i + 2  # skip header + separator
            break
    if start is None:
        return []

    facts: List[str] = []
    for line in lines[start:]:
        if not line.strip().startswith("|"):
            break
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) < 2:
            continue
        key = cols[0]
        val = cols[1]
        note = cols[2] if len(cols) >= 3 else ""
        if not key or not val:
            continue
        if "待填" in val or val in {"-", "—"}:
            continue
        if note and "待填" in note:
            note = ""
        facts.append(f"{key}：{val}" + (f"（{note}）" if note else ""))

    return facts


_PROFILE_KV_RE = re.compile(r"^\s*(?:[-*]\s*)?([^:：]{1,40})\s*[:：]\s*(.+?)\s*$")


def extract_kv_profile_facts(profile_md: str) -> List[str]:
    """
    若不是用表格，也允許用簡單的 key: value / key：value（可加 - 或 * 作為項目符號）。
    這讓你可以把 preference / profile 拆成多個小檔案，但仍能被讀進 prompt。
    """
    facts: List[str] = []
    for raw in (profile_md or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        # 略過程式碼區塊、表格列與標題，避免誤抓
        if line.startswith("```") or line.startswith("|") or line.startswith("#"):
            continue
        m = _PROFILE_KV_RE.match(line)
        if not m:
            continue
        key = (m.group(1) or "").strip()
        val = (m.group(2) or "").strip()
        if not key or not val:
            continue
        if val in {"-", "—"} or "待填" in val:
            continue
        facts.append(f"{key}：{val}")
    return facts


def extract_profile_facts(profile_md: str) -> List[str]:
    # 先表格，沒有再補 key/value；最後做一次去重（保留順序）
    out: List[str] = []
    seen: set[str] = set()
    for f in extract_filled_profile_facts(profile_md) + extract_kv_profile_facts(profile_md):
        s = (f or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _iter_profile_markdown_files(profile_dir: Path) -> List[Path]:
    exts = {".md", ".markdown"}
    out: List[Path] = []
    try:
        for p in profile_dir.rglob("*"):
            if not p.is_file():
                continue
            if p.name.startswith("."):
                continue
            if p.suffix.lower() not in exts:
                continue
            out.append(p)
    except Exception:
        # rglob 在某些權限/符號連結情境可能會炸；保守退回不讀
        return []
    return sorted(out, key=lambda x: str(x).lower())


def load_profile_facts(profile_path: Path) -> List[str]:
    """
    支援兩種形式：
    - 檔案：傳入 profile.md（或任意 .md/.markdown）
    - 資料夾：傳入 profile/，會遞迴讀取底下所有 .md/.markdown 並合併抽取

    相容性：若傳入 profile.md 但檔案不存在，會自動嘗試同層的 profile/ 資料夾。
    """
    # 重要：把相對路徑固定解讀為「專案根目錄（core.py 所在資料夾）」底下，
    # 避免你從不同 cwd 啟動（例如用 systemd / IDE / 其他腳本）導致讀不到。
    project_root = Path(__file__).resolve().parent
    p = profile_path
    if not p.is_absolute():
        p = (project_root / p).resolve()

    if not p.exists() and p.suffix.lower() in {".md", ".markdown"}:
        cand = p.with_suffix("")  # e.g. profile.md -> profile/
        if cand.exists() and cand.is_dir():
            p = cand
        else:
            # 若你把個人穩定資訊集中到一個資料夾，允許最小摩擦的搬移：
            # 預設 profile.md 不存在時，自動嘗試 personal_stable_MD/profile.md
            alt = project_root / "personal_stable_MD" / p.name
            if alt.exists():
                p = alt
    if not p.exists():
        return []

    texts: List[str] = []
    if p.is_dir():
        for md in _iter_profile_markdown_files(p):
            try:
                texts.append(md.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
    else:
        try:
            texts.append(p.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return []

    facts: List[str] = []
    seen: set[str] = set()
    for t in texts:
        for f in extract_profile_facts(t):
            if f in seen:
                continue
            seen.add(f)
            facts.append(f)
    return facts


def format_hits(
    hits: List[RetrievalHit],
    *,
    question: str,
    max_chars: int,
    min_score: float,
    max_items: int,
    # relevance gates
    min_top_score: float = 0.0,
    use_relative_threshold: bool = True,
    rel_score_ratio: float = 0.0,
    rel_score_drop: float = 0.0,
    keyword_gate: bool = True,
    min_keyword_hits: int = 1,
    keyword_force_keep_score: float = 0.45,
    lexical_gate: bool = True,
    min_lexical_overlap: float = 0.06,
    debug_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[List[dict], str]:
    used = 0
    blocks: List[str] = []
    items: List[dict] = []
    seen: set[str] = set()

    top_score = float(hits[0].score) if hits else float("-inf")
    if hits and float(min_top_score) > float("-inf") and top_score < float(min_top_score):
        if debug_stats is not None:
            debug_stats["gate_top_score"] = {
                "top_score": top_score,
                "min_top_score": float(min_top_score),
                "passed": False,
            }
        return [], ""

    # 動態門檻：只保留「接近 top hit」的片段，避免把弱相關也一起塞進去。
    dyn_min_score = float(min_score)
    if hits and use_relative_threshold:
        cand = []
        if rel_score_ratio and rel_score_ratio > 0:
            cand.append(top_score * float(rel_score_ratio))
        if rel_score_drop and rel_score_drop > 0:
            cand.append(top_score - float(rel_score_drop))
        if cand:
            dyn_min_score = max(dyn_min_score, max(cand))

    keywords = _extract_query_keywords(question) if keyword_gate else []
    gate_active = keyword_gate and bool(keywords)
    if debug_stats is not None:
        debug_stats["gate_top_score"] = {
            "top_score": top_score,
            "min_top_score": float(min_top_score),
            "passed": True,
        }
        debug_stats["dyn_min_score"] = dyn_min_score
        debug_stats["query_keywords"] = keywords
        debug_stats["gate_keyword_active"] = gate_active
        debug_stats["gate_lexical_active"] = bool(lexical_gate)
        debug_stats["min_lexical_overlap"] = float(min_lexical_overlap)
        debug_stats["filtered_counts"] = {
            "by_score": 0,
            "by_empty": 0,
            "by_keyword": 0,
            "by_lexical": 0,
            "by_dedupe": 0,
            "by_budget": 0,
        }

    for h in hits:
        if len(items) >= max_items:
            break
        if h.score < dyn_min_score:
            if debug_stats is not None:
                debug_stats["filtered_counts"]["by_score"] += 1
            continue

        meta = h.meta or {}
        raw = str(meta.get("text") or "").strip()
        if not raw:
            if debug_stats is not None:
                debug_stats["filtered_counts"]["by_empty"] += 1
            continue

        cleaned = _clean_chunk_for_prompt(raw)
        if not cleaned:
            if debug_stats is not None:
                debug_stats["filtered_counts"]["by_empty"] += 1
            continue

        if float(h.score) < float(keyword_force_keep_score):
            if gate_active:
                kh = _count_keyword_hits(cleaned, keywords)
                if kh < int(min_keyword_hits):
                    # keyword 沒過，還可以用 lexical gate 做最後一道「擋完全不相干主題」的保險
                    if lexical_gate:
                        ov = _ngram_overlap_ratio(question, cleaned)
                        if ov < float(min_lexical_overlap):
                            if debug_stats is not None:
                                debug_stats["filtered_counts"]["by_lexical"] += 1
                            continue
                    else:
                        if debug_stats is not None:
                            debug_stats["filtered_counts"]["by_keyword"] += 1
                        continue
            elif lexical_gate:
                ov = _ngram_overlap_ratio(question, cleaned)
                if ov < float(min_lexical_overlap):
                    if debug_stats is not None:
                        debug_stats["filtered_counts"]["by_lexical"] += 1
                    continue

        key = _sha256(cleaned[:4000])
        if key in seen:
            if debug_stats is not None:
                debug_stats["filtered_counts"]["by_dedupe"] += 1
            continue
        seen.add(key)

        src = str(meta.get("source_path") or "")
        title = str(meta.get("title") or "")
        rs = meta.get("round_start")
        re_ = meta.get("round_end")
        # prompt 內不放 round / 檔名這類檢索痕跡；只留片段編號供引用即可
        header = f"【片段 {len(items)+1}】"
        block = header + "\n" + cleaned

        truncated = False
        if used + len(block) + 2 > max_chars:
            remaining = max(0, max_chars - used - len(header) - 1)
            if remaining < 200:
                if debug_stats is not None:
                    debug_stats["filtered_counts"]["by_budget"] += 1
                break
            cleaned = cleaned[:remaining].rstrip()
            block = header + "\n" + cleaned + "\n（以上為截斷）"
            truncated = True

        items.append(
            {
                "index": len(items) + 1,
                "score": float(h.score),
                "title": title,
                "source_path": src,
                "round_start": rs,
                "round_end": re_,
                "text": cleaned,
                "truncated": truncated,
            }
        )

        blocks.append(block)
        used += len(block) + 2

    return items, ("\n\n".join(blocks).strip())


def render_prompt(
    *,
    question: str,
    style: str,
    profile_facts: List[str],
    retrieved_text: str,
    retrieved_summary: str = "",
    include_raw_fragments: bool = False,
) -> str:
    resolved = _infer_style_from_question(question) if _normalize_style(style) == "auto" else _normalize_style(style)
    rules_block, tone_block = _style_rules(resolved)
    rules = "你將收到一個問題與一段背景。請嚴格遵守以下規則：\n" + rules_block

    parts: List[str] = []
    parts.append("## 使用規則")
    parts.append(rules.strip())
    parts.append("")
    parts.append("## 回答模板")
    parts.append(f"本次模板：{resolved}\n{tone_block}".strip())
    parts.append("")
    parts.append("## 背景（只包含與此問題相關的必要資訊）")
    if profile_facts:
        parts.append("### 長期不變資訊（由使用者維護）")
        parts.append("\n".join(profile_facts).strip())
        parts.append("")
    if retrieved_summary.strip():
        parts.append("### 已整理的最小必要背景（由歷史片段壓縮而成）")
        parts.append(retrieved_summary.strip())
        parts.append("")

    if include_raw_fragments and retrieved_text.strip():
        parts.append("### 相關片段（從歷史對話檢索）")
        parts.append(retrieved_text.strip())
        parts.append("")

    if (not retrieved_summary.strip()) and (not retrieved_text.strip()):
        parts.append("（沒有檢索到足夠相關的片段；請先靠釐清問題補足必要背景。）")
        parts.append("")
    parts.append("## 問題")
    parts.append(question.strip())
    return "\n".join(parts).strip() + "\n"


def _l2_normalize_2d(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return x / denom


def select_profile_facts(
    *,
    client: object,
    embedding_model: str,
    query_vec: np.ndarray,
    facts: List[str],
    max_facts: int,
    min_score: float,
) -> List[str]:
    if not facts or max_facts <= 0:
        return []
    mat = embed_texts(client, facts, model=embedding_model)
    mat = _l2_normalize_2d(mat)
    q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
    q = _l2_normalize_2d(q)[0]
    scores = (mat @ q.reshape(-1, 1)).reshape(-1)
    idxs = np.argsort(-scores)
    chosen: List[str] = []
    for i in idxs[: max_facts * 2]:
        s = float(scores[int(i)])
        if s < float(min_score):
            continue
        chosen.append(facts[int(i)])
        if len(chosen) >= max_facts:
            break
    return chosen


def prepare(
    *,
    question: str,
    style: str = "auto",
    store_dir: Path,
    profile_path: Path,
    embedding_model: str = "",
    max_profile_facts: int = 4,
    min_profile_score: float = 0.25,
    top_k: int = 8,
    max_retrieved_chars: int = 3200,
    min_score: float = 0.22,
    max_items: int = 4,
    # relevance gates (defaults chosen to prefer "寧可不放，也不要放錯")
    min_top_score: float = 0.0,
    use_relative_threshold: bool = True,
    rel_score_ratio: float = 0.90,
    rel_score_drop: float = 0.08,
    keyword_gate: bool = True,
    min_keyword_hits: int = 1,
    keyword_force_keep_score: float = 0.45,
    lexical_gate: bool = True,
    min_lexical_overlap: float = 0.06,
    debug: bool = False,
) -> dict:
    t0 = time.perf_counter()
    wall0 = time.time()

    # 用於 debug 的「時間線」：一步一步記下實際順序（避免看 JSON 欄位順序產生錯覺）
    _events: List[dict] = []

    def _utc_now() -> str:
        return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

    def _utc_from_wall(ts: float) -> str:
        # ts 為 time.time()（epoch seconds）
        return datetime.utcfromtimestamp(ts).isoformat(timespec="milliseconds") + "Z"

    def _rel_ms(now_wall: float) -> float:
        return round((now_wall - wall0) * 1000.0, 2)

    def _event(name: str, start_wall: float, end_wall: float, *, detail: Optional[dict] = None) -> None:
        _events.append(
            {
                "name": name,
                "ts_start_utc": _utc_from_wall(start_wall),
                "ts_end_utc": _utc_from_wall(end_wall),
                "t_rel_ms_start": _rel_ms(start_wall),
                "t_rel_ms_end": _rel_ms(end_wall),
                "dur_ms": round((end_wall - start_wall) * 1000.0, 2),
                "detail": detail or {},
            }
        )

    store = LocalRagStore.load(store_dir)
    model = str(embedding_model or "").strip()
    if not model:
        model = str(store.config.get("embedding_model") or os.getenv("RAG_EMBEDDING_MODEL") or "text-embedding-3-large")

    client = create_client(root_dir=Path(__file__).resolve().parent)
    # --- (optional) query expansion before embedding / retrieval ---
    w_expand0 = time.time()
    t_expand0 = time.perf_counter()
    expand = _expand_queries_via_llm(client=client, question=question)
    t_expand1 = time.perf_counter()
    w_expand1 = time.time()

    expand_enabled = os.getenv("RAG_EXPAND_ENABLE", "0").strip() in {"1", "true", "True"}
    expand_strict = os.getenv("RAG_EXPAND_STRICT", "0").strip() in {"1", "true", "True"}
    if expand_enabled and expand_strict and expand.error_detail:
        raise RuntimeError(
            "Query 擴寫失敗（RAG_EXPAND_STRICT=1）。"
            f"（本次嘗試：model={expand.model or '(unset)'} api={expand.used_api or '(unknown)'} err={expand.error_detail or '(empty)'}）"
        )

    # 若外部擴寫失敗/回空字串，不使用寫死的 fallback，直接使用原始問題進行 RAG 檢索。
    # （避免寫死的 fallback 產生與問題無關的查詢）
    expand_fallback_used = False
    expand_fallback_reason = ""

    # expansion 事件（包含 fallback 狀態）
    _event(
        "expand_queries",
        w_expand0,
        w_expand1,
        detail={
            "enabled": bool(expand_enabled),
            "strict": bool(expand_strict),
            "model": expand.model,
            "api": expand.used_api,
            "error": expand.error_detail,
            "fallback_used": bool(expand_fallback_used),
            "fallback_reason": expand_fallback_reason,
        },
    )

    # 組出實際用於檢索的 queries：永遠包含原始問題；其餘來自擴寫 queries + (可選) tags 聚合
    retrieval_queries: List[str] = []
    q0 = (question or "").strip()
    if q0:
        retrieval_queries.append(q0)
    for q in (expand.queries or []):
        s = (q or "").strip()
        if not s:
            continue
        retrieval_queries.append(s)
    if expand.tags:
        tag_query = " ".join(t.strip() for t in expand.tags if t and t.strip()).strip()
        if tag_query:
            retrieval_queries.append(tag_query)

    # 去重（保留順序），並限制總數避免延遲爆炸
    max_total_queries = 1 + int(os.getenv("RAG_EXPAND_NUM_QUERIES") or "6") + 1
    max_total_queries = max(1, min(30, max_total_queries))
    seen_q: set[str] = set()
    rq: List[str] = []
    for q in retrieval_queries:
        key = _WS_RE.sub(" ", q.strip().lower())
        if not key or key in seen_q:
            continue
        seen_q.add(key)
        rq.append(q.strip())
        if len(rq) >= max_total_queries:
            break
    retrieval_queries = rq

    # --- embed (batch) ---
    w_embed0 = time.time()
    t_embed0 = time.perf_counter()
    qmat = embed_texts(client, retrieval_queries, model=model)
    qvec = qmat[0]  # 原始問題向量，用於 profile facts ranking
    t_embed1 = time.perf_counter()
    w_embed1 = time.time()
    _event(
        "embed_queries",
        w_embed0,
        w_embed1,
        detail={"embedding_model": model, "n_queries": len(retrieval_queries)},
    )

    w_search0 = time.time()
    t_search0 = time.perf_counter()
    # 多 queries 檢索：每個 query 各撈一批候選，再以「同 id 取最高分」合併
    per_query_debug: List[dict] = []
    merged: Dict[int, RetrievalHit] = {}
    per_query_topk = max(int(top_k) * 3, int(top_k))
    for i, q in enumerate(retrieval_queries):
        vec = np.asarray(qmat[i], dtype=np.float32)
        hits = store.search(vec, top_k=per_query_topk)
        # 合併（同一 id 取最高 score）
        for h in hits:
            prev = merged.get(int(h.id))
            if prev is None or float(h.score) > float(prev.score):
                merged[int(h.id)] = h
        # debug：只存 top few，避免 log 過大
        if debug:
            per_query_debug.append(
                {
                    "query_index": i,
                    "query": q,
                    "hits_count": len(hits),
                    "top_hits": [{"id": int(h.id), "score": float(h.score)} for h in hits[: min(8, len(hits))]],
                }
            )
    raw_hits = sorted(merged.values(), key=lambda x: float(x.score), reverse=True)
    t_search1 = time.perf_counter()
    w_search1 = time.time()
    _event(
        "vector_search_merge",
        w_search0,
        w_search1,
        detail={
            "per_query_topk": per_query_topk,
            "retrieval_queries_used": retrieval_queries,
            "raw_hits_count": len(raw_hits),
        },
    )

    w_fmt0 = time.time()
    t_fmt0 = time.perf_counter()
    gate_dbg: Dict[str, Any] = {}
    items, retrieved_text = format_hits(
        raw_hits,
        question=question,
        max_chars=int(max_retrieved_chars),
        min_score=float(min_score),
        max_items=int(max_items),
        min_top_score=float(min_top_score),
        use_relative_threshold=bool(use_relative_threshold),
        rel_score_ratio=float(rel_score_ratio),
        rel_score_drop=float(rel_score_drop),
        keyword_gate=bool(keyword_gate),
        min_keyword_hits=int(min_keyword_hits),
        keyword_force_keep_score=float(keyword_force_keep_score),
        lexical_gate=bool(lexical_gate),
        min_lexical_overlap=float(min_lexical_overlap),
        debug_stats=gate_dbg if debug else None,
    )
    t_fmt1 = time.perf_counter()
    w_fmt1 = time.time()
    _event(
        "format_hits",
        w_fmt0,
        w_fmt1,
        detail={"kept_items": len(items), "retrieved_text_chars": len(retrieved_text or "")},
    )

    compressed = ""
    compress_used_api = ""
    compress_model = ""
    compress_error = ""
    if retrieved_text.strip():
        w_comp0 = time.time()
        t_comp0 = time.perf_counter()
        cres = _compress_background_via_llm(client=client, question=question, retrieved_text=retrieved_text)
        compressed = (cres.text or "").strip()
        compress_used_api = cres.used_api
        compress_model = cres.model
        compress_error = cres.error_detail
        t_comp1 = time.perf_counter()
        w_comp1 = time.time()
        _event(
            "compress_context",
            w_comp0,
            w_comp1,
            detail={
                "model": compress_model,
                "api": compress_used_api,
                "error": compress_error,
                "summary_chars": len(compressed or ""),
            },
        )
    else:
        t_comp0 = t_comp1 = time.perf_counter()
        # 沒有片段可壓縮也記一筆，讓時間線更完整
        w_now = time.time()
        _event("compress_context", w_now, w_now, detail={"skipped": True, "reason": "empty_retrieved_text"})
    # 預設只輸出「壓縮後背景」，不把 raw 片段塞進輸出框。
    include_raw_fragments = os.getenv("RAG_PROMPT_INCLUDE_RAW_FRAGMENTS", "0").strip() in {"1", "true", "True"}

    # 若使用者要求一定要摘要（預設開），摘要失敗就直接回錯，避免輸出退化成 raw 片段。
    require_summary = os.getenv("RAG_REQUIRE_COMPRESSED_CONTEXT", "1").strip() not in {"0", "false", "False"}
    if require_summary and retrieved_text.strip() and not compressed.strip():
        # 有些 provider/模型會回 200 但 message.content 是空字串（例如把 token 花在內部推理），
        # 這時如果硬要求摘要，會讓整個流程退化成「永遠 400」。
        # 預設啟用本機 fallback，至少讓流程可用；若你想維持嚴格行為可設 RAG_COMPRESS_FALLBACK_ON_FAILURE=0。
        fallback_on = os.getenv("RAG_COMPRESS_FALLBACK_ON_FAILURE", "1").strip() not in {"0", "false", "False"}
        if fallback_on:
            compressed = _fallback_compress_retrieved_text(retrieved_text)
            if compressed.strip():
                compress_error = (compress_error or "").strip() or "llm_compress_empty_output"
                compress_used_api = (compress_used_api or "").strip() or "unknown"
                compress_used_api = compress_used_api + "+fallback"
        if not compressed.strip():
            raise RuntimeError(
                "背景已檢索到，但壓縮摘要未產生。請在 `.env` 設定可用的 `RAG_COMPRESS_MODEL`（例如 gpt-5.2），"
                "並確認你的 API key 有 chat/response 模型權限；或將 `RAG_REQUIRE_COMPRESSED_CONTEXT=0` 改回允許輸出原始片段。"
                f"（本次嘗試：model={compress_model or '(unset)'} api={compress_used_api or '(unknown)'} err={compress_error or '(empty)'}）"
            )

    all_profile_facts = load_profile_facts(profile_path)
    w_prof0 = time.time()
    profile_facts = select_profile_facts(
        client=client,
        embedding_model=model,
        query_vec=qvec,
        facts=all_profile_facts,
        max_facts=int(max_profile_facts),
        min_score=float(min_profile_score),
    )
    w_prof1 = time.time()
    _event(
        "select_profile_facts",
        w_prof0,
        w_prof1,
        detail={"profile_facts_total": len(all_profile_facts), "profile_facts_used": len(profile_facts)},
    )

    w_prompt0 = time.time()
    prompt = render_prompt(
        question=question,
        style=style,
        profile_facts=profile_facts,
        retrieved_text=retrieved_text,
        retrieved_summary=compressed.strip(),
        include_raw_fragments=include_raw_fragments,
    )
    w_prompt1 = time.time()
    _event("render_prompt", w_prompt0, w_prompt1, detail={"prompt_chars": len(prompt or "")})
    t1 = time.perf_counter()

    # debug：只用於落盤/除錯，不建議直接塞進 UI
    dbg: dict = {}
    if debug:
        raw_hits_brief = []
        for h in raw_hits:
            m = h.meta or {}
            raw_hits_brief.append(
                {
                    "id": int(h.id),
                    "score": float(h.score),
                    "title": str(m.get("title") or ""),
                    "source_path": str(m.get("source_path") or ""),
                    "round_start": m.get("round_start"),
                    "round_end": m.get("round_end"),
                    "subchunk_index": m.get("subchunk_index"),
                    "subchunk_total": m.get("subchunk_total"),
                    "char_len": m.get("char_len"),
                }
            )
        dbg = {
            "events": _events,
            "query_expansion": {
                "enabled": bool(expand_enabled),
                "strict": bool(expand_strict),
                "model": expand.model,
                "api": expand.used_api,
                "error": expand.error_detail,
                "fallback_used": bool(expand_fallback_used),
                "fallback_reason": expand_fallback_reason,
                "fallback_queries": [],
                "fallback_tags": [],
                "system_prompt": expand.system_prompt_used,
                "raw_json": expand.raw_json_text,
                "tags": expand.tags,
                "queries": expand.queries,
                "retrieval_queries_used": retrieval_queries,
                "per_query_search_debug": per_query_debug,
            },
            "raw_hits_count": len(raw_hits),
            "raw_hits": raw_hits_brief,
            # 這是「清理後、已被採用」的片段串（含【片段 N】標頭）
            "retrieved_text": retrieved_text,
            "retrieval_gates": gate_dbg,
            "timings_ms": {
                "expand_queries": round((t_expand1 - t_expand0) * 1000, 2),
                "embed_query": round((t_embed1 - t_embed0) * 1000, 2),
                "search": round((t_search1 - t_search0) * 1000, 2),
                "format_hits": round((t_fmt1 - t_fmt0) * 1000, 2),
                "compress": round((t_comp1 - t_comp0) * 1000, 2),
                "total": round((t1 - t0) * 1000, 2),
            },
        }
    return {
        "question": question,
        "style": _normalize_style(style),
        "style_resolved": _infer_style_from_question(question) if _normalize_style(style) == "auto" else _normalize_style(style),
        "profile_facts": profile_facts,
        "profile_facts_total": len(all_profile_facts),
        "retrieved_items": items,
        "prompt": prompt,
        "store_dir": str(store_dir),
        "embedding_model": model,
        "retrieved_summary": compressed.strip(),
        # query expansion (for logs / debugging; UI may ignore)
        "expand_enabled": bool(expand_enabled),
        "expand_model": expand.model,
        "expand_api": expand.used_api,
        "expand_error": expand.error_detail,
        "expand_tags": expand.tags,
        "expand_queries": expand.queries,
        "expand_raw_json": expand.raw_json_text,
        "retrieval_queries_used": retrieval_queries,
        "compress_model": compress_model,
        "compress_api": compress_used_api,
        "compress_error": compress_error,
        "include_raw_fragments": include_raw_fragments,
        "debug": dbg,
    }


