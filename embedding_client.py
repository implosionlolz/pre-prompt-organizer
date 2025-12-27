from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from dotenv import dotenv_values, load_dotenv
from openai import OpenAI


def _load_env(*, root_dir: Optional[Path] = None) -> None:
    """
    只做一件事：確保 `.env` 會被讀到。

    你遇到的 500 很常見：uvicorn 的工作目錄（cwd）不一定是專案根目錄，
    導致 `load_dotenv()` 找不到 `.env`。
    這裡改成「固定從專案根目錄（或呼叫者指定的 root_dir）讀取」。
    """
    root = (root_dir or Path(__file__).resolve().parent).resolve()
    dotenv_path = root / ".env"
    if dotenv_path.exists():
        # 重要：這個專案常在同一個長跑的 uvicorn process 內反覆改 `.env` 調參。
        # `load_dotenv(..., override=False)` 會造成「第一次載入後就黏住」：後續改 `.env` 不會生效。
        #
        # 但我們也不能暴力 override=True，因為 `.env` 裡常會保留空白的 key（例如 OPENAI_API_KEY=），
        # 這會把你在 shell/系統環境已設定的 key 覆蓋成空字串。
        #
        # 因此策略是：
        # - 只把 `.env` 中「非空值」寫入 os.environ（允許熱更新）
        # - 空值不覆蓋既有環境變數（避免意外清空 key）
        vals = dotenv_values(dotenv_path=str(dotenv_path))
        for k, v in (vals or {}).items():
            if not k:
                continue
            if v is None:
                continue
            s = str(v)
            if s == "":
                continue
            os.environ[str(k)] = s
    else:
        # fallback：仍嘗試讓 python-dotenv 自動找（例如使用者用系統環境變數）
        load_dotenv(override=False)


def create_client(*, root_dir: Optional[Path] = None) -> OpenAI:
    """
    建立 OpenAI client。

    - 優先使用 SUPER_MIND_API_KEY（若存在，且未指定 SUPER_MIND_BASE_URL，會套用預設 gateway）
    - 否則使用 OPENAI_API_KEY
    """
    _load_env(root_dir=root_dir)
    supermind_key = os.getenv("SUPER_MIND_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    api_key = supermind_key or openai_key
    if not api_key:
        root = (root_dir or Path(__file__).resolve().parent).resolve()
        raise RuntimeError(
            "找不到 API key。請在專案根目錄的 `.env` 設定 SUPER_MIND_API_KEY 或 OPENAI_API_KEY，"
            f"或改用系統環境變數。預期位置：{root / '.env'}"
        )

    # base_url 決策：
    # - 若使用 SUPER_MIND_API_KEY：走 SUPER_MIND_BASE_URL（未設則套預設）
    # - 否則若使用 OPENAI_API_KEY：可用 OPENAI_BASE_URL 覆寫（預設官方）
    base_url = ""
    if supermind_key:
        base_url = (os.getenv("SUPER_MIND_BASE_URL") or "").strip()
        if not base_url:
            base_url = "https://space.ai-builders.com/backend/v1"
    else:
        base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
        if not base_url:
            base_url = "https://api.openai.com/v1"

    return OpenAI(api_key=api_key, base_url=base_url)


def embed_texts(client: OpenAI, texts: List[str], *, model: str) -> np.ndarray:
    resp = client.embeddings.create(model=model, input=texts)
    vectors = [np.asarray(item.embedding, dtype=np.float32) for item in resp.data]
    return np.vstack(vectors)


def embed_query(client: OpenAI, text: str, *, model: str) -> np.ndarray:
    return embed_texts(client, [text], model=model)[0]


