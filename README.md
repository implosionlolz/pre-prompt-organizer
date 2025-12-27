```bash
source .venv/bin/activate
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### Pre-prompt info organizer

你多半不是缺資訊，而是缺一個穩定的開局：外部模型不知道你是誰、你偏好什麼、你正在做什麼，所以每次都得重新補一段背景，最後還常常補太多、反而讓答案偏航。這個專案把那段開局背景變成一個可重複使用的本機流程：先把舊對話整理成可檢索資產，再在你提問當下只挑出「與此題真正相關」的最小必要脈絡，輸出成一段可直接貼給外部模型的文字。

前端是靜態頁面（`web/`），後端是 `FastAPI`；日常使用只需要開瀏覽器貼問題、按下轉換。

### 開源 / 隱私注意事項

這個專案本質上會接觸到你的私人資料（對話匯出、個人 profile、向量庫、除錯 log），所以我把預設策略設計成「可以開源程式碼，但不會把你的資料一起帶出去」。你會在專案裡看到 `logs/`、`past_chat_MD/`、`personal_limbo_MD/`、`personal_stable_MD/`、`rag_store/` 這些資料夾：它們在開源版只保留一個 README 與資料夾骨架，實際內容會被 `.gitignore` 全面忽略，避免誤傳到 GitHub。

另外，後端的 session JSON log 也新增了 `RAG_LOG_INCLUDE_CONTENT`：預設為 0 時，log 會自動把 `question/prompt/檢索片段原文/expand queries` 等內容打碼，只留下除錯必需的結構與分數資訊。這是為了降低你或貢獻者在分享 log 時不小心外洩的風險，但最穩妥的做法仍然是不要把 `logs/` 放進版本控制。

### 安裝（務必在 `.venv` 內）

在專案根目錄建立虛擬環境並安裝依賴（建議用較新的 Python 版本，避免 `numpy/faiss` wheel 不相容）：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 設定 `.env`

先複製範本：

```bash
cp env.example .env
```

（你若偏好 `.env.example` 這種命名，也可以在本機自行把 `env.example` 另存成 `.env.example`；程式實際讀取的是專案根目錄的 `.env`。）

然後在 `.env` 至少完成兩件事：

第一，你需要一把 API key（兩者擇一即可）：
`OPENAI_API_KEY` 或 `SUPER_MIND_API_KEY`。若你用 `SUPER_MIND_API_KEY` 且要自訂 gateway，可另外設定 `SUPER_MIND_BASE_URL`（未設時會套用程式內建的預設值）。

第二，請設定壓縮背景用的模型（這會直接影響輸出框能不能只留下最小必要背景）：
`RAG_COMPRESS_MODEL=<你的 chat 模型 id>`。啟動後你也可以在介面右側按下「列出後端可用模型」，或用 `GET /api/models` 取得清單，然後把回傳的 model id 填回 `.env`。

補充一個很實用的小技巧：`RAG_COMPRESS_MODEL`（以及下面的 `RAG_EXPAND_MODEL`）支援用逗號提供多個模型作為 fallback。當某個模型偶爾回空字串、或 gateway 不支援某個端點時，會自動換下一個模型，整體穩定度會明顯提升。

如果你要啟用「先擴寫再檢索」（建議你有大量中文歷史資料、但單次問題容易太短或太含混時使用），你可以在 `.env` 另外設定：
`RAG_EXPAND_ENABLE=1`，並用 `RAG_EXPAND_PROMPT_PATH` 指到你自己的提示詞檔案；模型則用 `RAG_EXPAND_MODEL`（未填時會退回用 `LLM_MODEL`；舊版 `GPT_MODEL` 仍相容）。

### supermind（AI Builders gateway）怎麼 call

如果你用的是 `SUPER_MIND_API_KEY`，預設 base_url 會是 `https://space.ai-builders.com/backend/v1`，對應的 OpenAI 相容端點就是 `POST /chat/completions`（也就是 `https://space.ai-builders.com/backend/v1/chat/completions`）。

下面是一個最小可用的 Python 範例（可用來快速驗證 key/base_url 是否正確、模型 id 是否存在）：

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["SUPER_MIND_API_KEY"],
    base_url=os.environ.get("SUPER_MIND_BASE_URL", "https://space.ai-builders.com/backend/v1"),
)

resp = client.chat.completions.create(
    model="supermind-agent-v1",
    messages=[
        {"role": "user", "content": "回我一句：你好，我在測試 supermind API。"},
    ],
    temperature=0.2,
    max_tokens=200,
)

print(resp.choices[0].message.content)
```

### 準備資料（一次做完，之後就能反覆用）

你可以把 Chatbox 匯出 JSON 轉成 Markdown（每個 session 一個檔案，另有 `index.md`）：

```bash
python3 chatbox_export_to_md.py \
  --input past_chat_MD/chatbox-exported-data-2025-12-20.json \
  --output-dir past_chat_MD/md_sessions
```

接著把 Markdown 建成本機向量庫（FAISS + metadata + embedding cache），預設會輸出到 `rag_store/`：

```bash
python3 build_rag_store.py \
  --input-dir past_chat_MD/md_sessions \
  --out-dir rag_store
```

如果你想改 embedding 模型，可以在 `.env` 設 `RAG_EMBEDDING_MODEL`；未設定時會用 `text-embedding-3-large`（同一個模型 id 也會被用在查詢端的 embedding）。

### 啟動與使用

啟動後端後，用瀏覽器打開 `http://127.0.0.1:8000`。你貼上問題並按下轉換後，畫面會同時顯示三種東西：這次引用到的個人檔案欄位、這次實際採用的模板（`auto/general/academic/life/quick`）、以及可追查的 log 路徑；最下方的輸出框則是「可直接複製貼給外部模型」的整段文字。

### 個人資訊（profile）怎麼放

後端讀取 profile 的預設位置是 `profile.md`（可留空）。你也可以改成資料夾形式：建立 `profile/`，把多個 `.md` / `.markdown` 檔案放進去，後端會遞迴讀取並合併抽取已填欄位。

此外為了降低搬移成本，如果你原本用 `profile.md`，但它不存在，後端會依序嘗試兩個相容路徑：同層的 `profile/` 資料夾，以及 `personal_stable_MD/profile.md`。如果你想明確指定來源，請在 `.env` 設定 `RAG_PROFILE_PATH`（可填檔案或資料夾路徑，且相對路徑一律以專案根目錄解讀）。

### 常見狀況（幾乎都跟設定或資料沒對齊有關）

如果你看到缺少 API key 的錯誤，代表後端沒有讀到 `.env` 或 `.env` 內沒有設定 `OPENAI_API_KEY` / `SUPER_MIND_API_KEY`。這個專案會固定從專案根目錄讀取 `.env`，所以通常是 `.env` 沒放對位置、或你改了環境變數但忘記重啟後端。

如果你看到背景摘要產生失敗（壓縮摘要未產生），通常是 `RAG_COMPRESS_MODEL` 填了不存在的 model id，或你的 key 對該模型沒有 chat/response 權限。你可以先用 `/api/models` 取得可用模型清單，再把正確的 id 填回 `.env`；必要時也能用 `RAG_COMPRESS_API=responses` 指定呼叫模式（預設為 `auto`）。

如果後端回報找不到 `index.faiss/meta.jsonl/config.json` 之類的檔案，代表你還沒建立向量庫或路徑不一致；請先跑一次 `build_rag_store.py`，或在前端/請求 payload 裡把 `store_dir` 指到正確資料夾。

### 進階設定（不看也能用）

這裡只列「真的會改變行為」的開關，細節以 `env.example` 與程式碼為準：

`RAG_REQUIRE_COMPRESSED_CONTEXT`：預設為 1，代表只要檢索到歷史片段，就一定要成功產生壓縮摘要；若摘要失敗就直接回錯，避免退化成把原始片段塞進輸出框。

`RAG_COMPRESS_MAX_TOKENS`：控制壓縮背景的長度，越小越精簡。

`RAG_PROMPT_INCLUDE_RAW_FRAGMENTS`：預設為 0；設為 1 時，會把清理後的原始檢索片段也放進輸出框（通常不建議，除非你要除錯或你很確定外部模型需要原文）。

`RAG_LOG_EACH_REQUEST`、`RAG_LOG_MAX_CHARS`、`RAG_SESSION_DEBUG_LOG`：控制每次 `/api/prepare` 是否落盤 JSON log、單一 log 最大大小、以及是否寫 session 專屬的 `debug.log`。log 預設寫在 `logs/sessions/<session_id>/...`，其中 `<session_id>` 由前端存到瀏覽器 localStorage（你也可以自行在請求 payload 指定）。

另外新增 `RAG_LOG_INCLUDE_CONTENT`（建議維持預設 0）：當為 0 時，session JSON log 會自動把 `question/prompt/檢索片段原文/expand queries` 等內容打碼，只保留分數與來源等除錯必需資訊；當為 1 時才會完整寫入內容。這是為了降低開源/分享時「不小心把私密資料落盤後又誤傳」的風險。

### 專案結構（讀一次就夠）

```text
.
├─ app.py                  # FastAPI 入口（/api/prepare、/api/models；並提供 web/ 靜態頁）
├─ core.py                 # 檢索、閘門、壓縮、prompt 組裝的核心流程
├─ embedding_client.py     # 讀取 .env、建立 OpenAI client、embedding helper
├─ chatbox_export_to_md.py # Chatbox 匯出 JSON → Markdown sessions
├─ build_rag_store.py      # Markdown sessions → rag_store/（FAISS + meta + embeddings cache）
├─ local_rag.py            # 本機 RAG store：載入、搜尋、cache schema
├─ web/                    # 單頁介面（貼問題、轉換、複製輸出）
├─ past_chat_MD/           # 你的原始匯出與產出的 md_sessions（可自行換路徑）
├─ rag_store/              # 建好的向量庫（index.faiss/meta.jsonl/config.json/embeddings.sqlite3）
└─ logs/sessions/          # 每次轉換的 request log 與 session debug log
```
