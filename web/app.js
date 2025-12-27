async function apiPrepare(payload) {
  const resp = await fetch("/api/prepare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    let msg = "";
    try {
      const data = await resp.json();
      if (data && data.error) {
        msg = data.error;
        if (data.trace_id) msg += `（trace_id=${data.trace_id}）`;
        if (data.log_file) msg += `，request_log：${data.log_file}`;
        if (data.debug_log_file) msg += `，debug_log：${data.debug_log_file}`;
      } else {
        msg = JSON.stringify(data);
      }
    } catch (_) {
      msg = await resp.text();
    }
    throw new Error(`API 失敗 (${resp.status}): ${msg}`);
  }
  return await resp.json();
}

async function apiModels() {
  const resp = await fetch("/api/models");
  if (!resp.ok) {
    let msg = "";
    try {
      msg = JSON.stringify(await resp.json());
    } catch (_) {
      msg = await resp.text();
    }
    throw new Error(`API 失敗 (${resp.status}): ${msg}`);
  }
  return await resp.json();
}

function getOrCreateSessionId() {
  const key = "preprompt_session_id";
  try {
    const existing = localStorage.getItem(key);
    if (existing && existing.trim()) return existing.trim();
    const sid = `sess_${Date.now()}_${Math.random().toString(16).slice(2)}`;
    localStorage.setItem(key, sid);
    return sid;
  } catch (_) {
    // 若瀏覽器禁用 localStorage，就退回每次隨機（仍可用，只是 log 不好分組）
    return `sess_${Date.now()}_${Math.random().toString(16).slice(2)}`;
  }
}

function setStatus(msg) {
  const el = document.getElementById("status");
  el.textContent = msg || "";
}

function renderFacts(containerId, facts) {
  const el = document.getElementById(containerId);
  if (!facts || facts.length === 0) {
    el.classList.add("empty");
    el.textContent = "（這次沒有引用到任何已填的 profile 欄位）";
    return;
  }
  el.classList.remove("empty");
  const ul = document.createElement("ul");
  for (const f of facts) {
    const li = document.createElement("li");
    li.textContent = f;
    ul.appendChild(li);
  }
  el.innerHTML = "";
  el.appendChild(ul);
}

function renderRetrievedSummary(containerId, data) {
  const el = document.getElementById(containerId);
  if (!el) return;
  const t = (data && data.retrieved_summary ? data.retrieved_summary : "").trim();
  const items = (data && data.retrieved_items) || [];
  if (!t) {
    el.classList.add("empty");
    el.textContent =
      items && items.length === 0
        ? "（本次沒有檢索到足夠相關的歷史片段，因此不輸出摘要）"
        : "（尚未產生摘要；請確認 RAG_COMPRESS_MODEL，必要時設 RAG_COMPRESS_API=responses）";
    return;
  }
  el.classList.remove("empty");
  el.textContent = t;
}

function renderStyleInfo(containerId, data) {
  const el = document.getElementById(containerId);
  if (!el) return;
  if (!data) {
    el.classList.add("empty");
    el.textContent = "（尚未轉換）";
    return;
  }
  const requested = (data.style || "").trim();
  const resolved = (data.style_resolved || "").trim();
  const lines = [];
  lines.push(`requested: ${requested || "（未回傳）"}`);
  lines.push(`resolved: ${resolved || "（未回傳）"}`);
  el.classList.remove("empty");
  el.textContent = lines.join("\n");
}

function renderLogInfo(containerId, data) {
  const el = document.getElementById(containerId);
  if (!el) return;
  if (!data) {
    el.classList.add("empty");
    el.textContent = "（尚未轉換）";
    return;
  }
  const rid = (data.request_id || "").trim();
  const file = (data.log_file || "").trim();
  const dbg = (data.session_debug_log_file || data.debug_log_file || "").trim();
  const lines = [];
  lines.push(`request_id: ${rid || "（未回傳）"}`);
  lines.push(`log_file: ${file || "（未回傳）"}`);
  if (dbg) lines.push(`debug_log_file: ${dbg}`);
  el.classList.remove("empty");
  el.textContent = lines.join("\n");
}

function renderCompressInfo(containerId, data) {
  const el = document.getElementById(containerId);
  if (!el) return;
  if (!data) {
    el.classList.add("empty");
    el.textContent = "（尚未轉換）";
    return;
  }
  const model = (data.compress_model || "").trim();
  const api = (data.compress_api || "").trim();
  const err = (data.compress_error || "").trim();
  const lines = [];
  lines.push(`model: ${model || "（未回傳）"}`);
  lines.push(`api: ${api || "（未回傳）"}`);
  if (err) lines.push(`error: ${err}`);
  el.classList.remove("empty");
  el.textContent = lines.join("\n");
}

function renderModelsList(containerId, modelsResp) {
  const el = document.getElementById(containerId);
  if (!el) return;
  const models = (modelsResp && modelsResp.models) || [];
  if (!models || models.length === 0) {
    el.classList.add("empty");
    el.textContent = "（後端沒有回傳模型清單；可能 gateway 不支援 /models）";
    return;
  }
  el.classList.remove("empty");
  el.textContent = models.join("\n");
}

function renderRetrievedRaw(containerId, items) {
  const el = document.getElementById(containerId);
  if (!items || items.length === 0) {
    el.classList.add("empty");
    el.textContent = "（沒有檢索到足夠相關的片段）";
    return;
  }
  el.classList.remove("empty");
  const ul = document.createElement("ul");
  for (const it of items) {
    const li = document.createElement("li");
    const title = it.title ? it.title : "（無標題）";
    li.textContent = `#${it.index} ${title}`;
    ul.appendChild(li);
  }
  el.innerHTML = "";
  el.appendChild(ul);
}

async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (_) {
    // fallback
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.left = "-9999px";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  }
}

function getInt(id, defVal) {
  const v = parseInt(document.getElementById(id).value, 10);
  return Number.isFinite(v) ? v : defVal;
}

function getFloat(id, defVal) {
  const v = parseFloat(document.getElementById(id).value);
  return Number.isFinite(v) ? v : defVal;
}

async function onConvert() {
  const question = document.getElementById("question").value.trim();
  if (!question) {
    setStatus("請先輸入問題。");
    return;
  }

  const btn = document.getElementById("btnConvert");
  const btnCopy = document.getElementById("btnCopy");
  btn.disabled = true;
  btnCopy.disabled = true;
  setStatus("轉換中…");

  try {
    const style = document.getElementById("style").value || "auto";
    const session_id = getOrCreateSessionId();
    const payload = {
      question,
      style,
      session_id,
      min_score: getFloat("minScore", 0.22),
      max_profile_facts: getInt("maxProfileFacts", 4),
      max_items: getInt("maxItems", 4),
      max_retrieved_chars: getInt("maxRetrievedChars", 3200),
    };
    const data = await apiPrepare(payload);
    document.getElementById("output").value = data.prompt || "";
    renderFacts("profileFacts", data.profile_facts || []);
    renderStyleInfo("styleInfo", data);
    renderLogInfo("logInfo", data);
    renderCompressInfo("compressInfo", data);
    renderRetrievedRaw("retrievedRaw", data.retrieved_items || []);
    btnCopy.disabled = !(data.prompt && data.prompt.length > 0);
    setStatus("完成。");
  } catch (e) {
    setStatus(e && e.message ? e.message : "發生未知錯誤。");
  } finally {
    btn.disabled = false;
  }
}

async function onModels() {
  const btn = document.getElementById("btnModels");
  if (btn) btn.disabled = true;
  setStatus("取得後端模型清單中…");
  try {
    const data = await apiModels();
    renderModelsList("modelsList", data);
    setStatus(`已取得模型清單（${data.count || 0} 個）。`);
  } catch (e) {
    setStatus(e && e.message ? e.message : "取得模型清單失敗。");
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function onCopy() {
  const out = document.getElementById("output").value || "";
  if (!out) return;
  const ok = await copyToClipboard(out);
  setStatus(ok ? "已複製到剪貼簿。" : "複製失敗（瀏覽器限制）。");
}

function main() {
  document.getElementById("btnConvert").addEventListener("click", onConvert);
  document.getElementById("btnCopy").addEventListener("click", onCopy);
  const btnModels = document.getElementById("btnModels");
  if (btnModels) btnModels.addEventListener("click", onModels);

  // Cmd/Ctrl + Enter 觸發轉換
  document.getElementById("question").addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      onConvert();
    }
  });
}

main();


