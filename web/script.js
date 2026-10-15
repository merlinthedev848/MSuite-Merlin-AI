/* ==========================================================================
   MERLIN AI — GEMINI UI CLIENT ENGINE & STREAMING LOGIC
   ========================================================================== */

let activeConversationId = null;
let currentMode = "assistant";
let activeEventSource = null;
let attachedFile = null;
let recognition = null;
let isDictating = false;

// Theme Engine (Dark & Light)
function initTheme() {
  const saved = localStorage.getItem("merlin-theme") || "dark";
  setTheme(saved);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme");
  const next = current === "dark" ? "light" : "dark";
  setTheme(next);
}

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("merlin-theme", theme);
  const icon = document.getElementById("theme-icon");
  if (icon) {
    icon.innerHTML = theme === "dark" 
      ? '<path d="M12 3c-4.97 0-9 4.03-9 9s4.03 9 9 9 9-4.03 9-9c0-.46-.04-.92-.1-1.36-.98 1.37-2.58 2.26-4.4 2.26-2.98 0-5.4-2.42-5.4-5.4 0-1.81.89-3.42 2.26-4.4-.44-.06-.9-.1-1.36-.1z"/>'
      : '<path d="M12 7c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5zM2 13h2c.55 0 1-.45 1-1s-.45-1-1-1H2c-.55 0-1 .45-1 1s.45 1 1 1zm18 0h2c.55 0 1-.45 1-1s-.45-1-1-1h-2c-.55 0-1 .45-1 1s.45 1 1 1zM11 2v2c0 .55.45 1 1 1s1-.45 1-1V2c0-.55-.45-1-1-1s-1 .45-1 1zm0 18v2c0 .55.45 1 1 1s1-.45 1-1v-2c0-.55-.45-1-1-1s-1 .45-1 1zM5.99 4.58c-.39-.39-1.03-.39-1.41 0s-.39 1.03 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0s.39-1.03 0-1.41L5.99 4.58zm12.37 12.37c-.39-.39-1.03-.39-1.41 0s-.39 1.03 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0s.39-1.03 0-1.41l-1.06-1.06zm1.06-10.96c.39-.39.39-1.03 0-1.41s-1.03-.39-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41s1.03.39 1.41 0l1.06-1.06zM7.05 18.36c.39-.39.39-1.03 0-1.41s-1.03-.39-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41s1.03.39 1.41 0l1.06-1.06z"/>';
  }
}

// Collapsible Sidebar
function toggleSidebar() {
  const sb = document.getElementById("sidebar");
  if (sb) sb.classList.toggle("collapsed");
}

// Mode Selection
function setMode(mode) {
  currentMode = mode;
  document.querySelectorAll(".mode-pill").forEach(p => p.classList.remove("active"));
  const target = document.getElementById(`pill-${mode}`);
  if (target) target.classList.add("active");
}

// Auto-Expanding Textarea
function autoResizeTextarea(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 180) + "px";
  const btn = document.getElementById("btn-send");
  if (btn) {
    btn.disabled = !textarea.value.trim() && !attachedFile;
  }
}

function handleKeyDown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    submitMessage();
  }
}

// Modal Helpers
function openModal(id) {
  const m = document.getElementById(id);
  if (m) m.style.display = "flex";
  if (id === "settings-modal") {
    loadTelemetry();
    loadModels();
  }
}

function closeModal(id) {
  const m = document.getElementById(id);
  if (m) m.style.display = "none";
}

// Rich Markdown and LaTeX Math Parser
function renderMarkdown(rawText) {
  if (!rawText) return "";
  let txt = rawText
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // DeepSeek-R1 <think>...</think> reasoning blocks
  if (txt.includes("&lt;think&gt;")) {
    txt = txt.replace(/&lt;think&gt;([\s\S]*?)&lt;\/think&gt;/g, (m, thought) => {
      return `<details class="think-block" style="background:var(--bg-surface);border:1px solid var(--border-subtle);border-radius:10px;padding:8px 12px;margin:8px 0;font-size:0.85rem;color:var(--text-muted);"><summary style="cursor:pointer;font-weight:500;">💭 Reasoning Process</summary><div style="margin-top:6px;line-height:1.4;">${thought.trim().replace(/\n/g, "<br>")}</div></details>`;
    });
    txt = txt.replace(/&lt;think&gt;([\s\S]*)/g, (m, thought) => {
      return `<div class="think-block-streaming" style="background:var(--bg-surface);border-left:3px solid var(--border-focus);padding:6px 10px;margin:8px 0;font-size:0.85rem;color:var(--text-muted);font-style:italic;">💭 Thinking: ${thought.trim().replace(/\n/g, "<br>")}</div>`;
    });
  }

  // Code blocks: ```lang\ncode\n```
  txt = txt.replace(/```([a-zA-Z0-9_\-\+]*)\n([\s\S]*?)```/g, (match, lang, code) => {
    const langLabel = lang.trim() || "code";
    return `
      <div class="code-container">
        <div class="code-header-bar">
          <span>${langLabel}</span>
          <button class="copy-btn" onclick="copyCode(this)">
            <svg style="width:13px;height:13px;fill:currentColor;" viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>
            Copy
          </button>
        </div>
        <pre><code>${code.trim()}</code></pre>
      </div>
    `;
  });

  // Display math $$...$$
  txt = txt.replace(/\$\$([\s\S]*?)\$\$/g, '<div class="math-display">$1</div>');

  // Inline math $...$
  txt = txt.replace(/\$([^\$\n]+)\$/g, '<span class="math-inline">$1</span>');

  // Inline code `...`
  txt = txt.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Bold & Italic
  txt = txt.replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
  txt = txt.replace(/\*([^*]+)\*/g, '<i>$1</i>');

  // Headings
  txt = txt.replace(/^### (.*$)/gim, '<h3 style="margin:14px 0 6px 0;font-size:1.05rem;">$1</h3>');
  txt = txt.replace(/^## (.*$)/gim, '<h2 style="margin:16px 0 8px 0;font-size:1.2rem;">$1</h2>');
  txt = txt.replace(/^# (.*$)/gim, '<h1 style="margin:20px 0 10px 0;font-size:1.35rem;">$1</h1>');

  // Blockquotes
  txt = txt.replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>');

  // Process breaks outside code blocks
  const parts = txt.split(/(<div class="code-container">[\s\S]*?<\/div>)/g);
  return parts.map(p => {
    if (p.startsWith('<div class="code-container">')) return p;
    return p.replace(/\n/g, "<br>");
  }).join("");
}

function copyCode(btn) {
  const pre = btn.closest(".code-container").querySelector("pre code");
  navigator.clipboard.writeText(pre.innerText).then(() => {
    btn.innerHTML = "✓ Copied";
    setTimeout(() => {
      btn.innerHTML = `<svg style="width:13px;height:13px;fill:currentColor;" viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg> Copy`;
    }, 2000);
  });
}

// Session & Conversation Management
async function loadConversations() {
  try {
    const res = await fetch("/api/conversations");
    const convs = await res.json();
    const list = document.getElementById("conversations-list");
    if (!list) return;
    list.innerHTML = "";

    if (!convs.length) {
      list.innerHTML = '<div style="padding:12px;font-size:0.75rem;color:var(--text-muted);">No saved chats yet.</div>';
      return;
    }

    convs.forEach(c => {
      const item = document.createElement("div");
      item.className = `conv-item ${c.id === activeConversationId ? 'active' : ''} ${c.pinned ? 'pinned' : ''}`;
      item.onclick = (e) => {
        if (e.target.closest(".conv-action-btn")) return;
        switchConversation(c.id);
      };
      item.innerHTML = `
        <div class="conv-title">${c.title || "New Chat"}</div>
        <div class="conv-actions">
          <button class="conv-action-btn" title="Pin / Unpin" onclick="togglePin('${c.id}', ${c.pinned ? 0 : 1})">${c.pinned ? '★' : '☆'}</button>
          <button class="conv-action-btn" title="Delete" onclick="deleteConv('${c.id}')">✕</button>
        </div>
      `;
      list.appendChild(item);
    });
  } catch (e) {
    console.error("Failed loading conversations:", e);
  }
}

async function createNewChat() {
  const res = await fetch("/api/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "New Chat", mode: currentMode })
  });
  const conv = await res.json();
  activeConversationId = conv.id;
  loadConversations();
  showConversationMessages(conv.id, []);
}

async function switchConversation(convId) {
  activeConversationId = convId;
  loadConversations();
  const res = await fetch(`/api/conversations/${convId}`);
  const data = await res.json();
  if (data.conversation) {
    setMode(data.conversation.mode || "assistant");
    showConversationMessages(convId, data.messages || []);
  }
}

async function togglePin(convId, newPinned) {
  await fetch(`/api/conversations/${convId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pinned: newPinned })
  });
  loadConversations();
}

async function deleteConv(convId) {
  await fetch(`/api/conversations/${convId}/delete`, { method: "POST" });
  if (activeConversationId === convId) {
    activeConversationId = null;
    document.getElementById("hero-view").style.display = "flex";
    document.getElementById("messages-list").style.display = "none";
  }
  loadConversations();
}

async function clearCurrentChat() {
  if (activeConversationId) {
    await fetch(`/api/conversations/${activeConversationId}/clear`, { method: "POST" });
    showConversationMessages(activeConversationId, []);
  }
  closeModal("settings-modal");
}

function showConversationMessages(convId, messages) {
  const hero = document.getElementById("hero-view");
  const list = document.getElementById("messages-list");
  list.innerHTML = "";

  if (!messages || !messages.length) {
    hero.style.display = "flex";
    list.style.display = "none";
    return;
  }

  hero.style.display = "none";
  list.style.display = "flex";

  messages.forEach(m => {
    appendMessageElement(m.role, m.content);
  });
  scrollViewportToBottom();
}

function appendMessageElement(role, content) {
  const list = document.getElementById("messages-list");
  document.getElementById("hero-view").style.display = "none";
  list.style.display = "flex";

  const block = document.createElement("div");
  block.className = `message-block ${role}`;

  if (role === "assistant") {
    block.innerHTML = `
      <div class="assistant-header">
        <span class="sparkle-icon"></span>
        Merlin
      </div>
      <div class="bubble-content">${renderMarkdown(content)}</div>
    `;
  } else {
    block.innerHTML = `<div class="bubble-content">${renderMarkdown(content)}</div>`;
  }

  list.appendChild(block);
  scrollViewportToBottom();
  return block.querySelector(".bubble-content");
}

function scrollViewportToBottom() {
  const vp = document.getElementById("messages-viewport");
  if (vp) vp.scrollTop = vp.scrollHeight;
}

function promptHero(text) {
  const ta = document.getElementById("chat-textarea");
  ta.value = text;
  autoResizeTextarea(ta);
  submitMessage();
}

// Real-Time Token Streaming & Generation
async function submitMessage() {
  const ta = document.getElementById("chat-textarea");
  const promptText = ta.value.trim();
  if (!promptText && !attachedFile) return;

  if (!activeConversationId) {
    await createNewChat();
  }

  let payloadText = promptText;
  if (attachedFile && attachedFile.extracted_text) {
    payloadText = `--- [ATTACHED FILE: ${attachedFile.name}] ---\n${attachedFile.extracted_text}\n--- [END OF FILE] ---\n\n${promptText}`;
  }

  ta.value = "";
  autoResizeTextarea(ta);
  clearAttachment();

  // Render User Message
  appendMessageElement("user", promptText);

  // Save to persistent database
  fetch(`/api/conversations/${activeConversationId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role: "user", content: payloadText })
  });

  // Prepare Assistant UI
  const assistantBubble = appendMessageElement("assistant", "");
  assistantBubble.innerHTML = '<span class="status-thinking" style="color:var(--text-muted);font-size:0.88rem;display:inline-flex;align-items:center;gap:6px;">⚡ Generating response...</span>';
  assistantBubble.classList.add("streaming-cursor");
  toggleStreamingUI(true);

  let fullResponse = "";
  const streamUrl = `/api/chat/stream?prompt=${encodeURIComponent(payloadText)}&mode=${currentMode}&session_id=${activeConversationId}`;
  activeEventSource = new EventSource(streamUrl);

  activeEventSource.onmessage = function(event) {
    if (event.data === "[DONE]") {
      finishStreaming(assistantBubble, fullResponse);
      return;
    }
    try {
      const data = JSON.parse(event.data);
      if (data.chunk) {
        fullResponse += data.chunk;
        assistantBubble.innerHTML = renderMarkdown(fullResponse);
        scrollViewportToBottom();
      }
    } catch (e) {}
  };

  activeEventSource.onerror = function() {
    finishStreaming(assistantBubble, fullResponse);
  };
}

function toggleStreamingUI(isStreaming) {
  const sendBtn = document.getElementById("btn-send");
  const stopBtn = document.getElementById("btn-stop");
  if (sendBtn) sendBtn.style.display = isStreaming ? "none" : "flex";
  if (stopBtn) stopBtn.style.display = isStreaming ? "flex" : "none";
}

function stopGeneration() {
  if (activeEventSource) {
    activeEventSource.close();
    activeEventSource = null;
  }
  if (activeConversationId) {
    fetch("/api/chat/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: activeConversationId })
    });
  }
  document.querySelectorAll(".streaming-cursor").forEach(c => c.classList.remove("streaming-cursor"));
  toggleStreamingUI(false);
}

function finishStreaming(bubbleEl, fullResponse) {
  if (activeEventSource) {
    activeEventSource.close();
    activeEventSource = null;
  }
  bubbleEl.classList.remove("streaming-cursor");
  bubbleEl.innerHTML = renderMarkdown(fullResponse);
  toggleStreamingUI(false);

  if (activeConversationId && fullResponse) {
    fetch(`/api/conversations/${activeConversationId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role: "assistant", content: fullResponse })
    });
  }
  loadConversations();
}

// File Attachment Handler
function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = async () => {
    const base64Data = reader.result;
    const res = await fetch("/api/upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: file.name, data: base64Data })
    });
    const d = await res.json();
    if (d.success) {
      attachedFile = d.file;
      document.getElementById("attached-file-name").innerText = `${file.name} (${d.file.size_str})`;
      document.getElementById("attachment-tray").style.display = "flex";
      document.getElementById("btn-send").disabled = false;
    }
  };
  reader.readAsDataURL(file);
}

function clearAttachment() {
  attachedFile = null;
  document.getElementById("attachment-tray").style.display = "none";
  document.getElementById("file-uploader").value = "";
  autoResizeTextarea(document.getElementById("chat-textarea"));
}

// Voice Dictation
function toggleDictation() {
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    alert("Speech Recognition is not supported in this browser.");
    return;
  }
  if (isDictating) {
    recognition.stop();
    return;
  }

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = "en-US";

  recognition.onstart = () => {
    isDictating = true;
    document.getElementById("btn-mic").style.color = "var(--danger)";
  };
  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    const ta = document.getElementById("chat-textarea");
    ta.value = (ta.value + " " + transcript).trim();
    autoResizeTextarea(ta);
  };
  recognition.onend = () => {
    isDictating = false;
    document.getElementById("btn-mic").style.color = "var(--text-muted)";
  };
  recognition.start();
}

// Hardware Telemetry & Model Management
async function loadTelemetry() {
  try {
    const res = await fetch("/api/system");
    const d = await res.json();
    document.getElementById("diag-temp").innerText = d.temperature;
    document.getElementById("diag-cpu").innerText = d.cpu_usage;
    document.getElementById("diag-mem").innerText = d.memory;
    document.getElementById("diag-disk").innerText = d.disk;
  } catch (e) {}
}

async function loadModels() {
  try {
    const res = await fetch("/api/models");
    const models = await res.json();
    const sel = document.getElementById("model-select");
    sel.innerHTML = "";
    if (!models.length) {
      sel.innerHTML = '<option value="">No local .gguf models found</option>';
      return;
    }
    models.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.filename;
      opt.innerText = `${m.name} (${m.size_mb} MB) ${m.is_active ? '✓ Active' : ''}`;
      if (m.is_active) opt.selected = true;
      sel.appendChild(opt);
    });
  } catch (e) {}
}

async function switchModel(val) {
  if (!val) return;
  await fetch("/api/models/active", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: val })
  });
  loadModels();
}

// Teacher Sync & Hugging Face Datasets
async function startTeacherSync() {
  const apiKey = document.getElementById("teacher-api-key").value.trim();
  const topic = document.getElementById("teacher-topic").value.trim();
  if (!apiKey) return alert("Please enter an API Key.");
  closeModal("sync-modal");
  alert("Teacher distillation started in the background. Learned skills will be indexed into local memory.");
  await fetch("/api/sync", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: apiKey, topic })
  });
}

async function startDatasetImport() {
  const dataset = document.getElementById("hf-dataset-select").value;
  const limit = parseInt(document.getElementById("hf-limit").value) || 50;
  closeModal("dataset-modal");
  alert(`Streaming ingestion started for ${dataset}.`);
  await fetch("/api/dataset/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset, limit })
  });
}

// Bootstrap
window.addEventListener("DOMContentLoaded", () => {
  initTheme();
  loadConversations();
});
