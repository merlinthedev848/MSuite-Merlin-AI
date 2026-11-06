/* ==========================================================================
   MERLIN AI — UI CLIENT ENGINE & STREAMING LOGIC
   ========================================================================== */

let activeConversationId = null;
let currentMode = "auto";
let activeEventSource = null;
let activeWebSocket = null;
let attachedFile = null;
let recognition = null;
let isDictating = false;
let syncPollTimer = null;
let datasetPollTimer = null;
let skillsSearchTimer = null;
let popularDatasetsLoaded = false;
let currentUser = null;
let currentAuthTab = "login";
const notesById = new Map();

function getAuthToken() {
  return localStorage.getItem("merlin_auth_token") || "";
}

function authHeaders(extraHeaders = {}) {
  const token = getAuthToken();
  const headers = { "Content-Type": "application/json", ...extraHeaders };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  return headers;
}

// HTML Escaping Utility (Prevents XSS in dynamic titles, notes, tasks, and skills)
function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// Non-blocking Toast Notification Engine
function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = `toast-item toast-${type}`;
  toast.innerText = message;
  toast.setAttribute("role", "alert");
  toast.title = "Click to dismiss";
  toast.onclick = () => {
    toast.classList.add("toast-fade-out");
    setTimeout(() => toast.remove(), 150);
  };
  container.appendChild(toast);
  setTimeout(() => {
    if (toast.parentElement) {
      toast.classList.add("toast-fade-out");
      setTimeout(() => toast.remove(), 250);
    }
  }, 3600);
}

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

// Collapsible Sidebar with Mobile Backdrop Support
function openSidebar() {
  const sb = document.getElementById("sidebar");
  const backdrop = document.getElementById("sidebar-backdrop");
  if (sb) {
    sb.classList.remove("collapsed");
    document.body.classList.add("sidebar-open");
    if (backdrop && window.innerWidth <= 768) {
      backdrop.style.display = "block";
    }
  }
}

function closeSidebar() {
  const sb = document.getElementById("sidebar");
  const backdrop = document.getElementById("sidebar-backdrop");
  if (sb) {
    sb.classList.add("collapsed");
    document.body.classList.remove("sidebar-open");
    if (backdrop) {
      backdrop.style.display = "none";
    }
  }
}

function toggleSidebar() {
  const sb = document.getElementById("sidebar");
  if (!sb) return;
  if (sb.classList.contains("collapsed")) {
    openSidebar();
  } else {
    closeSidebar();
  }
}

// Auto-Expanding Textarea
function autoResizeTextarea(textarea) {
  if (!textarea) return;
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
  const adminModals = ["settings-modal", "sync-modal", "dataset-modal", "telephony-modal", "memory-modal", "pa-modal", "calendar-modal"];
  if (adminModals.includes(id) && (!currentUser || !currentUser.is_admin)) {
    showToast("Admin access required to open settings & system tools.", "warning");
    return;
  }
  if (window.innerWidth <= 768) {
    closeSidebar();
  }
  const m = document.getElementById(id);
  if (m) m.style.display = "flex";
  if (id === "settings-modal") {
    loadTelemetry();
    loadModels();
    loadAuditLogs();
    const soundChk = document.getElementById("setting-sound-chimes");
    if (soundChk) {
      soundChk.checked = localStorage.getItem("merlin_sound_chimes") === "true";
    }
    const styleSel = document.getElementById("setting-voice-style");
    if (styleSel) {
      styleSel.value = localStorage.getItem("merlin_voice_style") || "freeman";
    }
    populateVoiceSelect();
  } else if (id === "pa-modal") {
    loadTodos();
    loadNotes();
  } else if (id === "memory-modal") {
    loadGaps();
    loadSkills();
  } else if (id === "sync-modal") {
    checkSyncStatus();
  } else if (id === "dataset-modal") {
    loadPopularDatasets();
    checkDatasetStatus();
  } else if (id === "argon-modal") {
    initArgonModal();
  } else if (id === "calendar-modal") {
    loadCalendarEvents();
  } else if (id === "telephony-modal") {
    loadTelephonyStatus();
  } else if (id === "auth-modal") {
    renderAuthModalContent();
  }
}

function closeModal(id) {
  const m = document.getElementById(id);
  if (m) m.style.display = "none";
  if (id === "sync-modal" && syncPollTimer) {
    clearTimeout(syncPollTimer);
    syncPollTimer = null;
  } else if (id === "dataset-modal" && datasetPollTimer) {
    clearTimeout(datasetPollTimer);
    datasetPollTimer = null;
  } else if (id === "argon-modal" && argonTimerInterval) {
    clearInterval(argonTimerInterval);
    argonTimerInterval = null;
  }
}

// Rich Markdown and LaTeX Math Parser (Preserves code blocks from prose regex corruption)
function renderMarkdown(rawText) {
  if (!rawText) return "";
  // Normalize Windows CRLF and CR line endings
  let txt = String(rawText).replace(/\r\n/g, "\n").replace(/\r/g, "\n");

  // Escape HTML entities to prevent XSS
  txt = txt
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

  // 1. Robust line-based extraction of fenced code blocks
  // Avoids regex delimiter-overlap bugs, supports CRLF, open/streaming fences, any backtick count (``` or ````), spaces, and any language tag
  const lines = txt.split("\n");
  const proseLines = [];
  const codeBlocks = [];
  let inCode = false;
  let codeFence = "";
  let currentLang = "";
  let currentCodeLines = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const fenceMatch = line.match(/^[ \t]*(`{3,})([a-zA-Z0-9_\-\+\#]*)[^\n`]*$/);

    if (!inCode) {
      if (fenceMatch) {
        inCode = true;
        codeFence = fenceMatch[1];
        currentLang = fenceMatch[2] || "code";
        currentCodeLines = [];
      } else {
        proseLines.push(line);
      }
    } else {
      // Check if closing fence matches
      if (line.trim().startsWith(codeFence)) {
        inCode = false;
        const codeText = currentCodeLines.join("\n").replace(/\n+$/, "");
        const html = `<div class="code-container"><div class="code-header-bar"><span>${escapeHtml(currentLang)}</span><button class="copy-btn" onclick="copyCode(this)"><svg style="width:13px;height:13px;fill:currentColor;" viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>Copy</button></div><pre><code>${codeText}</code></pre></div>`;
        const idx = codeBlocks.push(html) - 1;
        proseLines.push(`\x00CODEBLOCK_${idx}\x00`);
      } else {
        currentCodeLines.push(line);
      }
    }
  }

  // Unclosed fence (e.g. active token stream or truncated generation)
  if (inCode) {
    const codeText = currentCodeLines.join("\n").replace(/\n+$/, "");
    const html = `<div class="code-container streaming-code"><div class="code-header-bar"><span>${escapeHtml(currentLang)}</span><button class="copy-btn" onclick="copyCode(this)"><svg style="width:13px;height:13px;fill:currentColor;" viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>Copy</button></div><pre><code>${codeText}</code></pre></div>`;
    const idx = codeBlocks.push(html) - 1;
    proseLines.push(`\x00CODEBLOCK_${idx}\x00`);
  }

  txt = proseLines.join("\n");

  // 2. Extract inline code spans into placeholders
  const inlineCodes = [];
  txt = txt.replace(/`([^`\n]+)`/g, (match, code) => {
    const idx = inlineCodes.push(`<code>${code}</code>`) - 1;
    return `\x00INLINECODE_${idx}\x00`;
  });

  // 3. Apply math, links, and markdown formatting only to non-code prose
  txt = txt.replace(/\$\$([\s\S]*?)\$\$/g, '<div class="math-display">$1</div>');
  txt = txt.replace(/\$([^\$\n]+)\$/g, '<span class="math-inline">$1</span>');
  txt = txt.replace(/\[([^\]]+)\]\((https?:\/\/[^\s\)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" style="color:var(--accent-primary);text-decoration:underline;">$1</a>');
  txt = txt.replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
  txt = txt.replace(/\*([^*\n]+)\*/g, '<i>$1</i>');
  txt = txt.replace(/^### (.*$)/gim, '<h3 style="margin:14px 0 6px 0;font-size:1.05rem;">$1</h3>');
  txt = txt.replace(/^## (.*$)/gim, '<h2 style="margin:16px 0 8px 0;font-size:1.2rem;">$1</h2>');
  txt = txt.replace(/^# (.*$)/gim, '<h1 style="margin:20px 0 10px 0;font-size:1.35rem;">$1</h1>');
  txt = txt.replace(/^&gt; (.*$)/gim, '<blockquote>$1</blockquote>');
  txt = txt.replace(/^[•\-\*]\s+(.*$)/gim, '• $1');
  txt = txt.replace(/\n/g, "<br>");

  // 4. Restore inline code and fenced code blocks untouched
  txt = txt.replace(/\x00INLINECODE_(\d+)\x00/g, (_, i) => inlineCodes[Number(i)]);
  txt = txt.replace(/\x00CODEBLOCK_(\d+)\x00/g, (_, i) => codeBlocks[Number(i)]);

  // Clean redundant line breaks around code blocks for clean margins
  txt = txt.replace(/(?:<br>)+\s*(<div class="code-container)/g, "$1");
  txt = txt.replace(/(<\/div>)\s*(?:<br>)+/g, "$1");

  return txt;
}

function copyToClipboard(text, onSuccess) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(() => {
      if (onSuccess) onSuccess();
    }).catch(() => {
      fallbackCopyToClipboard(text, onSuccess);
    });
  } else {
    fallbackCopyToClipboard(text, onSuccess);
  }
}

function fallbackCopyToClipboard(text, onSuccess) {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.left = "-9999px";
  ta.style.top = "-9999px";
  document.body.appendChild(ta);
  ta.focus();
  ta.select();
  try {
    document.execCommand("copy");
    if (onSuccess) onSuccess();
  } catch (err) {
    console.warn("Fallback copy failed:", err);
  } finally {
    document.body.removeChild(ta);
  }
}

function copyCode(btn) {
  const pre = btn.closest(".code-container").querySelector("pre code");
  if (!pre) return;
  copyToClipboard(pre.innerText, () => {
    btn.innerHTML = "✓ Copied";
    setTimeout(() => {
      btn.innerHTML = `<svg style="width:13px;height:13px;fill:currentColor;" viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg> Copy`;
    }, 2000);
  });
}

// ==========================================================================
// SESSION & CONVERSATION MANAGEMENT
// ==========================================================================
async function loadConversations() {
  try {
    const res = await fetch("/api/conversations", { headers: authHeaders() });
    const convs = await res.json();
    const list = document.getElementById("conversations-list");
    if (!list) return;
    list.innerHTML = "";

    if (!convs.length) {
      const msg = currentUser ? "No saved chats for this account yet." : "No guest chats yet.";
      list.innerHTML = `<div style="padding:12px;font-size:0.75rem;color:var(--text-muted);">${msg}</div>`;
      return;
    }

    convs.forEach(c => {
      const item = document.createElement("div");
      item.className = `conv-item ${c.id === activeConversationId ? 'active' : ''} ${c.pinned ? 'pinned' : ''}`;
      item.onclick = (e) => {
        if (e.target.closest(".conv-action-btn")) return;
        switchConversation(c.id);
      };
      const safeId = escapeHtml(c.id);
      const safeTitle = escapeHtml(c.title || "New Chat");
      item.innerHTML = `
        <div class="conv-title" title="${safeTitle}">${safeTitle}</div>
        <div class="conv-actions">
          <button class="conv-action-btn" title="Rename" onclick="renameConv('${safeId}')">✎</button>
          <button class="conv-action-btn" title="Pin / Unpin" onclick="togglePin('${safeId}', ${c.pinned ? 0 : 1})">${c.pinned ? '★' : '☆'}</button>
          <button class="conv-action-btn" title="Delete" onclick="deleteConv('${safeId}')">✕</button>
        </div>
      `;
      list.appendChild(item);
    });
  } catch (e) {
    console.error("Failed loading conversations:", e);
  }
}

async function createNewChat() {
  if (window.innerWidth <= 768) {
    closeSidebar();
  }
  const res = await fetch("/api/conversations", {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ title: "New Chat", mode: "auto" })
  });
  const conv = await res.json();
  activeConversationId = conv.id;
  await loadConversations();
  showConversationMessages(conv.id, []);
}

async function switchConversation(convId) {
  if (window.innerWidth <= 768) {
    closeSidebar();
  }
  if (activeEventSource) {
    stopGeneration();
  }
  activeConversationId = convId;
  loadConversations();
  try {
    const res = await fetch(`/api/conversations/${encodeURIComponent(convId)}`, { headers: authHeaders() });
    if (!res.ok) {
      showToast("Could not load conversation.", "danger");
      return;
    }
    const data = await res.json();
    if (data.conversation) {
      showConversationMessages(convId, data.messages || []);
    }
  } catch (err) {
    showToast("Network error loading conversation.", "danger");
  }
}

async function renameConv(convId) {
  const newTitle = prompt("Enter a new title for this conversation:");
  if (!newTitle || !newTitle.trim()) return;
  await fetch(`/api/conversations/${encodeURIComponent(convId)}`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ title: newTitle.trim() })
  });
  loadConversations();
}

async function togglePin(convId, newPinned) {
  await fetch(`/api/conversations/${encodeURIComponent(convId)}`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ pinned: newPinned })
  });
  loadConversations();
}

async function deleteConv(convId) {
  if (!confirm("Are you sure you want to delete this conversation?")) return;
  await fetch(`/api/conversations/${encodeURIComponent(convId)}/delete`, {
    method: "POST",
    headers: authHeaders()
  });
  if (activeConversationId === convId) {
    activeConversationId = null;
    document.getElementById("hero-view").style.display = "flex";
    document.getElementById("messages-list").style.display = "none";
  }
  loadConversations();
}

async function clearCurrentChat() {
  if (activeConversationId) {
    await fetch(`/api/conversations/${encodeURIComponent(activeConversationId)}/clear`, {
      method: "POST",
      headers: authHeaders()
    });
    showConversationMessages(activeConversationId, []);
    showToast("Conversation history cleared.", "info");
  }
  closeModal("settings-modal");
}

function formatDisplayedUserMessage(content) {
  if (!content) return "";
  if (content.includes("--- [ATTACHED FILE:") && content.includes("--- [END OF")) {
    const fileMatch = content.match(/---\s*\[ATTACHED FILE:\s*([^\]]+)\]/);
    const parts = content.split(/---\s*\[END OF[^\]]*\]\s*---/);
    const userPrompt = (parts[1] || "").trim();
    const fileName = fileMatch ? fileMatch[1].trim() : "Attached File";
    return userPrompt ? `📎 **${fileName}**\n\n${userPrompt}` : `📎 **${fileName}**`;
  }
  return content;
}

function showConversationMessages(convId, messages) {
  const hero = document.getElementById("hero-view");
  const list = document.getElementById("messages-list");
  list.innerHTML = "";

  if (hero) hero.style.display = "flex"; // Tesla Sentinel Eye is ALWAYS front and center!

  if (!messages || !messages.length) {
    list.style.display = "none";
    return;
  }

  if (document.body && document.body.classList.contains("voice-mode-active")) {
    list.style.display = "none";
  } else {
    list.style.display = "flex";
  }

  messages.forEach(m => {
    const displayContent = m.role === "user" ? formatDisplayedUserMessage(m.content) : m.content;
    appendMessageElement(m.role, displayContent);
  });
  scrollViewportToBottom();
}

function appendMessageElement(role, content) {
  const list = document.getElementById("messages-list");
  const hero = document.getElementById("hero-view");
  if (hero) hero.style.display = "flex"; // Cyber AI HUD remains permanently centered!
  if (list && (!document.body || !document.body.classList.contains("voice-mode-active"))) {
    list.style.display = "flex";
  }

  const block = document.createElement("div");
  block.className = `message-block ${role}`;

  if (role === "assistant") {
    block.innerHTML = `
      <div class="assistant-header">
        <svg class="gemini-sparkle-star" viewBox="0 0 24 24" style="width:18px;height:18px;flex-shrink:0;">
          <path fill="url(#gemini-logo-grad)" d="M12 24c-.2 0-.4-.1-.5-.2C11.3 22 9.5 15.6 7.5 13.5 5.4 11.5 0 9.7 0 9.5s5.4-2 7.5-4.1C9.5 3.3 11.3 0 11.5 0s2 3.3 4.1 5.4c2.1 2.1 7.5 3.9 7.5 4.1s-5.4 2-7.5 4.1C13.5 15.6 11.7 22 11.5 24c-.1.1-.3.2-.5.2z"/>
        </svg>
        <span>Merlin</span>
      </div>
      <div class="bubble-content">${renderMarkdown(content)}</div>
      ${content ? `
        <div class="assistant-actions-bar">
          <button class="msg-action-btn" onclick="copyMessageText(this)" title="Copy message">
            <svg viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>
            <span>Copy</span>
          </button>
          <button class="msg-action-btn" onclick="likeMessage(this)" title="Good response">
            <svg viewBox="0 0 24 24"><path d="M1 21h4V9H1v12zm22-11c0-1.1-.9-2-2-2h-6.31l.95-4.57.03-.32c0-.41-.17-.79-.44-1.06L14.17 1 7.59 7.59C7.22 7.95 7 8.45 7 9v10c0 1.1.9 2 2 2h9c.83 0 1.54-.5 1.84-1.22l3.02-7.05c.09-.23.14-.47.14-.73v-2z"/></svg>
          </button>
        </div>` : ''}
    `;
  } else {
    block.innerHTML = `<div class="bubble-content">${renderMarkdown(content)}</div>`;
  }

  list.appendChild(block);
  scrollViewportToBottom();
  return block.querySelector(".bubble-content");
}

function appendMessageActions(bubbleEl, text) {
  if (!text || !bubbleEl) return;
  const parent = bubbleEl.parentElement;
  if (!parent.querySelector(".assistant-actions-bar")) {
    const bar = document.createElement("div");
    bar.className = "assistant-actions-bar";
    bar.innerHTML = `
      <button class="msg-action-btn" onclick="copyMessageText(this)" title="Copy response">
        <svg viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>
        <span>Copy</span>
      </button>
      <button class="msg-action-btn" onclick="likeMessage(this)" title="Good response">
        <svg viewBox="0 0 24 24"><path d="M1 21h4V9H1v12zm22-11c0-1.1-.9-2-2-2h-6.31l.95-4.57.03-.32c0-.41-.17-.79-.44-1.06L14.17 1 7.59 7.59C7.22 7.95 7 8.45 7 9v10c0 1.1.9 2 2 2h9c.83 0 1.54-.5 1.84-1.22l3.02-7.05c.09-.23.14-.47.14-.73v-2z"/></svg>
      </button>
    `;
    parent.appendChild(bar);
  }
}

function copyMessageText(btn) {
  const bubble = btn.closest(".message-block").querySelector(".bubble-content");
  if (bubble) {
    copyToClipboard(bubble.innerText, () => {
      const span = btn.querySelector("span");
      if (span) span.innerText = "Copied!";
      setTimeout(() => { if (span) span.innerText = "Copy"; }, 2000);
    });
  }
}

function likeMessage(btn) {
  btn.style.color = "var(--accent-primary)";
  showToast("Feedback recorded!", "success");
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

// ==========================================================================
// REAL-TIME STREAMING TTS PIPELINE (SUB-500MS SENTENCE CHUNKING)
// ==========================================================================
let ttsStreamingQueue = [];
let ttsIsSpeakingChunk = false;
let ttsTextBuffer = "";
let ttsStreamActive = false;

function resetSpeechStream() {
  ttsStreamingQueue = [];
  ttsIsSpeakingChunk = false;
  ttsTextBuffer = "";
  ttsStreamActive = false;
  if ('speechSynthesis' in window) {
    try { window.speechSynthesis.cancel(); } catch(e) {}
  }
  const cap = document.getElementById("sentinel-voice-caption");
  if (cap) {
    cap.style.display = "none";
    cap.textContent = "";
  }
}

function processIncomingSpeechChunk(tokenChunk, isFinal = false) {
  if (!isMerlinVoiceActive && !isJarvisVoiceActive) return;
  ttsStreamActive = !isFinal;
  if (tokenChunk) ttsTextBuffer += tokenChunk;

  // Real-time sentence chunking: detect punctuation (. ! ?) followed by whitespace or line breaks
  while (ttsTextBuffer.length > 0) {
    const match = ttsTextBuffer.match(/^([\s\S]*?[.!?]+(?:\s+|\n+))([\s\S]*)$/);
    if (match) {
      const sentence = match[1].trim();
      ttsTextBuffer = match[2];
      if (sentence) queueSentenceForSpeech(sentence);
      continue;
    }
    // Clause break fallback for long streaming segments (>80 chars) without punctuation
    if (ttsTextBuffer.length > 80) {
      const clauseMatch = ttsTextBuffer.match(/^([\s\S]*?[,;:\n]+(?:\s+|\n+))([\s\S]*)$/);
      if (clauseMatch) {
        const clause = clauseMatch[1].trim();
        ttsTextBuffer = clauseMatch[2];
        if (clause) queueSentenceForSpeech(clause);
        continue;
      }
    }
    break;
  }

  // On stream done, flush any remaining text in the buffer immediately
  if (isFinal && ttsTextBuffer.trim()) {
    queueSentenceForSpeech(ttsTextBuffer.trim());
    ttsTextBuffer = "";
  }
}

function queueSentenceForSpeech(sentence) {
  const clean = prepareTextForSpeech(sentence);
  if (!clean) return;
  ttsStreamingQueue.push(clean);
  playNextSpeechChunk();
}

function playNextSpeechChunk() {
  if (ttsIsSpeakingChunk || ttsStreamingQueue.length === 0) return;
  if (!('speechSynthesis' in window)) return;

  const nextSentence = ttsStreamingQueue.shift();
  if (!nextSentence) return;

  ttsIsSpeakingChunk = true;
  isMerlinSpeaking = true;
  lastSpokenText = nextSentence.toLowerCase();
  lastSpokenTime = Date.now();
  setMerlinHudState("speaking");

  // Show live subtitle caption directly on HUD
  const captionEl = document.getElementById("sentinel-voice-caption");
  if (captionEl) {
    captionEl.style.display = "inline-block";
    captionEl.innerHTML = `&bull; MERLIN: "${escapeHtml(nextSentence)}"`;
  }
  const subEl = document.getElementById("sentinel-sub-text");
  if (subEl) {
    subEl.textContent = `MERLIN: "${nextSentence}"`;
  }

  // Instantly mute microphone while synthesizing speech to avoid acoustic feedback echo
  if (recognition) {
    try { recognition.abort(); } catch(e) {}
  }

  if (!bestVoice) updateBestVoice();

  const utterance = new SpeechSynthesisUtterance(nextSentence);
  if (bestVoice) utterance.voice = bestVoice;
  const voiceCfg = getVoiceStyleConfig();
  utterance.rate = voiceCfg.rate;
  utterance.pitch = voiceCfg.pitch;

  utterance.onend = () => {
    ttsIsSpeakingChunk = false;
    lastSpokenTime = Date.now();
    if (ttsStreamingQueue.length > 0) {
      playNextSpeechChunk();
    } else if (!ttsStreamActive) {
      // Entire streaming response completed!
      isMerlinSpeaking = false;
      refreshConversationSession(35000);
      setMerlinHudState("active_dialogue");
      updateWakePillUI(true, "Active Dialogue (Speak freely)");
      // Re-enable microphone after room echo clears (650ms)
      setTimeout(() => {
        if (isMerlinVoiceActive && !isMerlinSpeaking && !isRecognitionRunning) {
          startMerlinListening();
        }
      }, 650);
    }
  };

  utterance.onerror = (err) => {
    console.warn("TTS utterance stream error:", err);
    ttsIsSpeakingChunk = false;
    if (ttsStreamingQueue.length > 0) {
      playNextSpeechChunk();
    } else {
      isMerlinSpeaking = false;
      startMerlinListening();
    }
  };

  window.speechSynthesis.speak(utterance);
}

// ==========================================================================
// REAL-TIME TOKEN STREAMING & GENERATION
// ==========================================================================
async function submitMessage(fromVoice = false) {
  const ta = document.getElementById("chat-textarea");
  const promptText = ta.value.trim();
  if (!promptText && !attachedFile) return;

  if (!activeConversationId) {
    await createNewChat();
  }

  const isVoiceMode = isMerlinVoiceActive || fromVoice;
  resetSpeechStream();

  const currentAttachment = attachedFile;
  let payloadText = promptText;
  let displayUserText = promptText;

  if (currentAttachment) {
    const defaultQuestion = `Please analyze the attached ${currentAttachment.category} file (${currentAttachment.name}).`;
    const effectiveQuestion = promptText || defaultQuestion;
    displayUserText = promptText
      ? `📎 **${currentAttachment.name}** (${currentAttachment.size_str})\n\n${promptText}`
      : `📎 **${currentAttachment.name}** (${currentAttachment.size_str})`;

    if (currentAttachment.extracted_text) {
      payloadText = `--- [ATTACHED FILE: ${currentAttachment.name}] ---\n${currentAttachment.extracted_text}\n--- [END OF FILE] ---\n\n${effectiveQuestion}`;
    } else {
      const dimInfo = currentAttachment.dimensions ? `, Dimensions: ${currentAttachment.dimensions}` : "";
      payloadText = `--- [ATTACHED FILE: ${currentAttachment.name} (Category: ${currentAttachment.category}, Size: ${currentAttachment.size_str}${dimInfo})] ---\n[Binary/Media Attachment URL: ${currentAttachment.url}]\n--- [END OF FILE] ---\n\n${effectiveQuestion}`;
    }
  }

  ta.setAttribute("data-last-prompt", promptText);
  ta.value = "";
  clearAttachment();
  autoResizeTextarea(ta);
  dismissJarvisHudDisplay();

  let assistantBubble;
  if (!isVoiceMode) {
    // Normal Text Mode: Render message bubbles in scrolling conversation list
    appendMessageElement("user", displayUserText);
    assistantBubble = appendMessageElement("assistant", "");
    assistantBubble.innerHTML = '<span class="status-thinking" style="color:var(--text-muted);font-size:0.88rem;display:inline-flex;align-items:center;gap:6px;">⚡ Generating response...</span>';
    assistantBubble.classList.add("streaming-cursor");
  } else {
    // Full Voice Mode: Pure Cyber HUD! Hide scrolling message list & all text subtitles for True Jarvis feel
    const statusEl = document.getElementById("sentinel-status-text");
    const subEl = document.getElementById("sentinel-sub-text");
    const cap = document.getElementById("sentinel-voice-caption");
    if (statusEl) statusEl.style.display = "none";
    if (subEl) subEl.style.display = "none";
    if (cap) cap.style.display = "none";
    // Detached dummy element so markdown rendering / backend DB logging function seamlessly
    assistantBubble = document.createElement("div");
  }

  toggleStreamingUI(true);

  let fullResponse = "";
  let renderScheduled = false;

  function scheduleRender() {
    if (!renderScheduled) {
      renderScheduled = true;
      requestAnimationFrame(() => {
        if (!isVoiceMode) {
          assistantBubble.innerHTML = renderMarkdown(fullResponse);
          scrollViewportToBottom();
        }
        renderScheduled = false;
      });
    }
  }

  let streamMode = "ws";
  let wsStarted = false;
  let sseStarted = false;

  function startSSEStream() {
    if (sseStarted) return;
    sseStarted = true;
    streamMode = "sse";
    if (activeWebSocket) {
      try { activeWebSocket.close(); } catch(e) {}
      activeWebSocket = null;
    }
    if (activeEventSource) {
      activeEventSource.close();
      activeEventSource = null;
    }
    const tokenParam = getAuthToken() ? `&token=${encodeURIComponent(getAuthToken())}` : "";
    const activeMode = (isMerlinVoiceActive || isJarvisVoiceActive) ? "merlin" : "auto";
    const streamUrl = `/api/chat/stream?prompt=${encodeURIComponent(payloadText)}&mode=${activeMode}&session_id=${encodeURIComponent(activeConversationId)}${tokenParam}`;
    activeEventSource = new EventSource(streamUrl);

    activeEventSource.onmessage = function(event) {
      if (event.data === "[DONE]") {
        processIncomingSpeechChunk("", true);
        finishStreaming(assistantBubble, fullResponse, isVoiceMode);
        return;
      }
      try {
        const data = JSON.parse(event.data);
        if (data.chunk) {
          fullResponse += data.chunk;
          scheduleRender();
          if (isVoiceMode) {
            processIncomingSpeechChunk(data.chunk, false);
          }
        }
      } catch (e) {}
    };

    activeEventSource.onerror = function() {
      if (activeEventSource) {
        activeEventSource.close();
        activeEventSource = null;
      }
      processIncomingSpeechChunk("", true);
      finishStreaming(assistantBubble, fullResponse, isVoiceMode);
    };
  }

  // Attempt zero-overhead WebSocket streaming first, with instantaneous fallback to SSE
  if (activeWebSocket) {
    try { activeWebSocket.close(); } catch(e) {}
    activeWebSocket = null;
  }
  if (activeEventSource) {
    try { activeEventSource.close(); } catch(e) {}
    activeEventSource = null;
  }

  try {
    const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProto}//${window.location.host}/ws/chat`;
    const ws = new WebSocket(wsUrl);
    activeWebSocket = ws;

    ws.onopen = function() {
      if (sseStarted) {
        try { ws.close(); } catch(e) {}
        return;
      }
      wsStarted = true;
      const activeMode = (isMerlinVoiceActive || isJarvisVoiceActive) ? "merlin" : "auto";
      ws.send(JSON.stringify({
        prompt: payloadText,
        session_id: activeConversationId,
        mode: activeMode,
        token: getAuthToken()
      }));
    };

    ws.onmessage = function(event) {
      if (sseStarted) return;
      try {
        const data = JSON.parse(event.data);
        if (data.type === "chunk" && data.delta) {
          fullResponse += data.delta;
          scheduleRender();
          if (isVoiceMode) {
            processIncomingSpeechChunk(data.delta, false);
          }
        } else if (data.type === "done") {
          processIncomingSpeechChunk("", true);
          finishStreaming(assistantBubble, fullResponse, isVoiceMode);
        } else if (data.type === "error") {
          fullResponse += `\n[${data.message}]`;
          scheduleRender();
        }
      } catch (e) {}
    };

    ws.onerror = function() {
      if (!wsStarted && !sseStarted) {
        activeWebSocket = null;
        startSSEStream();
      } else if (!sseStarted) {
        processIncomingSpeechChunk("", true);
        finishStreaming(assistantBubble, fullResponse, isVoiceMode);
      }
    };

    ws.onclose = function() {
      if (sseStarted) {
        return;
      }
      if (!wsStarted && !sseStarted) {
        activeWebSocket = null;
        startSSEStream();
        return;
      }
      if (activeWebSocket === ws) {
        activeWebSocket = null;
        processIncomingSpeechChunk("", true);
        finishStreaming(assistantBubble, fullResponse, isVoiceMode);
      }
    };
  } catch (err) {
    startSSEStream();
  }
}

function toggleStreamingUI(isStreaming) {
  const sendBtn = document.getElementById("btn-send");
  const stopBtn = document.getElementById("btn-stop");
  if (sendBtn) sendBtn.style.display = isStreaming ? "none" : "flex";
  if (stopBtn) stopBtn.style.display = isStreaming ? "flex" : "none";
}

function stopGeneration() {
  resetSpeechStream();
  if (window.speechSynthesis) {
    try { window.speechSynthesis.cancel(); } catch(e) {}
  }
  if (isMerlinVoiceActive || isJarvisVoiceActive) {
    setMerlinHudState("idle", "INTERRUPTED // STANDBY");
  }
  if (activeWebSocket) {
    try {
      activeWebSocket.send(JSON.stringify({ action: "stop", session_id: activeConversationId }));
      activeWebSocket.close();
    } catch(e) {}
    activeWebSocket = null;
  }
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

function finishStreaming(bubbleEl, fullResponse, isVoiceMode = false) {
  if (activeWebSocket) {
    try { activeWebSocket.close(); } catch(e) {}
    activeWebSocket = null;
  }
  if (activeEventSource) {
    activeEventSource.close();
    activeEventSource = null;
  }
  bubbleEl.classList.remove("streaming-cursor");
  if (!isVoiceMode) {
    if (!fullResponse || !fullResponse.trim()) {
      bubbleEl.innerHTML = '<span style="color:var(--text-muted);font-style:italic;">⚠️ No response generated by local inference engine. Ensure Ollama or local LLM server is running.</span>';
    } else {
      bubbleEl.innerHTML = renderMarkdown(fullResponse);
      appendMessageActions(bubbleEl, fullResponse);
    }
  } else {
    // True Jarvis on-demand display: ONLY project on screen when user asks to see code/data/lists/tables
    const displayPanel = document.getElementById("jarvis-hud-display");
    const ta = document.getElementById("chat-textarea");
    const lastPrompt = (ta && ta.getAttribute("data-last-prompt") ? ta.getAttribute("data-last-prompt") : "").toLowerCase();
    const hasCodeOrTable = fullResponse.includes("```") || fullResponse.includes("<table") || fullResponse.includes("|---");
    const userAskedToDisplay = /\b(show|display|render|write|draw|print|list|table|code|script|schema|json|html|css|sql|python|c\+\+|rust|javascript)\b/i.test(lastPrompt);

    if (!fullResponse || !fullResponse.trim()) {
      speakMerlinText("I was unable to retrieve a response from the inference engine, Sir. Please ensure the local model server is running.");
      if (displayPanel) displayPanel.style.display = "none";
    } else if ((hasCodeOrTable || userAskedToDisplay) && fullResponse && fullResponse.trim()) {
      if (displayPanel) {
        displayPanel.innerHTML = `
          <div class="hud-display-header">
            <div class="hud-display-title">
              <span class="hud-display-indicator"></span>
              <span>HUD VISUAL PROJECTION</span>
            </div>
            <button class="hud-display-close" onclick="dismissJarvisHudDisplay()" title="Close holographic display (or say 'dismiss')">✕ Close</button>
          </div>
          <div class="hud-display-content">${renderMarkdown(fullResponse)}</div>
        `;
        displayPanel.style.display = "block";
        appendMessageActions(displayPanel.querySelector(".hud-display-content"), fullResponse);
      }
    } else if (displayPanel) {
      displayPanel.style.display = "none";
    }
  }
  toggleStreamingUI(false);
  // Note: engine.generate_stream() already saves both the user prompt and the assistant response
  // in SQLite on the backend. Refreshing the conversation list updates the auto-derived title.
  loadConversations();
}

function dismissJarvisHudDisplay() {
  const displayPanel = document.getElementById("jarvis-hud-display");
  if (displayPanel) {
    displayPanel.style.display = "none";
  }
}

// ==========================================================================
// FILE ATTACHMENT & MULTIMODAL VISION/AUDIO HANDLER
// ==========================================================================
function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = async () => {
    try {
      const base64Data = reader.result;
      const res = await fetch("/api/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: file.name, data: base64Data })
      });
      let d = {};
      try {
        d = await res.json();
      } catch (_) {
        d = { detail: `Upload error (HTTP ${res.status})` };
      }
      if (res.ok && d.success) {
        attachedFile = d.file;
        document.getElementById("attached-file-name").innerText = `${file.name} (${d.file.size_str})`;
        document.getElementById("attachment-tray").style.display = "flex";
        const mmBtn = document.getElementById("btn-multimodal-analyze");
        if (mmBtn) {
          mmBtn.style.display = (d.file.category === "image" || d.file.category === "audio") ? "inline-flex" : "none";
        }
        document.getElementById("btn-send").disabled = false;
        showToast(`Attached ${file.name} (${d.file.category})`, "success");
      } else {
        showToast(d.detail || d.error || "File upload failed", "danger");
      }
    } catch (err) {
      showToast("Error uploading file", "danger");
    }
  };
  reader.readAsDataURL(file);
  if (e.target) e.target.value = "";
}

function clearAttachment() {
  attachedFile = null;
  const tray = document.getElementById("attachment-tray");
  if (tray) tray.style.display = "none";
  const mmBtn = document.getElementById("btn-multimodal-analyze");
  if (mmBtn) mmBtn.style.display = "none";
  const uploader = document.getElementById("file-uploader");
  if (uploader) uploader.value = "";
  autoResizeTextarea(document.getElementById("chat-textarea"));
}

async function analyzeAttachedMultimodal() {
  if (!attachedFile) return;
  const ta = document.getElementById("chat-textarea");
  const userPrompt = ta.value.trim() || `Analyze this ${attachedFile.category} file (${attachedFile.name}) thoroughly.`;
  let apiKey = document.getElementById("teacher-api-key")?.value?.trim() || localStorage.getItem("merlin-gemini-key") || "";
  if (!apiKey) {
    apiKey = prompt("Enter a Google Gemini API Key (AIza...) for deep multimodal vision/audio analysis:") || "";
    if (!apiKey.trim()) return;
    localStorage.setItem("merlin-gemini-key", apiKey.trim());
  }

  if (!activeConversationId) {
    await createNewChat();
  }

  const fileObj = attachedFile;
  ta.value = "";
  clearAttachment();

  const userDisplay = `📎 **${fileObj.name}** (${fileObj.size_str})\n\n${userPrompt}`;
  appendMessageElement("user", userDisplay);
  await fetch(`/api/conversations/${encodeURIComponent(activeConversationId)}/messages`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ role: "user", content: userDisplay })
  });

  const assistantBubble = appendMessageElement("assistant", "");
  assistantBubble.innerHTML = '<span class="status-thinking" style="color:var(--text-muted);font-size:0.88rem;">✨ Running multimodal vision/audio analysis...</span>';

  try {
    const res = await fetch("/api/analyze/multimodal", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ file_id: fileObj.id, prompt: userPrompt, api_key: apiKey })
    });
    const d = await res.json();
    const reply = d.success ? d.analysis : `⚠️ Multimodal analysis error: ${d.error || d.detail || "Unknown error"}`;
    assistantBubble.innerHTML = renderMarkdown(reply);
    await fetch(`/api/conversations/${encodeURIComponent(activeConversationId)}/messages`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ role: "assistant", content: reply })
    });
    loadConversations();
  } catch (err) {
    assistantBubble.innerHTML = renderMarkdown(`⚠️ Multimodal request failed: ${err}`);
  }
}

// ==========================================================================
// MERLIN AUDIO CHIME & ULTRA-SMOOTH SPOKEN VOICE ENGINE
// Autonomous Conversational Communicator & Engineering Workhorse
// ==========================================================================
let isMerlinVoiceActive = true;
let isJarvisVoiceActive = true; // Backward-compatibility alias
let isMerlinConversationActive = false; // Active continuous dialogue state
let conversationActiveTimer = null;
let isRecognitionRunning = false;
let merlinAudioCtx = null;
let isMerlinSpeaking = false;
let lastSpokenText = "";
let lastSpokenTime = 0;
let bestVoice = null;

function refreshConversationSession(durationMs = 35000) {
  isMerlinConversationActive = true;
  if (conversationActiveTimer) {
    clearTimeout(conversationActiveTimer);
  }
  setMerlinHudState("active_dialogue");
  updateWakePillUI(true, "Active Dialogue (Speak freely)");
  conversationActiveTimer = setTimeout(() => {
    isMerlinConversationActive = false;
    setMerlinHudState("listening");
    updateWakePillUI(true, "Wake Word: Merlin (Always Listening)");
    console.debug("Merlin active conversation session timed out; returning to wake-word standby.");
  }, durationMs);
}

function endConversationSession(spokenResponse = null) {
  isMerlinConversationActive = false;
  if (conversationActiveTimer) {
    clearTimeout(conversationActiveTimer);
    conversationActiveTimer = null;
  }
  setMerlinHudState("listening", 'STANDBY &bull; WAKE WORD: "MERLIN"');
  updateWakePillUI(true, "Wake Word: Merlin (Always Listening)");
  if (spokenResponse) {
    speakMerlinText(spokenResponse);
  } else {
    playMerlinChime("finish");
  }
}

function getVoiceStyleConfig() {
  let style = "freeman";
  try {
    if (typeof localStorage !== "undefined" && localStorage.getItem) {
      style = localStorage.getItem("merlin_voice_style") || "freeman";
    }
  } catch(e) {}

  if (style === "freeman") {
    // Warm, deep resonant baritone narration (Morgan Freeman style)
    return { pitch: 0.84, rate: 0.94 };
  } else if (style === "deep") {
    // Deep bass resonance
    return { pitch: 0.78, rate: 0.90 };
  } else if (style === "jarvis") {
    // British transatlantic AI
    return { pitch: 0.96, rate: 1.02 };
  }
  return { pitch: 0.84, rate: 0.94 };
}

function updateBestVoice() {
  if (!('speechSynthesis' in window)) return;
  const voices = window.speechSynthesis.getVoices();
  if (!voices || !voices.length) return;

  let savedVoiceURI = null;
  try {
    if (typeof localStorage !== "undefined" && localStorage.getItem) {
      savedVoiceURI = localStorage.getItem("merlin_selected_voice_uri");
    }
  } catch(e) {}

  if (savedVoiceURI) {
    const saved = voices.find(v => v.voiceURI === savedVoiceURI || v.name === savedVoiceURI);
    if (saved) {
      bestVoice = saved;
      populateVoiceSelect();
      return;
    }
  }

  // Common female name/indicator filter
  const femalePattern = /\b(female|zira|susan|hazel|catherine|linda|jenny|aria|samantha|victoria|karen|moira|fiona|tessa|ava|emma|olivia|sophia|chloe|sonia|natasha|ana)\b/i;

  const englishVoices = voices.filter(v => v.lang.startsWith("en") && !femalePattern.test(v.name));

  // Priority 1: Resonant American baritone male voices (Guy, Christopher, Andrew, David, Mark)
  const usNarratorMale = englishVoices.find(v => 
    (v.lang === "en-US" || v.lang === "en_US") &&
    (v.name.includes("Guy") || v.name.includes("Christopher") || v.name.includes("Andrew") || v.name.includes("David") || v.name.includes("Mark"))
  );
  if (usNarratorMale) { bestVoice = usNarratorMale; populateVoiceSelect(); return; }

  // Priority 2: Natural / Neural male English voices
  const neuralMale = englishVoices.find(v => 
    (v.name.includes("Natural") || v.name.includes("Neural") || v.name.includes("Online")) &&
    (/\b(male|guy|christopher|andrew|david|ryan|george|oliver|daniel)\b/i.test(v.name) || !femalePattern.test(v.name))
  );
  if (neuralMale) { bestVoice = neuralMale; populateVoiceSelect(); return; }

  // Priority 3: Any American English male voice (David, Mark, Alex, etc.)
  const anyUSMale = englishVoices.find(v => 
    v.lang.startsWith("en-US") &&
    (/\b(male|david|mark|alex|brian|eric|charles)\b/i.test(v.name) || !femalePattern.test(v.name))
  );
  if (anyUSMale) { bestVoice = anyUSMale; populateVoiceSelect(); return; }

  // Priority 4: High quality British male voices (Ryan, George, Arthur, Oliver, Daniel)
  const gbMale = englishVoices.find(v => 
    (v.lang === "en-GB" || v.lang === "en_GB") &&
    (v.name.includes("Ryan") || v.name.includes("George") || v.name.includes("Arthur") || v.name.includes("Oliver") || v.name.includes("Daniel") || v.name.includes("Male"))
  );
  if (gbMale) { bestVoice = gbMale; populateVoiceSelect(); return; }

  // Priority 5: Any non-female English voice
  if (englishVoices.length > 0) {
    bestVoice = englishVoices[0];
    populateVoiceSelect();
    return;
  }

  // Fallback
  bestVoice = voices.find(v => v.lang.startsWith("en")) || voices[0];
  populateVoiceSelect();
}

function populateVoiceSelect() {
  const select = document.getElementById("setting-voice-device");
  if (!select || !('speechSynthesis' in window)) return;
  const voices = window.speechSynthesis.getVoices();
  if (!voices || !voices.length) return;

  let currentVal = "";
  try {
    if (typeof localStorage !== "undefined" && localStorage.getItem) {
      currentVal = localStorage.getItem("merlin_selected_voice_uri") || "";
    }
  } catch(e) {}
  if (!currentVal && bestVoice) currentVal = bestVoice.voiceURI;

  select.innerHTML = '<option value="">Auto-Select Deep Male Voice</option>';

  const femalePattern = /\b(female|zira|susan|hazel|catherine|linda|jenny|aria|samantha|victoria|karen|moira|fiona|tessa|ava|emma|olivia|sophia|chloe|sonia|natasha|ana)\b/i;

  const sorted = [...voices].sort((a, b) => {
    const aMale = !femalePattern.test(a.name) && a.lang.startsWith("en");
    const bMale = !femalePattern.test(b.name) && b.lang.startsWith("en");
    if (aMale && !bMale) return -1;
    if (!aMale && bMale) return 1;
    return a.name.localeCompare(b.name);
  });

  for (const v of sorted) {
    const opt = document.createElement("option");
    opt.value = v.voiceURI;
    const isM = !femalePattern.test(v.name) ? " [Male]" : "";
    opt.textContent = `${v.name} (${v.lang})${isM}`;
    if (v.voiceURI === currentVal || (bestVoice && v.voiceURI === bestVoice.voiceURI)) {
      opt.selected = true;
    }
    select.appendChild(opt);
  }
}

function changeVoiceDevice(voiceURI) {
  try {
    if (typeof localStorage !== "undefined" && localStorage.setItem) {
      if (!voiceURI) localStorage.removeItem("merlin_selected_voice_uri");
      else localStorage.setItem("merlin_selected_voice_uri", voiceURI);
    }
  } catch(e) {}
  updateBestVoice();
  showToast("Synthesizer voice updated.", "info");
}

function changeVoiceStyle(style) {
  try {
    if (typeof localStorage !== "undefined" && localStorage.setItem) {
      localStorage.setItem("merlin_voice_style", style);
    }
  } catch(e) {}
  showToast(`Voice style set to ${style === "freeman" ? "Morgan Freeman (Deep Baritone)" : style}.`, "info");
}

function testMerlinVoice() {
  speakMerlinText("Good evening. I am Merlin, your autonomous artificial intelligence. All systems are operational.");
}

if ('speechSynthesis' in window) {
  window.speechSynthesis.onvoiceschanged = updateBestVoice;
  updateBestVoice();
}

function prepareTextForSpeech(raw) {
  if (!raw) return "";
  return raw
    .replace(/```[\s\S]*?```/g, "Code implementation complete, Sir.")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/https?:\/\/\S+/g, "link")
    .replace(/(\d+)\s*GB/gi, "$1 gigabytes")
    .replace(/(\d+)\s*MB/gi, "$1 megabytes")
    .replace(/\bCPU\b/g, "C P U")
    .replace(/\bRAM\b/g, "ram")
    .replace(/\bOS\b/g, "operating system")
    .replace(/\bFTS5\b/g, "F T S five")
    .replace(/\bRAG\b/g, "rag")
    .replace(/\bUI\b/g, "U I")
    .replace(/[*_#~>]/g, "")
    .replace(/\[([^\]]+)\]\([^\)]+\)/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

function updateWakePillUI(active, customLabel = "") {
  const pill = document.getElementById("wake-word-pill");
  const label = document.getElementById("wake-pill-label");
  if (!pill) return;
  if (active) {
    pill.classList.add("active");
    if (label) label.textContent = customLabel || (isMerlinConversationActive ? "Active Dialogue (Speak freely)" : "Wake Word: Merlin (Always Listening)");
  } else {
    pill.classList.remove("active");
    if (label) label.textContent = customLabel || "Wake Word: Merlin (Paused)";
  }
}

function setMerlinHudState(state = "listening", customText = "") {
  const eye = document.getElementById("sentinel-eye");
  const statusEl = document.getElementById("sentinel-status-text");
  const subEl = document.getElementById("sentinel-sub-text");
  const coreAiText = document.getElementById("core-ai-text");
  const coreAiSub = document.getElementById("core-ai-subtext");

  if (eye) {
    eye.classList.remove("sentinel-listening", "sentinel-processing", "sentinel-speaking");
    if (state === "listening" || state === "active_dialogue") eye.classList.add("sentinel-listening");
    else if (state === "processing") eye.classList.add("sentinel-processing");
    else if (state === "speaking") eye.classList.add("sentinel-speaking");
  }

  if (coreAiText && coreAiSub) {
    if (state === "listening" || state === "active_dialogue") {
      coreAiText.textContent = "AI";
      coreAiSub.textContent = "LISTENING";
    } else if (state === "processing") {
      coreAiText.textContent = "AI";
      coreAiSub.textContent = "SYNAPSE";
    } else if (state === "speaking") {
      coreAiText.textContent = "AI";
      coreAiSub.textContent = "VOICE";
    } else {
      coreAiText.textContent = "AI";
      coreAiSub.textContent = "MERLIN";
    }
  }

  if (statusEl && customText) {
    statusEl.innerHTML = customText;
  }
  if (subEl && customText) {
    subEl.textContent = customText;
  }
}

function playMerlinChime(type = "wake") {
  // Silent by default: only play audio chime if explicitly enabled in Settings
  if (localStorage.getItem("merlin_sound_chimes") !== "true") return;
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    if (!merlinAudioCtx) merlinAudioCtx = new AudioCtx();
    if (merlinAudioCtx.state === "suspended") merlinAudioCtx.resume();

    const now = merlinAudioCtx.currentTime;
    const osc1 = merlinAudioCtx.createOscillator();
    const osc2 = merlinAudioCtx.createOscillator();
    const gain = merlinAudioCtx.createGain();

    if (type === "wake") {
      // Harmonic Stark HUD pulse: D5 (587.33Hz) -> A5 (880.00Hz)
      osc1.type = "sine";
      osc1.frequency.setValueAtTime(587.33, now);
      osc1.frequency.exponentialRampToValueAtTime(880.00, now + 0.12);

      osc2.type = "triangle";
      osc2.frequency.setValueAtTime(880.00, now);
      osc2.frequency.exponentialRampToValueAtTime(1174.66, now + 0.12);

      gain.gain.setValueAtTime(0.001, now);
      gain.gain.linearRampToValueAtTime(0.18, now + 0.04);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.28);

      osc1.connect(gain);
      osc2.connect(gain);
      gain.connect(merlinAudioCtx.destination);

      osc1.start(now);
      osc2.start(now);
      osc1.stop(now + 0.30);
      osc2.stop(now + 0.30);
    } else if (type === "ack") {
      // Dual-ping acknowledgment: G5 (783.99Hz) -> C6 (1046.50Hz)
      osc1.type = "sine";
      osc1.frequency.setValueAtTime(783.99, now);
      osc1.frequency.setValueAtTime(1046.50, now + 0.08);

      gain.gain.setValueAtTime(0.001, now);
      gain.gain.linearRampToValueAtTime(0.14, now + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.20);

      osc1.connect(gain);
      gain.connect(merlinAudioCtx.destination);

      osc1.start(now);
      osc1.stop(now + 0.22);
    } else {
      // Confirmation pulse: A5 (880Hz) -> D6 (1174.66Hz)
      osc1.type = "sine";
      osc1.frequency.setValueAtTime(880.00, now);
      osc1.frequency.exponentialRampToValueAtTime(1174.66, now + 0.10);

      gain.gain.setValueAtTime(0.001, now);
      gain.gain.linearRampToValueAtTime(0.12, now + 0.03);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.22);

      osc1.connect(gain);
      gain.connect(merlinAudioCtx.destination);

      osc1.start(now);
      osc1.stop(now + 0.24);
    }
  } catch (e) {
    console.debug("AudioContext chime error:", e);
  }
}
const playJarvisChime = playMerlinChime;

function toggleSoundChimes(enabled) {
  localStorage.setItem("merlin_sound_chimes", enabled ? "true" : "false");
  showToast(enabled ? "Wake word audio chime enabled." : "Wake word audio chime silenced.", "info");
}

function speakMerlinText(text, onComplete) {
  if (!('speechSynthesis' in window)) {
    if (onComplete) onComplete();
    return;
  }

  // 1. Instantly mute & abort speech recognition to prevent speaker feedback loop
  isMerlinSpeaking = true;
  if (recognition) {
    try { recognition.abort(); } catch(e) {}
  }
  window.speechSynthesis.cancel();

  const cleanText = prepareTextForSpeech(text);
  if (!cleanText) {
    isMerlinSpeaking = false;
    if (onComplete) onComplete();
    return;
  }

  lastSpokenText = cleanText.toLowerCase();
  lastSpokenTime = Date.now();
  setMerlinHudState("speaking");

  if (!bestVoice) updateBestVoice();

  // Split into natural sentences for silky-smooth cadence and buffer-free audio delivery
  const sentences = cleanText.match(/[^.!?]+[.!?]+|\S+/g) || [cleanText];
  let sentenceIdx = 0;

  function speakNext() {
    if (sentenceIdx >= sentences.length) {
      isMerlinSpeaking = false;
      lastSpokenTime = Date.now();
      // Keep conversation session active for 35 seconds of follow-up listening!
      refreshConversationSession(35000);
      // Acoustic grace period: wait 700ms for room reverb/speaker echo to clear before re-enabling mic
      setTimeout(() => {
        if (isMerlinVoiceActive && !isMerlinSpeaking && (!window.speechSynthesis || !window.speechSynthesis.speaking)) {
          startMerlinListening();
        }
      }, 700);
      if (onComplete) onComplete();
      return;
    }

    const currentSentence = sentences[sentenceIdx++].trim();
    if (!currentSentence) {
      speakNext();
      return;
    }

    const utterance = new SpeechSynthesisUtterance(currentSentence);
    if (bestVoice) utterance.voice = bestVoice;
    const voiceCfg = getVoiceStyleConfig();
    utterance.rate = voiceCfg.rate;
    utterance.pitch = voiceCfg.pitch;

    utterance.onend = () => {
      setTimeout(speakNext, 40);
    };

    utterance.onerror = () => {
      isMerlinSpeaking = false;
      setMerlinHudState("listening");
      if (onComplete) onComplete();
    };

    window.speechSynthesis.speak(utterance);
  }

  speakNext();
}
const speakJarvisText = speakMerlinText;

function toggleMerlinVoice(forceState) {
  const micBtn = document.getElementById("btn-mic");
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    showToast("Speech Recognition is not supported in this browser.", "warning");
    return;
  }

  const newState = (typeof forceState === "boolean") ? forceState : !isMerlinVoiceActive;

  if (!newState) {
    isMerlinVoiceActive = false;
    isMerlinConversationActive = false;
    isMerlinSpeaking = false;
    resetSpeechStream();
    dismissJarvisHudDisplay();
    if (document.body) document.body.classList.remove("voice-mode-active");
    const list = document.getElementById("messages-list");
    if (list && list.children.length > 0) list.style.display = "flex";
    localStorage.setItem("merlin_handsfree_wake", "false");
    if (conversationActiveTimer) {
      clearTimeout(conversationActiveTimer);
      conversationActiveTimer = null;
    }
    if (recognition) {
      try { recognition.abort(); } catch(e) {}
    }
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    if (micBtn) {
      micBtn.classList.remove("merlin-voice-active", "jarvis-active");
      micBtn.style.color = "var(--text-muted)";
    }
    updateWakePillUI(false);
    playMerlinChime("finish");
    showToast("Merlin Microphone Paused (Text mode active).", "info");
    return;
  }

  isMerlinVoiceActive = true;
  isMerlinSpeaking = false;
  dismissJarvisHudDisplay();
  if (document.body) document.body.classList.add("voice-mode-active");
  const list = document.getElementById("messages-list");
  if (list) list.style.display = "none";
  localStorage.setItem("merlin_handsfree_wake", "true");
  if (micBtn) {
    micBtn.classList.add("merlin-voice-active", "jarvis-active");
  }
  updateWakePillUI(true);
  setMerlinHudState("listening");
  playMerlinChime("wake");
  showToast("Merlin Hands-Free Voice Online. Always listening.", "success");
  startMerlinListening();
}
const toggleJarvisVoice = toggleMerlinVoice;

function startMerlinListening() {
  if (!isMerlinVoiceActive) return;
  // If Merlin is synthesizing speech, do not engage microphone!
  if (isMerlinSpeaking || (window.speechSynthesis && window.speechSynthesis.speaking)) {
    return;
  }
  if (isRecognitionRunning) return;

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) return;

  if (!recognition) {
    recognition = new SpeechRecognition();
  }
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = "en-US";

  recognition.onstart = () => {
    isRecognitionRunning = true;
    const micBtn = document.getElementById("btn-mic");
    if (micBtn) micBtn.classList.add("merlin-voice-active", "jarvis-active");
    updateWakePillUI(true);
    if (!isMerlinConversationActive) {
      setMerlinHudState("listening");
    }
  };

  recognition.onspeechstart = () => {
    // Natural Barge-In: if user speaks while assistant is speaking, halt speech immediately
    if (window.speechSynthesis && window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel();
      isMerlinSpeaking = false;
      refreshConversationSession();
    }
  };

  recognition.onresult = async (event) => {
    // 1. Acoustic Feedback Protection: Drop immediately if Merlin is speaking
    if (isMerlinSpeaking || (window.speechSynthesis && window.speechSynthesis.speaking)) {
      console.debug("Dropped transcript during active speech synthesis.");
      return;
    }

    let transcript = (event.results[0][0].transcript || "").trim();
    if (!transcript) return;
    const lower = transcript.toLowerCase();

    // 2. Room Echo Guard: Drop if transcript matches text Merlin just spoke within the last 2.2 seconds
    if (Date.now() - lastSpokenTime < 2200 && lastSpokenText) {
      if (lastSpokenText.includes(lower) || lower.includes("assist you") || lower.includes("how can i help") || lower.includes("how may i assist")) {
        console.debug("Acoustic echo filtered out from Merlin speakers:", transcript);
        return;
      }
    }

    // 3. Direct Voice Directive: Stop / Cancel / Standby
    if (/^(merlin|hey merlin)?[,\s]*(stop|shut up|be quiet|silence|hold on|pause|standby|sleep|go to sleep|mute|that's all|thank you|thanks merlin)$/i.test(lower) || lower === "stop") {
      stopGeneration();
      if (window.speechSynthesis) window.speechSynthesis.cancel();
      isMerlinSpeaking = false;
      dismissJarvisHudDisplay();
      endConversationSession("Standing by, Sir.");
      return;
    }

    // 4. Direct Voice Directive: Clear Screen / Dismiss On-Demand Display
    if (/^(merlin|hey merlin)?[,\s]*(clear screen|clear chat|new chat|dismiss|hide display|close display|clear display|hide window)$/i.test(lower)) {
      dismissJarvisHudDisplay();
      createNewChat();
      playMerlinChime("ack");
      speakMerlinText("Display cleared, Sir.");
      refreshConversationSession();
      return;
    }

    // 5. Identity Check: Redirect if called Jarvis
    if (/^(hey\s+)?jarvis\b/i.test(lower) || /\bjarvis\b/i.test(lower)) {
      playMerlinChime("ack");
      speakMerlinText("My name is Merlin, Sir. Jarvis is not here. How may I assist you?");
      refreshConversationSession();
      return;
    }

    // 6. Wake Word Matching for "Merlin"
    const wakePatterns = [
      /^(hey\s+)?merlin[\s,.:;!?-]*/i,
      /^(wake\s+up\s+)?merlin[\s,.:;!?-]*/i,
      /^merlin\b[\s,.:;!?-]*/i
    ];
    let matchedWakeWord = false;
    let strippedPrompt = transcript;
    for (const pat of wakePatterns) {
      if (pat.test(strippedPrompt)) {
        matchedWakeWord = true;
        strippedPrompt = strippedPrompt.replace(pat, "").trim();
        break;
      }
    }

    if (!matchedWakeWord && /\bmerlin\b/i.test(transcript)) {
      matchedWakeWord = true;
      strippedPrompt = transcript.replace(/\bmerlin\b[,\s.:;!?-]*/gi, "").trim();
    }

    // CRUCIAL: In hands-free standby, if NO wake word was uttered AND we are not in active dialogue: ignore ambient room noise!
    if (!matchedWakeWord && !isMerlinConversationActive) {
      console.debug("Ignored ambient speech without wake word 'Merlin':", transcript);
      return;
    }

    // If user only uttered the wake word ("Merlin" or "Hey Merlin") while in standby
    if (matchedWakeWord && !strippedPrompt) {
      refreshConversationSession();
      playMerlinChime("wake");
      speakMerlinText("At your service, Sir. How may I assist you?");
      return;
    }

    // Either wake word was uttered with a question, OR user is actively conversing in an active session!
    refreshConversationSession();

    const finalPrompt = strippedPrompt || transcript;
    setMerlinHudState("processing");
    const ta = document.getElementById("chat-textarea");
    ta.value = finalPrompt;
    autoResizeTextarea(ta);
    await submitMessage(true);
  };

  recognition.onerror = (e) => {
    isRecognitionRunning = false;
    if (e && e.error === "not-allowed") {
      console.warn("Microphone permission pending user gesture.");
      return;
    }
    if (isMerlinVoiceActive && !isMerlinSpeaking) {
      setTimeout(() => {
        if (isMerlinVoiceActive && !isMerlinSpeaking && !isRecognitionRunning) {
          startMerlinListening();
        }
      }, 700);
    }
  };

  recognition.onend = () => {
    isRecognitionRunning = false;
    // Always listening loop: auto-restart listener continuously
    if (isMerlinVoiceActive && !isMerlinSpeaking && (!window.speechSynthesis || !window.speechSynthesis.speaking)) {
      setTimeout(() => {
        if (isMerlinVoiceActive && !isMerlinSpeaking && !isRecognitionRunning) {
          try {
            recognition.start();
            isRecognitionRunning = true;
          } catch(err) {
            startMerlinListening();
          }
        }
      }, 150);
    }
  };

  try {
    recognition.start();
    isRecognitionRunning = true;
  } catch(e) {
    isRecognitionRunning = false;
  }
}
const startJarvisListening = startMerlinListening;

function toggleDictation() {
  toggleMerlinVoice();
}

// Always listening initialization: start automatically without requiring mic button press
function initAlwaysListeningMerlin() {
  if (('webkitSpeechRecognition' in window) || ('SpeechRecognition' in window)) {
    isMerlinVoiceActive = true;
    if (document.body) document.body.classList.add("voice-mode-active");
    const list = document.getElementById("messages-list");
    if (list) list.style.display = "none";
    updateWakePillUI(true);
    startMerlinListening();
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initAlwaysListeningMerlin);
} else {
  initAlwaysListeningMerlin();
}

// User-gesture fallback if browser security blocks microphone until first page interaction
const ensureListeningOnGesture = () => {
  if (isMerlinVoiceActive && !isRecognitionRunning && !isMerlinSpeaking) {
    startMerlinListening();
  }
};
window.addEventListener("click", ensureListeningOnGesture, { passive: true });
window.addEventListener("keydown", ensureListeningOnGesture, { passive: true });
window.addEventListener("touchstart", ensureListeningOnGesture, { passive: true });

// True Jarvis Auto-Hiding Input Dock in Voice Mode:
// Dock stays completely hidden to leave only the central cyber core animation on screen;
// Reveals smoothly on mousemove near bottom of viewport or keyboard input.
let dockHideTimeout = null;
function revealDockTemporarily() {
  if (!document.body.classList.contains("voice-mode-active")) return;
  document.body.classList.add("show-dock");
  clearTimeout(dockHideTimeout);
  dockHideTimeout = setTimeout(() => {
    const ta = document.getElementById("chat-textarea");
    if (document.activeElement !== ta) {
      document.body.classList.remove("show-dock");
    }
  }, 4000);
}

window.addEventListener("mousemove", (e) => {
  if (e.clientY >= window.innerHeight - 75) {
    revealDockTemporarily();
  }
}, { passive: true });

window.addEventListener("keydown", (e) => {
  if (!['Control', 'Shift', 'Alt', 'Meta'].includes(e.key)) {
    revealDockTemporarily();
  }
}, { passive: true });

// ==========================================================================
// PERSONAL ASSISTANT: TO-DOS & TECHNICAL NOTES
// ==========================================================================
function switchPaTab(tab) {
  document.getElementById("pa-tab-todos").style.display = tab === "todos" ? "block" : "none";
  document.getElementById("pa-tab-notes").style.display = tab === "notes" ? "block" : "none";
  document.getElementById("tab-btn-todos").classList.toggle("active", tab === "todos");
  document.getElementById("tab-btn-notes").classList.toggle("active", tab === "notes");
}

async function loadTodos() {
  try {
    const res = await fetch("/api/todos");
    const todos = await res.json();
    const container = document.getElementById("todos-list-container");
    const badge = document.getElementById("badge-todos");
    const pendingCount = todos.filter(t => !t.completed).length;

    if (badge) {
      badge.innerText = pendingCount;
      badge.style.display = pendingCount > 0 ? "inline-block" : "none";
    }
    if (!container) return;

    if (!todos.length) {
      container.innerHTML = '<div class="empty-list-msg">No tasks recorded yet. Add one above!</div>';
      return;
    }

    container.innerHTML = todos.map(t => `
      <div class="list-card ${t.completed ? 'completed-task' : ''}">
        <div class="list-card-main" onclick="toggleTodoItem(${t.id})" style="cursor:pointer;">
          <input type="checkbox" ${t.completed ? 'checked' : ''} onclick="event.stopPropagation(); toggleTodoItem(${t.id})">
          <span class="task-text">${escapeHtml(t.task)}</span>
          <span class="priority-pill priority-${escapeHtml(t.priority || 'normal')}">${escapeHtml(t.priority || 'normal')}</span>
        </div>
        <button class="conv-action-btn" onclick="deleteTodoItem(${t.id})" title="Delete task">✕</button>
      </div>
    `).join("");
  } catch (e) {
    console.error("Failed loading todos:", e);
  }
}

async function createTodoItem() {
  const input = document.getElementById("new-todo-input");
  const priority = document.getElementById("new-todo-priority").value;
  const task = input.value.trim();
  if (!task) return;
  await fetch("/api/todos", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task, priority })
  });
  input.value = "";
  loadTodos();
  showToast("Task added to To-Do list.", "success");
}

async function toggleTodoItem(id) {
  await fetch("/api/todos/toggle", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id })
  });
  loadTodos();
}

async function deleteTodoItem(id) {
  await fetch("/api/todos/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id })
  });
  loadTodos();
}

let notesSearchTimer = null;

function searchNotesDebounced() {
  clearTimeout(notesSearchTimer);
  notesSearchTimer = setTimeout(() => {
    const q = document.getElementById("notes-search-input")?.value?.trim() || "";
    loadNotes(q);
  }, 220);
}

async function loadNotes(query = "") {
  try {
    const url = query
      ? `/api/notes/search?q=${encodeURIComponent(query)}&limit=50`
      : "/api/notes";
    const res = await fetch(url);
    const notes = await res.json();
    const container = document.getElementById("notes-list-container");
    if (!container) return;

    notesById.clear();
    if (!notes.length) {
      container.innerHTML = query
        ? `<div class="empty-list-msg">No notes matching "${escapeHtml(query)}".</div>`
        : '<div class="empty-list-msg">No technical notes saved yet.</div>';
      return;
    }

    notes.forEach(n => notesById.set(Number(n.id), n));

    container.innerHTML = notes.map(n => `
      <div class="list-card list-card-column">
        <div style="display:flex;justify-content:space-between;align-items:center;width:100%;">
          <b style="font-size:0.9rem;color:var(--accent-primary);">${escapeHtml(n.title)}</b>
          <div style="display:flex;gap:6px;">
            <button class="conv-action-btn" onclick="editNoteById(${Number(n.id)})">✎ Edit</button>
            <button class="conv-action-btn" onclick="deleteNoteItem(${Number(n.id)})">✕</button>
          </div>
        </div>
        <div style="font-size:0.84rem;color:var(--text-main);white-space:pre-wrap;margin-top:4px;">${escapeHtml(n.content)}</div>
        ${n.tags ? `<div style="font-size:0.72rem;color:var(--text-muted);margin-top:4px;">🏷️ ${escapeHtml(n.tags)}</div>` : ''}
      </div>
    `).join("");
  } catch (e) {
    console.error("Failed loading notes:", e);
  }
}

function editNoteById(id) {
  const note = notesById.get(Number(id));
  if (!note) return;
  editNoteItem(note.id, note.title, note.content, note.tags || "");
}

function editNoteItem(id, title, content, tags) {
  document.getElementById("note-edit-id").value = id;
  document.getElementById("note-title-input").value = title;
  document.getElementById("note-content-input").value = content;
  document.getElementById("note-tags-input").value = tags || "";
  document.getElementById("note-save-btn").innerText = "Update Note";
  document.getElementById("note-cancel-edit-btn").style.display = "inline-block";
}

function resetNoteForm() {
  document.getElementById("note-edit-id").value = "";
  document.getElementById("note-title-input").value = "";
  document.getElementById("note-content-input").value = "";
  document.getElementById("note-tags-input").value = "";
  document.getElementById("note-save-btn").innerText = "Save Note";
  document.getElementById("note-cancel-edit-btn").style.display = "none";
}

async function saveNoteItem() {
  const editId = document.getElementById("note-edit-id").value;
  const title = document.getElementById("note-title-input").value.trim() || "Untitled Note";
  const content = document.getElementById("note-content-input").value.trim();
  const tags = document.getElementById("note-tags-input").value.trim();
  if (!content) {
    showToast("Note content cannot be empty.", "warning");
    return;
  }

  if (editId) {
    await fetch("/api/notes/update", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: parseInt(editId, 10), title, content, tags })
    });
    showToast("Note updated.", "success");
  } else {
    await fetch("/api/notes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, content, tags })
    });
    showToast("Note saved.", "success");
  }
  resetNoteForm();
  loadNotes();
}

async function deleteNoteItem(id) {
  if (!confirm("Are you sure you want to delete this technical note?")) return;
  await fetch("/api/notes/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id })
  });
  loadNotes();
}

// ==========================================================================
// MEMORY BANK: KNOWLEDGE GAPS & LEARNED SKILLS
// ==========================================================================
function switchMemoryTab(tab) {
  document.getElementById("memory-tab-gaps").style.display = tab === "gaps" ? "block" : "none";
  document.getElementById("memory-tab-skills").style.display = tab === "skills" ? "block" : "none";
  document.getElementById("tab-btn-gaps").classList.toggle("active", tab === "gaps");
  document.getElementById("tab-btn-skills").classList.toggle("active", tab === "skills");
}

async function loadGaps() {
  try {
    const res = await fetch("/api/gaps");
    const gaps = await res.json();
    const container = document.getElementById("gaps-list-container");
    const badge = document.getElementById("badge-gaps");

    if (badge) {
      badge.innerText = gaps.length;
      badge.style.display = gaps.length > 0 ? "inline-block" : "none";
    }
    if (!container) return;

    if (!gaps.length) {
      container.innerHTML = '<div class="empty-list-msg">No unresolved knowledge gaps! Merlin is all caught up.</div>';
      return;
    }

    container.innerHTML = gaps.map(g => `
      <div class="list-card list-card-column">
        <div style="display:flex;justify-content:space-between;align-items:center;width:100%;">
          <b style="font-size:0.88rem;">#${g.id}: ${escapeHtml(g.question)}</b>
          <div style="display:flex;align-items:center;gap:6px;">
            <span class="priority-pill priority-${g.status === 'failed' ? 'high' : 'normal'}">${escapeHtml(g.status)}</span>
            <button class="conv-action-btn" onclick="deleteGapItem(${g.id})" title="Delete gap">✕</button>
          </div>
        </div>
        ${g.context ? `<div style="font-size:0.8rem;color:var(--text-muted);margin-top:3px;">Context: ${escapeHtml(g.context)}</div>` : ''}
        ${g.last_error ? `<div style="font-size:0.75rem;color:var(--danger);margin-top:3px;">Last error: ${escapeHtml(g.last_error)}</div>` : ''}
      </div>
    `).join("");
  } catch (e) {
    console.error("Failed loading gaps:", e);
  }
}

async function createGapItem() {
  const qInput = document.getElementById("new-gap-question");
  const cInput = document.getElementById("new-gap-context");
  const question = qInput.value.trim();
  const context = cInput.value.trim();
  if (!question) {
    showToast("Please enter a question or concept to record.", "warning");
    return;
  }
  await fetch("/api/gaps", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, context })
  });
  qInput.value = "";
  cInput.value = "";
  loadGaps();
  showToast("Knowledge gap recorded for Teacher distillation.", "success");
}

async function deleteGapItem(id) {
  await fetch("/api/gaps/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id })
  });
  loadGaps();
}

function toggleAddSkillForm() {
  const form = document.getElementById("add-skill-form");
  if (form) {
    form.style.display = form.style.display === "none" ? "block" : "none";
  }
}

function searchSkillsDebounced() {
  clearTimeout(skillsSearchTimer);
  skillsSearchTimer = setTimeout(() => {
    const q = document.getElementById("skills-search-input")?.value?.trim() || "";
    loadSkills(q);
  }, 220);
}

async function loadSkills(query = "") {
  try {
    const url = query
      ? `/api/knowledge/search?q=${encodeURIComponent(query)}&limit=50`
      : "/api/knowledge";
    const res = await fetch(url);
    const skills = await res.json();
    const container = document.getElementById("skills-list-container");
    const badge = document.getElementById("badge-knowledge");

    if (!query && badge) {
      badge.innerText = skills.length;
      badge.style.display = skills.length > 0 ? "inline-block" : "none";
    }
    if (!container) return;

    if (!skills.length) {
      container.innerHTML = '<div class="empty-list-msg">No matching skills in SQLite memory yet. Use Teacher Distillation or Hugging Face Datasets to populate!</div>';
      return;
    }

    container.innerHTML = skills.map(s => `
      <div class="list-card list-card-column">
        <div style="display:flex;justify-content:space-between;align-items:center;width:100%;">
          <b style="font-size:0.9rem;color:var(--accent-primary);">${escapeHtml(s.topic)}</b>
          <div style="display:flex;align-items:center;gap:6px;">
            ${s.source ? `<span style="font-size:0.7rem;color:var(--text-muted);">${escapeHtml(s.source)}</span>` : ''}
            <button class="conv-action-btn" onclick="deleteSkillItem(${s.id})" title="Delete skill">✕</button>
          </div>
        </div>
        <div style="font-size:0.83rem;color:var(--text-main);margin-top:4px;white-space:pre-wrap;">${escapeHtml((s.lesson || "").slice(0, 360))}${(s.lesson || "").length > 360 ? "..." : ""}</div>
        ${s.code_example ? `<pre style="margin-top:6px;padding:8px 10px;border-radius:8px;background:var(--code-bg);font-size:0.76rem;max-height:140px;overflow:auto;"><code>${escapeHtml(s.code_example)}</code></pre>` : ''}
        ${s.tags ? `<div style="font-size:0.72rem;color:var(--text-muted);margin-top:4px;">🏷️ ${escapeHtml(s.tags)}</div>` : ''}
      </div>
    `).join("");
  } catch (e) {
    console.error("Failed loading skills:", e);
  }
}

async function createSkillItem() {
  const topic = document.getElementById("skill-topic-input").value.trim();
  const lesson = document.getElementById("skill-lesson-input").value.trim();
  const code_example = document.getElementById("skill-code-input").value.trim();
  const tags = document.getElementById("skill-tags-input").value.trim();
  if (!topic || !lesson) {
    showToast("Topic and lesson are required.", "warning");
    return;
  }
  await fetch("/api/knowledge", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic, lesson, code_example, tags, source: "manual" })
  });
  document.getElementById("skill-topic-input").value = "";
  document.getElementById("skill-lesson-input").value = "";
  document.getElementById("skill-code-input").value = "";
  document.getElementById("skill-tags-input").value = "";
  toggleAddSkillForm();
  loadSkills();
  showToast("Skill indexed into SQLite FTS5 memory.", "success");
}

async function deleteSkillItem(id) {
  if (!confirm("Are you sure you want to delete this learned skill from SQLite memory?")) return;
  await fetch("/api/knowledge/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id })
  });
  loadSkills();
}

// ==========================================================================
// HARDWARE TELEMETRY, MODEL MANAGEMENT & AUDIT LOGS
// ==========================================================================
async function loadTelemetry() {
  try {
    const res = await fetch("/api/system");
    const d = await res.json();
    document.getElementById("diag-temp").innerText = d.temperature || "-";
    document.getElementById("diag-cpu").innerText = d.cpu_usage || "-";
    document.getElementById("diag-mem").innerText = d.memory || "-";
    document.getElementById("diag-disk").innerText = d.disk || "-";
    const osEl = document.getElementById("diag-os");
    if (osEl) osEl.innerText = `${d.os || ""} (${d.arch || ""})`.trim();
    const upEl = document.getElementById("diag-uptime");
    if (upEl) upEl.innerText = d.uptime || "-";
  } catch (e) {}
}

async function loadModels() {
  try {
    const [statusRes, modelsRes] = await Promise.all([
      fetch("/api/status"),
      fetch("/api/models")
    ]);
    const d = await statusRes.json();
    const models = await modelsRes.json();

    const display = document.getElementById("active-model-display");
    const tag = document.getElementById("active-backend-tag");
    const navLabel = document.getElementById("navbar-engine-label");

    if (display) {
      if (d.model && d.model !== "None" && !d.model.includes("Waiting")) {
        display.innerText = `${d.model} (${d.backend})`;
        if (tag) tag.innerText = "ONLINE";
        if (navLabel) navLabel.innerText = d.model;
      } else {
        display.innerText = "Offline Standby (Auto-Detect Active)";
        if (tag) tag.innerText = "STANDBY";
        if (navLabel) navLabel.innerText = "Auto-Engine Active";
      }
    }

    // Update sidebar badges
    const bKnowledge = document.getElementById("badge-knowledge");
    if (bKnowledge && typeof d.knowledge_count === "number") {
      bKnowledge.innerText = d.knowledge_count;
      bKnowledge.style.display = d.knowledge_count > 0 ? "inline-block" : "none";
    }
    const bGaps = document.getElementById("badge-gaps");
    if (bGaps && typeof d.pending_gaps === "number") {
      bGaps.innerText = d.pending_gaps;
      bGaps.style.display = d.pending_gaps > 0 ? "inline-block" : "none";
    }
    const bCalendar = document.getElementById("badge-calendar");
    if (bCalendar && typeof d.calendar_count === "number") {
      bCalendar.innerText = d.calendar_count;
      bCalendar.style.display = d.calendar_count > 0 ? "inline-block" : "none";
    }

    const select = document.getElementById("local-model-select");
    if (select) {
      if (!models || !models.length) {
        select.innerHTML = '<option value="">No local .gguf weights in models/ yet (run ./download_model.sh)</option>';
      } else {
        select.innerHTML = models.map(m =>
          `<option value="${escapeHtml(m.filename)}" ${m.is_active ? 'selected' : ''}>${escapeHtml(m.name)} (${m.size_mb} MB)</option>`
        ).join("");
      }
    }
  } catch (e) {
    const display = document.getElementById("active-model-display");
    if (display) display.innerText = "Offline Engine (Active)";
  }
}

async function switchLocalModel() {
  const select = document.getElementById("local-model-select");
  const model = select ? select.value : "";
  if (!model) {
    showToast("No local .gguf model file selected. Run ./download_model.sh to download weights.", "warning");
    return;
  }
  const res = await fetch("/api/models/active", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model })
  });
  const d = await res.json();
  if (d.success) {
    showToast(d.message || `Activated ${model}`, "success");
    loadModels();
  } else {
    showToast(d.error || d.detail || "Could not switch model", "danger");
  }
}

async function loadAuditLogs() {
  try {
    const res = await fetch("/api/audit");
    const logs = await res.json();
    const container = document.getElementById("audit-log-container");
    if (!container) return;
    if (!logs.length) {
      container.innerHTML = '<div class="empty-list-msg">No audit events recorded yet.</div>';
      return;
    }
    container.innerHTML = logs.map(l => `
      <div style="font-size:0.75rem;padding:5px 8px;border-bottom:1px solid var(--border-subtle);display:flex;justify-content:space-between;gap:8px;">
        <span><b>[${escapeHtml(l.event_type)}]</b> ${escapeHtml(l.details)}</span>
        <span style="color:var(--text-muted);white-space:nowrap;">${escapeHtml(l.created_at || '')}</span>
      </div>
    `).join("");
  } catch (e) {}
}

// ==========================================================================
// TEACHER DISTILLATION & LIVE PROGRESS POLLING
// ==========================================================================
function onTeacherProviderChange() {
  const provider = document.getElementById("teacher-provider").value;
  const select = document.getElementById("teacher-model-select");
  const defaults = {
    gemini: ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
    groq: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "deepseek-r1-distill-llama-70b"],
    openrouter: ["google/gemini-2.0-flash-exp:free", "deepseek/deepseek-r1:free", "qwen/qwen-2.5-coder-32b-instruct:free"],
    openai: ["gpt-4o-mini", "gpt-4o"]
  };
  const list = defaults[provider] || defaults.gemini;
  select.innerHTML = list.map(m => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`).join("");
}

async function fetchTeacherModels() {
  const provider = document.getElementById("teacher-provider").value;
  const apiKey = document.getElementById("teacher-api-key").value.trim();
  try {
    const res = await fetch(`/api/teacher/models?provider=${encodeURIComponent(provider)}&api_key=${encodeURIComponent(apiKey)}`);
    const d = await res.json();
    if (d.models && d.models.length) {
      const select = document.getElementById("teacher-model-select");
      select.innerHTML = d.models.map(m => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`).join("");
      showToast(`Loaded ${d.models.length} live models for ${provider}.`, "success");
    }
  } catch (e) {
    showToast("Could not fetch live provider models.", "warning");
  }
}

async function startTeacherSync() {
  const apiKey = document.getElementById("teacher-api-key").value.trim();
  const topic = document.getElementById("teacher-topic").value.trim();
  const model = document.getElementById("teacher-model-select")?.value || "";
  if (!apiKey) {
    showToast("Please enter a Commercial Provider API Key.", "warning");
    return;
  }

  const res = await fetch("/api/sync", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: apiKey, topic, model })
  });
  const d = await res.json();
  if (d.success) {
    showToast("Teacher distillation started!", "success");
    checkSyncStatus();
  } else {
    showToast(d.error || d.detail || "Failed to start distillation.", "danger");
  }
}

async function checkSyncStatus() {
  try {
    const res = await fetch("/api/sync");
    const st = await res.json();
    const box = document.getElementById("sync-progress-box");
    const text = document.getElementById("sync-status-text");
    const fill = document.getElementById("sync-progress-fill");
    const btn = document.getElementById("btn-start-sync");

    if (st.is_syncing) {
      if (box) box.style.display = "block";
      if (btn) btn.disabled = true;
      const pct = st.total > 0 ? Math.round((st.progress / st.total) * 100) : 50;
      if (fill) fill.style.width = `${pct}%`;
      if (text) text.innerText = st.current_task || `Distilling knowledge (${st.progress}/${st.total})...`;
      clearTimeout(syncPollTimer);
      syncPollTimer = setTimeout(checkSyncStatus, 1200);
    } else {
      if (btn) btn.disabled = false;
      if (st.last_result === "completed") {
        if (box) box.style.display = "block";
        if (fill) fill.style.width = "100%";
        if (text) text.innerText = "✓ Distillation completed and indexed into local SQLite memory!";
        loadModels();
      } else if (st.last_result === "error") {
        if (box) box.style.display = "block";
        if (fill) fill.style.width = "100%";
        if (text) text.innerText = `⚠️ Error: ${st.last_error || "Sync failed"}`;
      }
    }
  } catch (e) {}
}

// ==========================================================================
// HUGGING FACE DATASETS & LIVE PROGRESS POLLING
// ==========================================================================
async function loadPopularDatasets() {
  if (popularDatasetsLoaded) return;
  try {
    const res = await fetch("/api/dataset/popular");
    const datasets = await res.json();
    const select = document.getElementById("hf-dataset-select");
    if (!select || !Array.isArray(datasets) || !datasets.length) return;

    select.innerHTML =
      '<option value="__all__" selected>🚀 ALL Curated Datasets (Install Every Dataset Below)</option>' +
      datasets.map(ds =>
        `<option value="${escapeHtml(ds.id)}">${escapeHtml(ds.category)} — ${escapeHtml(ds.name)}</option>`
      ).join("") +
      '<option value="__custom__">Custom Hugging Face Repository...</option>';
    popularDatasetsLoaded = true;
  } catch (e) {}
}

function onDatasetSelectChange() {
  const val = document.getElementById("hf-dataset-select").value;
  const customInput = document.getElementById("hf-custom-repo");
  if (customInput) {
    customInput.style.display = val === "__custom__" ? "block" : "none";
  }
}

async function startDatasetImport() {
  const selectVal = document.getElementById("hf-dataset-select").value;
  const customVal = document.getElementById("hf-custom-repo")?.value?.trim() || "";
  const dataset = selectVal === "__custom__" ? customVal : selectVal;
  const rawLimit = parseInt(document.getElementById("hf-limit").value, 10);
  const limit = Number.isNaN(rawLimit) ? 0 : rawLimit;
  const token = document.getElementById("hf-token")?.value?.trim() || "";
  const save_local = Boolean(document.getElementById("hf-save-local")?.checked);

  if (!dataset) {
    showToast("Please specify a Hugging Face dataset repository ID.", "warning");
    return;
  }

  const res = await fetch("/api/dataset/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset, limit, token, save_local })
  });
  const d = await res.json();
  if (d.success) {
    const label = dataset === "__all__" ? "ALL Curated Datasets" : dataset;
    showToast(`Started streaming import for ${label}`, "success");
    checkDatasetStatus();
  } else {
    showToast(d.error || d.detail || "Could not start dataset import.", "danger");
  }
}

async function checkDatasetStatus() {
  try {
    const res = await fetch("/api/dataset/status");
    const st = await res.json();
    const box = document.getElementById("dataset-progress-box");
    const text = document.getElementById("dataset-status-text");
    const fill = document.getElementById("dataset-progress-fill");
    const btn = document.getElementById("btn-start-dataset");

    if (st.is_importing) {
      if (box) box.style.display = "block";
      if (btn) btn.disabled = true;
      const pct = st.total > 0 ? Math.min(100, Math.round((st.progress / st.total) * 100)) : 100;
      if (fill) fill.style.width = `${pct}%`;
      const totalLabel = st.total > 0 ? `${st.progress} / ${st.total}` : `${st.progress} (unlimited)`;
      if (text) text.innerText = `Assimilating ${st.current_dataset}: ${totalLabel} samples...`;
      clearTimeout(datasetPollTimer);
      datasetPollTimer = setTimeout(checkDatasetStatus, 1000);
    } else {
      if (btn) btn.disabled = false;
      if (st.last_result === "completed") {
        if (box) box.style.display = "block";
        if (fill) fill.style.width = "100%";
        if (text) text.innerText = `✓ Assimilated ${st.progress} samples from ${st.current_dataset}!`;
        loadModels();
      } else if (String(st.last_result).startsWith("failed")) {
        if (box) box.style.display = "block";
        if (fill) fill.style.width = "100%";
        if (text) text.innerText = `⚠️ ${st.last_result}`;
      }
    }
  } catch (e) {}
}

// ==========================================================================
// GEMINI 4 ARGON — FRONTIER REASONING COPILOT ENGINE
// ==========================================================================
let currentArgonResult = null;
let argonTimerInterval = null;

function initArgonModal() {
  const keyInput = document.getElementById("argon-api-key");
  if (keyInput) {
    const savedKey = localStorage.getItem("gemini_api_key") || localStorage.getItem("teacher_api_key") || "";
    if (savedKey && !keyInput.value) {
      keyInput.value = savedKey;
    }
  }
}

function saveArgonApiKey() {
  const keyInput = document.getElementById("argon-api-key");
  if (keyInput) {
    const val = keyInput.value.trim();
    if (val) {
      localStorage.setItem("gemini_api_key", val);
      const teacherKey = document.getElementById("teacher-api-key");
      if (teacherKey && !teacherKey.value) teacherKey.value = val;
    }
  }
}

function applyArgonPreset(type) {
  const promptInput = document.getElementById("argon-prompt-input");
  const sysInput = document.getElementById("argon-system-instruction");
  const budgetSelect = document.getElementById("argon-thinking-budget");
  if (!promptInput) return;

  if (type === "architect") {
    promptInput.value = "Design a high-throughput, lock-free, zero-copy message ring buffer in C / Python with memory barriers, cache-line alignment, and high-throughput thread pooling. Detail the architectural decisions, trade-offs, and complete runnable code.";
    if (sysInput) sysInput.value = "You are a Principal Systems Architect. Analyze memory models, CPU cache hierarchies, and concurrency edge-cases with deep step-by-step thinking.";
    if (budgetSelect) budgetSelect.value = "4096";
  } else if (type === "debugger") {
    promptInput.value = "Debug the following memory leak and race condition: A multi-threaded async Python server intermittently crashes under load with double-free / GIL deadlock. Analyze the failure modes, stack dynamics, and provide the complete verified fix.";
    if (sysInput) sysInput.value = "You are an Elite Systems Debugger. Dissect low-level memory dynamics, stack traces, and thread interleavings before proposing the verified resolution.";
    if (budgetSelect) budgetSelect.value = "2048";
  } else if (type === "distill") {
    promptInput.value = "Create an offline, production-grade implementation of a resilient WebSocket reconnection manager with exponential backoff and jitter for low-power edge nodes. Format the output with clear architectural principles and clean code.";
    if (sysInput) sysInput.value = "You are a Senior AI Mentor distilling high-yield offline technical skills for local offline models.";
    if (budgetSelect) budgetSelect.value = "2048";
  } else if (type === "proof") {
    promptInput.value = "Provide a rigorous mathematical and algorithmic analysis of KV-cache memory complexity in multi-head vs multi-query attention, calculating VRAM footprint during auto-regressive decoding across sequence lengths up to 128k.";
    if (sysInput) sysInput.value = "You are an AI Compiler and Mathematics Specialist. Formulate exact equations, step-by-step mathematical derivations, and memory benchmarks.";
    if (budgetSelect) budgetSelect.value = "4096";
  }
  showToast(`Applied preset: ${type}`, "info");
}

function clearArgonPrompt() {
  const promptInput = document.getElementById("argon-prompt-input");
  if (promptInput) promptInput.value = "";
  const output = document.getElementById("argon-output-container");
  if (output) output.style.display = "none";
  const status = document.getElementById("argon-query-status");
  if (status) status.innerText = "Ready • Press Ctrl+Enter";
}

async function queryArgonCopilot() {
  const promptInput = document.getElementById("argon-prompt-input");
  const modelSelect = document.getElementById("argon-model-select");
  const budgetSelect = document.getElementById("argon-thinking-budget");
  const keyInput = document.getElementById("argon-api-key");
  const sysInput = document.getElementById("argon-system-instruction");
  const runBtn = document.getElementById("btn-argon-query");
  const statusEl = document.getElementById("argon-query-status");
  const outputContainer = document.getElementById("argon-output-container");

  const prompt = promptInput ? promptInput.value.trim() : "";
  if (!prompt) {
    showToast("Please enter a question or problem to reason through.", "warning");
    return;
  }

  saveArgonApiKey();
  const apiKey = (keyInput ? keyInput.value.trim() : "") || localStorage.getItem("gemini_api_key") || localStorage.getItem("teacher_api_key") || "";
  const model = modelSelect ? modelSelect.value : "gemini-4-argon";
  const thinkingBudget = budgetSelect ? parseInt(budgetSelect.value, 10) : 2048;
  const systemInstruction = sysInput ? sysInput.value.trim() : "";

  if (runBtn) runBtn.disabled = true;
  let elapsed = 0;
  clearInterval(argonTimerInterval);
  if (statusEl) statusEl.innerText = `Reasoning with ${model} (0.0s)...`;
  argonTimerInterval = setInterval(() => {
    elapsed += 0.5;
    if (statusEl) statusEl.innerText = `Thinking & synthesizing with ${model} (${elapsed.toFixed(1)}s)...`;
  }, 500);

  try {
    const res = await fetch("/api/teacher/argon/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: prompt,
        api_key: apiKey,
        model: model,
        thinking_budget: thinkingBudget,
        reasoning_effort: thinkingBudget >= 4096 ? "high" : (thinkingBudget >= 2048 ? "medium" : "low"),
        system_instruction: systemInstruction
      })
    });

    const data = await res.json();
    clearInterval(argonTimerInterval);

    if (!res.ok || !data.success) {
      const errMsg = data.error || data.detail || "Query failed";
      if (statusEl) statusEl.innerText = `⚠️ Error: ${errMsg}`;
      showToast(`Argon query failed: ${errMsg}`, "danger");
      return;
    }

    currentArgonResult = data;
    if (statusEl) statusEl.innerText = `✓ Completed in ${elapsed.toFixed(1)}s via ${data.model}`;
    showToast(`Gemini 4 Argon reasoning completed via ${data.model}!`, "success");

    // Render output
    if (outputContainer) outputContainer.style.display = "block";

    // Thinking Trace
    const thinkingBody = document.getElementById("argon-thinking-body");
    const thinkingTokensBadge = document.getElementById("argon-thinking-tokens-badge");
    const thinkingBox = document.getElementById("argon-thinking-box");
    if (data.thinking) {
      if (thinkingBox) thinkingBox.style.display = "block";
      if (thinkingBody) thinkingBody.innerText = data.thinking;
      const count = data.thinking.split(/\s+/).length;
      if (thinkingTokensBadge) thinkingTokensBadge.innerText = `~${count} thinking words • budget: ${data.thinking_budget}`;
    } else {
      if (thinkingBox) thinkingBox.style.display = "none";
    }

    // Synthesized Response
    const responseBody = document.getElementById("argon-response-body");
    const modelBadge = document.getElementById("argon-model-used-badge");
    if (modelBadge) modelBadge.innerText = data.model;
    if (responseBody) responseBody.innerHTML = renderMarkdown(data.response);

    // Auto-populate Assimilation Fields
    const topicInput = document.getElementById("argon-assimilate-topic");
    if (topicInput) {
      let derivedTopic = prompt.split("\n")[0].substring(0, 70);
      if (derivedTopic.length >= 70) derivedTopic += "...";
      topicInput.value = derivedTopic;
    }
  } catch (err) {
    clearInterval(argonTimerInterval);
    if (statusEl) statusEl.innerText = `⚠️ Network Error: ${err.message}`;
    showToast(`Network error: ${err.message}`, "danger");
  } finally {
    if (runBtn) runBtn.disabled = false;
  }
}

function copyArgonResponse() {
  if (!currentArgonResult || !currentArgonResult.response) {
    showToast("No response to copy.", "warning");
    return;
  }
  navigator.clipboard.writeText(currentArgonResult.response).then(() => {
    showToast("Copied Argon response to clipboard!", "success");
  }).catch(() => {
    showToast("Could not copy to clipboard.", "warning");
  });
}

async function assimilateArgonToMemory() {
  if (!currentArgonResult || !currentArgonResult.response) {
    showToast("Please run an Argon reasoning query first.", "warning");
    return;
  }

  const topicInput = document.getElementById("argon-assimilate-topic");
  const tagsInput = document.getElementById("argon-assimilate-tags");
  const assimilateBtn = document.getElementById("btn-argon-assimilate");

  const topic = (topicInput ? topicInput.value.trim() : "") || "Gemini 4 Argon Distillation";
  const tags = (tagsInput ? tagsInput.value.trim() : "") || "gemini-4-argon,reasoning,distilled";

  let codeSnippet = "";
  const codeMatch = currentArgonResult.response.match(/```(?:[a-zA-Z0-9_\-\+]*)\n([\s\S]*?)```/);
  if (codeMatch) {
    codeSnippet = codeMatch[1].trim();
  }

  const lesson = currentArgonResult.response;

  if (assimilateBtn) assimilateBtn.disabled = true;

  try {
    const res = await fetch("/api/teacher/argon/assimilate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        topic: topic,
        lesson: lesson,
        code_example: codeSnippet,
        tags: tags
      })
    });
    const d = await res.json();
    if (d.success) {
      showToast(`✓ Assimilated '#${d.id}: ${topic}' into local SQLite memory!`, "success");
      loadSkills();
    } else {
      showToast(d.error || d.detail || "Assimilation failed.", "danger");
    }
  } catch (e) {
    showToast(`Assimilation error: ${e.message}`, "danger");
  } finally {
    if (assimilateBtn) assimilateBtn.disabled = false;
  }
}

async function runDatabaseVacuum() {
  const btn = document.getElementById("btn-vacuum-db");
  const origText = btn ? btn.innerText : "";
  if (btn) {
    btn.disabled = true;
    btn.innerText = "Vacuuming...";
  }
  try {
    const res = await fetch("/api/maintenance/vacuum", { method: "POST" });
    const data = await res.json();
    if (res.ok && data.status === "ok") {
      showToast("Database optimized and WAL checkpointed successfully.", "success");
      loadTelemetry();
    } else {
      showToast("Vacuum failed: " + (data.detail || "Unknown error"), "error");
    }
  } catch (err) {
    showToast("Error optimizing database: " + err.message, "error");
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerText = origText;
    }
  }
}

// ==========================================================================
// CALENDAR & APPOINTMENT SCHEDULING UI
// ==========================================================================
async function loadCalendarEvents() {
  const container = document.getElementById("calendar-events-list");
  if (!container) return;
  try {
    const res = await fetch("/api/calendar/events?limit=50");
    const data = await res.json();
    const events = data.events || [];

    const badge = document.getElementById("badge-calendar");
    if (badge) {
      const activeCount = events.filter(e => e.status === "confirmed").length;
      badge.innerText = activeCount;
      badge.style.display = activeCount > 0 ? "inline-block" : "none";
    }

    if (!events.length) {
      container.innerHTML = '<div style="color:var(--text-muted);font-size:0.82rem;padding:12px 0;">No appointments scheduled yet.</div>';
      return;
    }

    container.innerHTML = events.map(e => `
      <div style="background:var(--bg-canvas);border:1px solid var(--border-color);border-radius:6px;padding:10px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">
        <div style="flex:1;">
          <div style="display:flex;align-items:center;gap:8px;">
            <span style="font-weight:600;font-size:0.88rem;">${escapeHtml(e.title)}</span>
            <span class="nav-badge ${e.status === 'confirmed' ? '' : 'warning'}" style="font-size:0.7rem;">${escapeHtml(e.status.toUpperCase())}</span>
          </div>
          <div style="font-size:0.8rem;color:var(--text-muted);margin-top:3px;">
            🕒 <b>${escapeHtml(e.start_time)}</b> to <b>${escapeHtml(e.end_time)}</b>
          </div>
          <div style="font-size:0.78rem;color:var(--text-main);margin-top:2px;">
            👤 ${escapeHtml(e.attendee_name || "Guest")} ${e.attendee_phone ? '· 📞 ' + escapeHtml(e.attendee_phone) : ''} ${e.attendee_email ? '· ✉️ ' + escapeHtml(e.attendee_email) : ''}
          </div>
          ${e.description ? `<div style="font-size:0.75rem;color:var(--text-muted);margin-top:4px;">${escapeHtml(e.description)}</div>` : ''}
        </div>
        <div style="display:flex;gap:6px;">
          ${e.status === 'confirmed' ? `<button class="btn-secondary" style="padding:4px 8px;font-size:0.75rem;" onclick="cancelCalendarEventItem(${e.id})">Cancel</button>` : ''}
          <button class="btn-secondary" style="padding:4px 8px;font-size:0.75rem;color:#ef4444;" onclick="deleteCalendarEventItem(${e.id})">Delete</button>
        </div>
      </div>
    `).join("");
  } catch (err) {
    container.innerHTML = `<div style="color:var(--danger-color);font-size:0.82rem;">Failed to load events: ${escapeHtml(err.message)}</div>`;
  }
}

async function checkCalendarSlot() {
  const start = document.getElementById("cal-check-start").value;
  const end = document.getElementById("cal-check-end").value;
  const resEl = document.getElementById("cal-avail-result");
  if (!start || !end) {
    showToast("Please select both start and end times to check.", "error");
    return;
  }
  try {
    const res = await fetch("/api/calendar/availability", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ start_time: start, end_time: end })
    });
    const d = await res.json();
    if (d.available) {
      resEl.style.color = "var(--success-color)";
      resEl.innerText = "✅ Time slot is completely free and available!";
    } else {
      resEl.style.color = "var(--danger-color)";
      resEl.innerText = `❌ Conflicting appointment: ${d.conflicts && d.conflicts[0] ? d.conflicts[0].title : 'Busy'}`;
    }
  } catch (err) {
    resEl.style.color = "var(--danger-color)";
    resEl.innerText = "Error checking availability.";
  }
}

async function bookCalendarAppointmentManual() {
  const title = document.getElementById("cal-book-title").value.trim();
  const name = document.getElementById("cal-book-name").value.trim();
  const start = document.getElementById("cal-book-start").value;
  const end = document.getElementById("cal-book-end").value;
  const phone = document.getElementById("cal-book-phone").value.trim();
  const email = document.getElementById("cal-book-email").value.trim();
  const desc = document.getElementById("cal-book-desc").value.trim();

  if (!title || !name || !start || !end) {
    showToast("Title, attendee name, start time, and end time are required.", "error");
    return;
  }

  try {
    const res = await fetch("/api/calendar/book", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title, attendee_name: name, start_time: start, end_time: end,
        attendee_phone: phone, attendee_email: email, description: desc
      })
    });
    const d = await res.json();
    if (res.status === 409 || d.status === "conflict") {
      showToast(`Conflict: ${d.error || 'Slot already booked'}`, "error");
      return;
    }
    showToast(`Appointment booked successfully: ${d.title}`, "success");
    document.getElementById("cal-book-title").value = "";
    document.getElementById("cal-book-name").value = "";
    document.getElementById("cal-book-desc").value = "";
    loadCalendarEvents();
  } catch (err) {
    showToast(`Booking error: ${err.message}`, "error");
  }
}

async function cancelCalendarEventItem(id) {
  try {
    const res = await fetch("/api/calendar/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event_id: id })
    });
    const d = await res.json();
    if (d.success) {
      showToast("Appointment cancelled.", "info");
      loadCalendarEvents();
    }
  } catch (err) {
    showToast("Error cancelling event.", "error");
  }
}

async function deleteCalendarEventItem(id) {
  if (!confirm("Are you sure you want to permanently delete this calendar record?")) return;
  try {
    const res = await fetch("/api/calendar/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event_id: id })
    });
    const d = await res.json();
    if (d.success) {
      showToast("Appointment deleted.", "info");
      loadCalendarEvents();
    }
  } catch (err) {
    showToast("Error deleting event.", "error");
  }
}

// ==========================================================================
// TELEPHONY & VOIP PBX GATEWAY UI
// ==========================================================================
async function loadTelephonyStatus() {
  const container = document.getElementById("telephony-channels-list");
  if (!container) return;
  try {
    const res = await fetch("/api/telephony/status");
    const d = await res.json();
    const calls = d.active_calls || [];
    if (!calls.length) {
      container.innerHTML = '<div style="color:var(--text-muted);font-size:0.82rem;padding:8px 0;">No active VoIP channels currently streaming. PBX AudioSocket gateway is ready for incoming connections.</div>';
      return;
    }
    container.innerHTML = calls.map(c => `
      <div style="background:var(--bg-canvas);border:1px solid var(--border-color);border-radius:6px;padding:8px;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center;">
        <div>
          <b style="font-size:0.84rem;">📞 ${escapeHtml(c.caller_number)}</b> (${escapeHtml(c.caller_name)})
          <div style="font-size:0.75rem;color:var(--text-muted);">Session: ${escapeHtml(c.session_id)}</div>
        </div>
        <span class="nav-badge">LIVE</span>
      </div>
    `).join("");
  } catch (err) {
    container.innerHTML = `<div style="color:var(--danger-color);font-size:0.82rem;">Error: ${escapeHtml(err.message)}</div>`;
  }
}

async function simulateTelephonyCall() {
  const caller = document.getElementById("telephony-caller-id").value.trim() || "+1-555-0199";
  const query = document.getElementById("telephony-sim-query").value.trim();
  const respEl = document.getElementById("telephony-sim-response");
  if (!query) {
    showToast("Please enter what the caller says.", "error");
    return;
  }
  respEl.style.display = "block";
  respEl.innerHTML = '<span style="color:var(--text-muted);">📞 Connecting to Merlin Telephony Receptionist...</span>';
  try {
    const res = await fetch("/api/telephony/webhook", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ caller, query })
    });
    const d = await res.json();
    respEl.innerHTML = `
      <div style="font-size:0.78rem;color:var(--text-muted);margin-bottom:4px;"><b>Merlin Voice Reception:</b></div>
      <div style="line-height:1.4;">${renderMarkdown(d.reply)}</div>
    `;
    loadCalendarEvents();
  } catch (err) {
    respEl.innerHTML = `<span style="color:var(--danger-color);">Telephony error: ${escapeHtml(err.message)}</span>`;
  }
}

// ==========================================================================
// USER AUTHENTICATION & LOGIN (PER-USER SAVED CHATS & GUEST CHAT)
// ==========================================================================

async function checkAuthStatus() {
  const token = getAuthToken();
  const badge = document.getElementById("user-auth-badge");
  const nameLabel = document.getElementById("user-name-label");
  const avatarIcon = document.getElementById("user-avatar-icon");
  const sidebarTitle = document.getElementById("sidebar-conv-title");

  function setLoggedOutState() {
    currentUser = null;
    document.body.classList.remove("admin-logged-in");
    if (badge) badge.classList.remove("logged-in");
    if (nameLabel) nameLabel.innerText = "Sign In";
    if (avatarIcon) avatarIcon.innerText = "👤";
    if (sidebarTitle) sidebarTitle.innerText = "Guest Mode (Sign in to save)";
    const guestBadge = document.getElementById("sentinel-guest-badge");
    if (guestBadge) {
      guestBadge.style.display = "block";
      guestBadge.innerHTML = 'GUEST MODE (EPHEMERAL) &bull; ZERO CHATS STORED &bull; AUTONOMOUS LEARNING ACTIVE';
      guestBadge.style.color = "#ff9100";
    }
  }

  if (!token) {
    setLoggedOutState();
    return;
  }

  try {
    const res = await fetch("/api/auth/me", { headers: authHeaders() });
    if (!res.ok) throw new Error("Auth check failed");
    const data = await res.json();
    if (data.logged_in && data.user) {
      currentUser = data.user;
      if (badge) badge.classList.add("logged-in");
      if (currentUser.is_admin) {
        document.body.classList.add("admin-logged-in");
        if (avatarIcon) avatarIcon.innerText = "👑";
        if (nameLabel) nameLabel.innerText = `${currentUser.display_name || currentUser.username} (Admin)`;
      } else {
        document.body.classList.remove("admin-logged-in");
        if (avatarIcon) avatarIcon.innerText = "⭐";
        if (nameLabel) nameLabel.innerText = currentUser.display_name || currentUser.username;
      }
      if (sidebarTitle) sidebarTitle.innerText = `${currentUser.display_name || currentUser.username}'s Saved Chats`;
      const guestBadge = document.getElementById("sentinel-guest-badge");
      if (guestBadge) {
        guestBadge.style.display = "block";
        guestBadge.innerHTML = `OPERATOR ONLINE: @${currentUser.username} &bull; CONVERSATIONS SAVED TO ACCOUNT`;
        guestBadge.style.color = "#4ade80";
      }
    } else {
      localStorage.removeItem("merlin_auth_token");
      setLoggedOutState();
    }
  } catch (e) {
    setLoggedOutState();
  }
}

function openAuthModal() {
  openModal("auth-modal");
}

function renderAuthModalContent() {
  const loggedInView = document.getElementById("auth-logged-in-view");
  const formView = document.getElementById("auth-form-view");
  const currentUsernameEl = document.getElementById("auth-current-username");
  const currentMetaEl = document.getElementById("auth-current-meta");
  const btnClaim = document.getElementById("btn-claim-chat");
  const errorMsg = document.getElementById("auth-error-msg");

  if (errorMsg) {
    errorMsg.style.display = "none";
    errorMsg.innerText = "";
  }

  if (currentUser) {
    if (loggedInView) loggedInView.style.display = "block";
    if (formView) formView.style.display = "none";
    if (currentUsernameEl) {
      const adminTag = currentUser.is_admin ? " [ADMINISTRATOR]" : "";
      currentUsernameEl.innerText = (currentUser.display_name
        ? `${currentUser.display_name} (@${currentUser.username})`
        : `@${currentUser.username}`) + adminTag;
    }
    if (currentMetaEl) {
      currentMetaEl.innerText = currentUser.is_admin
        ? "Administrator access granted: Full system telemetry, teacher sync & settings unlocked."
        : "Your chats, notes, and memory are securely saved to your account.";
    }
    if (btnClaim) {
      btnClaim.style.display = activeConversationId ? "block" : "none";
    }
  } else {
    if (loggedInView) loggedInView.style.display = "none";
    if (formView) formView.style.display = "block";
    switchAuthTab(currentAuthTab || "login");
  }
}

function switchAuthTab(tab) {
  currentAuthTab = tab;
  const tabLogin = document.getElementById("tab-auth-login");
  const tabRegister = document.getElementById("tab-auth-register");
  const groupDisplayName = document.getElementById("group-display-name");
  const btnSubmit = document.getElementById("btn-auth-submit");
  const errorMsg = document.getElementById("auth-error-msg");

  if (errorMsg) {
    errorMsg.style.display = "none";
    errorMsg.innerText = "";
  }

  if (tab === "register") {
    if (tabLogin) tabLogin.classList.remove("active");
    if (tabRegister) tabRegister.classList.add("active");
    if (groupDisplayName) groupDisplayName.style.display = "block";
    if (btnSubmit) btnSubmit.innerText = "Create Account";
  } else {
    if (tabLogin) tabLogin.classList.add("active");
    if (tabRegister) tabRegister.classList.remove("active");
    if (groupDisplayName) groupDisplayName.style.display = "none";
    if (btnSubmit) btnSubmit.innerText = "Log In";
  }
}

async function handleAuthSubmit(event) {
  event.preventDefault();
  const usernameInput = document.getElementById("auth-username");
  const passwordInput = document.getElementById("auth-password");
  const displayNameInput = document.getElementById("auth-display-name");
  const errorMsg = document.getElementById("auth-error-msg");
  const btnSubmit = document.getElementById("btn-auth-submit");

  const username = usernameInput ? usernameInput.value.trim() : "";
  const password = passwordInput ? passwordInput.value : "";
  const displayName = displayNameInput ? displayNameInput.value.trim() : "";

  if (!username || !password) {
    if (errorMsg) {
      errorMsg.innerText = "Username and password are required.";
      errorMsg.style.display = "block";
    }
    return;
  }

  if (btnSubmit) btnSubmit.disabled = true;
  if (errorMsg) {
    errorMsg.style.display = "none";
    errorMsg.innerText = "";
  }

  const endpoint = currentAuthTab === "register" ? "/api/auth/register" : "/api/auth/login";
  const payload = { username, password };
  if (currentAuthTab === "register" && displayName) {
    payload.display_name = displayName;
  }

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) {
      const err = data.detail || data.error || "Authentication failed.";
      if (errorMsg) {
        errorMsg.innerText = err;
        errorMsg.style.display = "block";
      }
      return;
    }

    if (data.token) {
      localStorage.setItem("merlin_auth_token", data.token);
    }
    currentUser = data.user;

    // If there is an active guest conversation, claim it for this account
    if (activeConversationId) {
      try {
        await fetch("/api/auth/claim_session", {
          method: "POST",
          headers: authHeaders(),
          body: JSON.stringify({ session_id: activeConversationId })
        });
      } catch (claimErr) {
        console.warn("Could not auto-claim current session:", claimErr);
      }
    }

    closeModal("auth-modal");
    showToast(`Welcome, ${currentUser.display_name || currentUser.username}!`, "success");
    await checkAuthStatus();
    await loadConversations();
  } catch (err) {
    if (errorMsg) {
      errorMsg.innerText = `Network error: ${err.message}`;
      errorMsg.style.display = "block";
    }
  } finally {
    if (btnSubmit) btnSubmit.disabled = false;
  }
}

async function handleLogout() {
  try {
    await fetch("/api/auth/logout", {
      method: "POST",
      headers: authHeaders()
    });
  } catch (e) {
    console.warn("Logout request failed:", e);
  }
  localStorage.removeItem("merlin_auth_token");
  currentUser = null;
  closeModal("auth-modal");
  showToast("Logged out successfully. You are now in guest chat mode.", "info");
  await checkAuthStatus();
  await loadConversations();
}

async function claimActiveChat() {
  if (!activeConversationId) {
    showToast("No active conversation to claim.", "warning");
    return;
  }
  try {
    const res = await fetch("/api/auth/claim_session", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ session_id: activeConversationId })
    });
    const d = await res.json();
    if (res.ok && d.success) {
      showToast("Conversation attached to your account!", "success");
      closeModal("auth-modal");
      await loadConversations();
    } else {
      showToast(d.detail || "Could not claim conversation.", "danger");
    }
  } catch (err) {
    showToast(`Error claiming conversation: ${err.message}`, "danger");
  }
}

// ==========================================================================
// APPLICATION BOOTSTRAP
// ==========================================================================
window.addEventListener("DOMContentLoaded", async () => {
  initTheme();
  await checkAuthStatus();
  loadConversations();
  loadTodos();
  loadModels();
  loadCalendarEvents();

  // Ensure sidebar collapses by default on mobile / small screens
  if (window.innerWidth <= 768) {
    closeSidebar();
  }

  // Synchronize sidebar backdrop and responsive classes on window resize
  window.addEventListener("resize", () => {
    const backdrop = document.getElementById("sidebar-backdrop");
    const sb = document.getElementById("sidebar");
    if (window.innerWidth > 768) {
      if (backdrop) backdrop.style.display = "none";
      document.body.classList.remove("sidebar-open");
    } else {
      if (sb && sb.classList.contains("collapsed") && backdrop) {
        backdrop.style.display = "none";
      }
    }
  });

  // Mobile Touch Swipe-to-Close gesture for sidebar
  const sb = document.getElementById("sidebar");
  if (sb) {
    let touchStartX = 0;
    let touchStartY = 0;
    sb.addEventListener("touchstart", (e) => {
      touchStartX = e.changedTouches[0].screenX;
      touchStartY = e.changedTouches[0].screenY;
    }, { passive: true });
    sb.addEventListener("touchend", (e) => {
      const diffX = e.changedTouches[0].screenX - touchStartX;
      const diffY = e.changedTouches[0].screenY - touchStartY;
      if (diffX < -50 && Math.abs(diffX) > Math.abs(diffY)) {
        if (window.innerWidth <= 768) {
          closeSidebar();
        }
      }
    }, { passive: true });
  }

  // Global Keyboard Shortcuts
  window.addEventListener("keydown", (e) => {
    // Ctrl+N or Cmd+N: New Chat
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "n") {
      const tag = document.activeElement ? document.activeElement.tagName.toLowerCase() : "";
      if (tag !== "input" && tag !== "textarea") {
        e.preventDefault();
        createNewChat();
      }
    }
    // Escape: Close sidebar on mobile if open, or close open modals
    if (e.key === "Escape") {
      const sbEl = document.getElementById("sidebar");
      if (window.innerWidth <= 768 && sbEl && !sbEl.classList.contains("collapsed")) {
        closeSidebar();
        return;
      }
      const openModals = document.querySelectorAll(".modal-backdrop");
      openModals.forEach(m => {
        if (m.style.display === "flex" && m.id) {
          closeModal(m.id);
        }
      });
    }
  });

  // Close modals when clicking outside modal box (using closeModal so timers are cleaned up)
  window.addEventListener("click", (e) => {
    if (e.target.classList.contains("modal-backdrop") && e.target.id) {
      closeModal(e.target.id);
    }
  });

  // Clean up SSE and active timers on page unload
  window.addEventListener("beforeunload", () => {
    if (activeEventSource) {
      activeEventSource.close();
      activeEventSource = null;
    }
    if (syncPollTimer) clearTimeout(syncPollTimer);
    if (datasetPollTimer) clearTimeout(datasetPollTimer);
    if (argonTimerInterval) clearInterval(argonTimerInterval);
  });
});
