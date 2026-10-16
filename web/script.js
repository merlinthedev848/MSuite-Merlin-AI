/* ==========================================================================
   MERLIN AI — UI CLIENT ENGINE & STREAMING LOGIC
   ========================================================================== */

let activeConversationId = null;
let currentMode = "auto";
let activeEventSource = null;
let attachedFile = null;
let recognition = null;
let isDictating = false;
let syncPollTimer = null;
let datasetPollTimer = null;
let skillsSearchTimer = null;
let popularDatasetsLoaded = false;

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
  container.appendChild(toast);
  setTimeout(() => {
    toast.classList.add("toast-fade-out");
    setTimeout(() => toast.remove(), 250);
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
function toggleSidebar() {
  const sb = document.getElementById("sidebar");
  const backdrop = document.getElementById("sidebar-backdrop");
  if (sb) {
    sb.classList.toggle("collapsed");
    if (backdrop) {
      if (window.innerWidth <= 768 && !sb.classList.contains("collapsed")) {
        backdrop.style.display = "block";
      } else {
        backdrop.style.display = "none";
      }
    }
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
  const m = document.getElementById(id);
  if (m) m.style.display = "flex";
  if (id === "settings-modal") {
    loadTelemetry();
    loadModels();
    loadAuditLogs();
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

// ==========================================================================
// SESSION & CONVERSATION MANAGEMENT
// ==========================================================================
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
  const res = await fetch("/api/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "New Chat", mode: "auto" })
  });
  const conv = await res.json();
  activeConversationId = conv.id;
  await loadConversations();
  showConversationMessages(conv.id, []);
}

async function switchConversation(convId) {
  activeConversationId = convId;
  loadConversations();
  const res = await fetch(`/api/conversations/${encodeURIComponent(convId)}`);
  const data = await res.json();
  if (data.conversation) {
    showConversationMessages(convId, data.messages || []);
  }
}

async function renameConv(convId) {
  const newTitle = prompt("Enter a new title for this conversation:");
  if (!newTitle || !newTitle.trim()) return;
  await fetch(`/api/conversations/${encodeURIComponent(convId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: newTitle.trim() })
  });
  loadConversations();
}

async function togglePin(convId, newPinned) {
  await fetch(`/api/conversations/${encodeURIComponent(convId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pinned: newPinned })
  });
  loadConversations();
}

async function deleteConv(convId) {
  await fetch(`/api/conversations/${encodeURIComponent(convId)}/delete`, { method: "POST" });
  if (activeConversationId === convId) {
    activeConversationId = null;
    document.getElementById("hero-view").style.display = "flex";
    document.getElementById("messages-list").style.display = "none";
  }
  loadConversations();
}

async function clearCurrentChat() {
  if (activeConversationId) {
    await fetch(`/api/conversations/${encodeURIComponent(activeConversationId)}/clear`, { method: "POST" });
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

  if (!messages || !messages.length) {
    hero.style.display = "flex";
    list.style.display = "none";
    return;
  }

  hero.style.display = "none";
  list.style.display = "flex";

  messages.forEach(m => {
    const displayContent = m.role === "user" ? formatDisplayedUserMessage(m.content) : m.content;
    appendMessageElement(m.role, displayContent);
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

// ==========================================================================
// REAL-TIME TOKEN STREAMING & GENERATION
// ==========================================================================
async function submitMessage() {
  const ta = document.getElementById("chat-textarea");
  const promptText = ta.value.trim();
  if (!promptText && !attachedFile) return;

  if (!activeConversationId) {
    await createNewChat();
  }

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

  ta.value = "";
  clearAttachment();
  autoResizeTextarea(ta);

  // Render User Message
  appendMessageElement("user", displayUserText);

  // Prepare Assistant UI
  const assistantBubble = appendMessageElement("assistant", "");
  assistantBubble.innerHTML = '<span class="status-thinking" style="color:var(--text-muted);font-size:0.88rem;display:inline-flex;align-items:center;gap:6px;">⚡ Generating response...</span>';
  assistantBubble.classList.add("streaming-cursor");
  toggleStreamingUI(true);

  let fullResponse = "";
  let renderScheduled = false;

  function scheduleRender() {
    if (!renderScheduled) {
      renderScheduled = true;
      requestAnimationFrame(() => {
        assistantBubble.innerHTML = renderMarkdown(fullResponse);
        scrollViewportToBottom();
        renderScheduled = false;
      });
    }
  }

  const streamUrl = `/api/chat/stream?prompt=${encodeURIComponent(payloadText)}&mode=auto&session_id=${encodeURIComponent(activeConversationId)}`;
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
        scheduleRender();
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
  bubbleEl.innerHTML = renderMarkdown(fullResponse || "[Response stopped]");
  toggleStreamingUI(false);
  // Note: engine.generate_stream() already saves both the user prompt and the assistant response
  // in SQLite on the backend. Refreshing the conversation list updates the auto-derived title.
  loadConversations();
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
      const d = await res.json();
      if (d.success) {
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
        showToast(d.error || d.detail || "File upload failed", "danger");
      }
    } catch (err) {
      showToast("Error uploading file", "danger");
    }
  };
  reader.readAsDataURL(file);
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
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role: "user", content: userDisplay })
  });

  const assistantBubble = appendMessageElement("assistant", "");
  assistantBubble.innerHTML = '<span class="status-thinking" style="color:var(--text-muted);font-size:0.88rem;">✨ Running multimodal vision/audio analysis...</span>';

  try {
    const res = await fetch("/api/analyze/multimodal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_id: fileObj.id, prompt: userPrompt, api_key: apiKey })
    });
    const d = await res.json();
    const reply = d.success ? d.analysis : `⚠️ Multimodal analysis error: ${d.error || d.detail || "Unknown error"}`;
    assistantBubble.innerHTML = renderMarkdown(reply);
    await fetch(`/api/conversations/${encodeURIComponent(activeConversationId)}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role: "assistant", content: reply })
    });
    loadConversations();
  } catch (err) {
    assistantBubble.innerHTML = renderMarkdown(`⚠️ Multimodal request failed: ${err}`);
  }
}

// ==========================================================================
// VOICE DICTATION
// ==========================================================================
function toggleDictation() {
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    showToast("Speech Recognition is not supported in this browser.", "warning");
    return;
  }
  if (isDictating && recognition) {
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
    showToast("Listening for voice input...", "info");
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

async function loadNotes() {
  try {
    const res = await fetch("/api/notes");
    const notes = await res.json();
    const container = document.getElementById("notes-list-container");
    if (!container) return;

    if (!notes.length) {
      container.innerHTML = '<div class="empty-list-msg">No technical notes saved yet.</div>';
      return;
    }

    container.innerHTML = notes.map(n => `
      <div class="list-card list-card-column">
        <div style="display:flex;justify-content:space-between;align-items:center;width:100%;">
          <b style="font-size:0.9rem;color:var(--accent-primary);">${escapeHtml(n.title)}</b>
          <div style="display:flex;gap:6px;">
            <button class="conv-action-btn" onclick="editNoteItem(${n.id}, ${JSON.stringify(n.title).replace(/"/g, '&quot;')}, ${JSON.stringify(n.content).replace(/"/g, '&quot;')}, ${JSON.stringify(n.tags || '').replace(/"/g, '&quot;')})">✎ Edit</button>
            <button class="conv-action-btn" onclick="deleteNoteItem(${n.id})">✕</button>
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

    select.innerHTML = datasets.map(ds =>
      `<option value="${escapeHtml(ds.id)}" ${ds.is_default ? 'selected' : ''}>${escapeHtml(ds.category)} — ${escapeHtml(ds.name)}</option>`
    ).join("") + '<option value="__custom__">Custom Hugging Face Repository...</option>';
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
  const limit = parseInt(document.getElementById("hf-limit").value, 10) || 50;
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
    showToast(`Started streaming import for ${dataset}`, "success");
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
      const pct = st.total > 0 ? Math.round((st.progress / st.total) * 100) : 25;
      if (fill) fill.style.width = `${pct}%`;
      if (text) text.innerText = `Assimilating ${st.current_dataset}: ${st.progress} / ${st.total} samples...`;
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

// ==========================================================================
// APPLICATION BOOTSTRAP
// ==========================================================================
window.addEventListener("DOMContentLoaded", () => {
  initTheme();
  loadConversations();
  loadTodos();
  loadModels();

  // Close modals when clicking outside modal box
  window.addEventListener("click", (e) => {
    if (e.target.classList.contains("modal-backdrop")) {
      e.target.style.display = "none";
    }
  });
});
