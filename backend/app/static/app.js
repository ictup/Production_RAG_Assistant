const state = {
  apiKey: localStorage.getItem("rag.apiKey") || "dev-key",
  workspaceId: localStorage.getItem("rag.workspaceId") || "public",
  sessionId: localStorage.getItem("rag.sessionId") || "",
  sessions: [],
  sending: false,
};

const els = {
  apiKey: document.querySelector("#api-key"),
  workspaceId: document.querySelector("#workspace-id"),
  newSession: document.querySelector("#new-session"),
  reloadSessions: document.querySelector("#reload-sessions"),
  sessionList: document.querySelector("#session-list"),
  sessionTitle: document.querySelector("#session-title"),
  status: document.querySelector("#status"),
  messages: document.querySelector("#messages"),
  chatForm: document.querySelector("#chat-form"),
  question: document.querySelector("#question"),
  send: document.querySelector("#send"),
};

function init() {
  els.apiKey.value = state.apiKey;
  els.workspaceId.value = state.workspaceId;
  bindEvents();
  renderEmptyMessages();
  void loadSessions();
}

function bindEvents() {
  els.apiKey.addEventListener("input", () => {
    state.apiKey = els.apiKey.value.trim();
    localStorage.setItem("rag.apiKey", state.apiKey);
  });

  els.workspaceId.addEventListener("input", () => {
    state.workspaceId = els.workspaceId.value.trim() || "public";
    localStorage.setItem("rag.workspaceId", state.workspaceId);
  });

  els.newSession.addEventListener("click", () => {
    void createSession("New RAG conversation");
  });

  els.reloadSessions.addEventListener("click", () => {
    void loadSessions();
  });

  els.chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    void submitQuestion();
  });
}

function authHeaders() {
  return {
    Authorization: `Bearer ${state.apiKey || "dev-key"}`,
    "Content-Type": "application/json",
    "X-Workspace-ID": state.workspaceId || "public",
  };
}

async function apiFetch(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      ...authHeaders(),
      ...(options.headers || {}),
    },
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body);
    } catch {
      detail = await response.text();
    }
    throw new Error(detail);
  }

  return response;
}

async function loadSessions() {
  setStatus("Loading sessions");
  try {
    const response = await apiFetch("/chat/sessions?limit=50&offset=0");
    const body = await response.json();
    state.sessions = body.sessions || [];
    renderSessions();

    if (state.sessionId) {
      const selected = state.sessions.find((item) => item.id === state.sessionId);
      if (selected) {
        selectSession(selected.id, selected.title || "Untitled session");
      }
    }

    setStatus("Ready");
  } catch (error) {
    setError(error.message);
  }
}

async function createSession(title) {
  setStatus("Creating session");
  try {
    const response = await apiFetch("/chat/sessions", {
      method: "POST",
      body: JSON.stringify({
        title,
        metadata: {
          created_by: "web_ui",
        },
      }),
    });
    const body = await response.json();
    const session = body.session;
    state.sessionId = session.id;
    localStorage.setItem("rag.sessionId", state.sessionId);
    state.sessions = [session, ...state.sessions.filter((item) => item.id !== session.id)];
    renderSessions();
    selectSession(session.id, session.title || "Untitled session");
    setStatus("Session ready");
    return session;
  } catch (error) {
    setError(error.message);
    return null;
  }
}

function renderSessions() {
  els.sessionList.innerHTML = "";
  if (!state.sessions.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No sessions yet";
    els.sessionList.append(empty);
    return;
  }

  for (const session of state.sessions) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `session-item${session.id === state.sessionId ? " active" : ""}`;
    button.addEventListener("click", () => {
      selectSession(session.id, session.title || "Untitled session");
    });

    const title = document.createElement("span");
    title.className = "session-title";
    title.textContent = session.title || "Untitled session";

    const id = document.createElement("span");
    id.className = "session-id";
    id.textContent = session.id;

    button.append(title, id);
    els.sessionList.append(button);
  }
}

function selectSession(sessionId, title) {
  state.sessionId = sessionId;
  localStorage.setItem("rag.sessionId", sessionId);
  els.sessionTitle.textContent = title;
  renderSessions();
  void loadHistory(sessionId);
}

async function loadHistory(sessionId) {
  setStatus("Loading history");
  try {
    const response = await apiFetch(
      `/chat/sessions/${encodeURIComponent(sessionId)}/logs?limit=50&offset=0`,
    );
    const body = await response.json();
    els.messages.innerHTML = "";
    for (const log of body.logs || []) {
      appendMessage("user", log.question);
      appendMessage("assistant", log.answer, log.sources || []);
    }
    if (!(body.logs || []).length) {
      renderEmptyMessages();
    }
    setStatus("Ready");
  } catch (error) {
    setError(error.message);
  }
}

async function submitQuestion() {
  const question = els.question.value.trim();
  if (!question || state.sending) {
    return;
  }

  state.sending = true;
  els.send.disabled = true;
  els.question.value = "";
  appendMessage("user", question);
  const assistantMessage = appendMessage("assistant", "");

  try {
    if (!state.sessionId) {
      const title = question.length > 80 ? `${question.slice(0, 77)}...` : question;
      const session = await createSession(title);
      if (!session) {
        return;
      }
    }

    await streamAnswer(question, assistantMessage);
    setStatus("Ready");
  } catch (error) {
    setError(error.message);
    updateMessageContent(assistantMessage, `Request failed: ${error.message}`);
  } finally {
    state.sending = false;
    els.send.disabled = false;
    els.question.focus();
  }
}

async function streamAnswer(question, assistantMessage) {
  setStatus("Streaming answer");
  const response = await apiFetch("/chat/stream", {
    method: "POST",
    body: JSON.stringify({
      question,
      session_id: state.sessionId || null,
    }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";

    for (const block of blocks) {
      const event = parseSseBlock(block);
      if (!event) {
        continue;
      }
      if (event.name === "answer_delta") {
        answer += event.data.delta || "";
        updateMessageContent(assistantMessage, answer);
      }
      if (event.name === "final") {
        renderSources(assistantMessage, event.data.sources || []);
      }
      if (event.name === "error") {
        throw new Error(event.data.message || "stream failed");
      }
    }
  }
}

function parseSseBlock(block) {
  let name = "";
  const dataLines = [];

  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) {
      name = line.replace("event:", "").trim();
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.replace("data:", "").trimStart());
    }
  }

  if (!name || !dataLines.length) {
    return null;
  }

  return {
    name,
    data: JSON.parse(dataLines.join("\n")),
  };
}

function appendMessage(role, text, sources = []) {
  clearEmptyMessages();
  const message = document.createElement("article");
  message.className = `message ${role}`;

  const roleEl = document.createElement("div");
  roleEl.className = "role";
  roleEl.textContent = role === "user" ? "You" : "Assistant";

  const content = document.createElement("div");
  content.className = "content";
  content.textContent = text;

  message.append(roleEl, content);
  els.messages.append(message);
  renderSources(message, sources);
  els.messages.scrollTop = els.messages.scrollHeight;
  return message;
}

function updateMessageContent(message, text) {
  message.querySelector(".content").textContent = text;
  els.messages.scrollTop = els.messages.scrollHeight;
}

function renderSources(message, sources) {
  const existing = message.querySelector(".sources");
  if (existing) {
    existing.remove();
  }
  if (!sources.length) {
    return;
  }

  const wrapper = document.createElement("div");
  wrapper.className = "sources";
  for (const source of sources) {
    const item = document.createElement("div");
    item.className = "source";

    const index = document.createElement("div");
    index.className = "source-index";
    index.textContent = `[${source.source_id}]`;

    const title = document.createElement("div");
    title.className = "source-title";
    title.textContent = source.section
      ? `${source.title} / ${source.section}`
      : source.title;

    item.append(index, title);
    wrapper.append(item);
  }
  message.append(wrapper);
}

function renderEmptyMessages() {
  els.messages.innerHTML = "";
  const empty = document.createElement("div");
  empty.className = "empty";
  empty.textContent = "Create or select a session, then ask a question.";
  els.messages.append(empty);
}

function clearEmptyMessages() {
  for (const node of [...els.messages.querySelectorAll(".empty")]) {
    node.remove();
  }
}

function setStatus(message) {
  els.status.classList.remove("error");
  els.status.textContent = message;
}

function setError(message) {
  els.status.classList.add("error");
  els.status.textContent = message || "Request failed";
}

init();
