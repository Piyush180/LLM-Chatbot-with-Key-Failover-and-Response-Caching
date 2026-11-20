import { useEffect, useRef, useState } from "react";
import { newSessionId, sendChat } from "./api.js";

function AssistantMessage({ reply, metadata }) {
  return (
    <div className="msg assistant" data-testid="assistant-message">
      <div className="msg-text">{reply}</div>
      <div className="meta">
        <span className={`badge ${metadata.resolution_source === "LLM_CACHE" ? "cache" : "live"}`}>
          <span data-testid="meta-source">{metadata.resolution_source}</span>
        </span>
        <span className="meta-item">
          model <span data-testid="meta-model">{metadata.model}</span>
        </span>
        <span className="meta-item">
          latency <span data-testid="meta-latency">{metadata.latency_ms} ms</span>
        </span>
        <span className="meta-item">
          in <span data-testid="meta-input-tokens">{metadata.input_tokens}</span>
        </span>
        <span className="meta-item">
          out <span data-testid="meta-output-tokens">{metadata.output_tokens}</span>
        </span>
      </div>
    </div>
  );
}

export default function App() {
  // A new session starts on every page load.
  const [sessionId, setSessionId] = useState(newSessionId);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [pending, setPending] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, pending]);

  async function handleSend() {
    const text = input;
    if (!text.trim() || pending) return;
    setInput(""); // chat-input is empty after sending
    setMessages((m) => [...m, { kind: "user", content: text.trim() }]);
    setPending(true);

    const result = await sendChat({ sessionId, message: text, systemPrompt });

    setMessages((m) => [
      ...m,
      result.ok
        ? { kind: "assistant", reply: result.data.reply, metadata: result.data.metadata }
        : { kind: "error", content: result.error },
    ]);
    setPending(false);
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleNewChat() {
    setSessionId(newSessionId());
    setMessages([]);
    setInput("");
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>LLM Chatbot</h1>
          <div className="session">session {sessionId}</div>
        </div>
        <button className="secondary" onClick={handleNewChat} disabled={pending}>
          New chat
        </button>
      </header>

      <label className="system-prompt">
        <span>System prompt (optional, applies to this session)</span>
        <textarea
          data-testid="system-prompt-input"
          rows={2}
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          placeholder="e.g. You are a pirate. Answer in one sentence."
        />
      </label>

      <main className="chat">
        {messages.length === 0 && <div className="empty">Send a message to start the conversation.</div>}
        {messages.map((m, i) => {
          if (m.kind === "user") {
            return (
              <div key={i} className="msg user">
                <div className="msg-text">{m.content}</div>
              </div>
            );
          }
          if (m.kind === "assistant") {
            return <AssistantMessage key={i} reply={m.reply} metadata={m.metadata} />;
          }
          return (
            <div key={i} className="msg error" role="alert" data-testid="error-message">
              {m.content}
            </div>
          );
        })}
        {pending && <div className="msg assistant thinking">Thinking…</div>}
        <div ref={bottomRef} />
      </main>

      <footer className="composer">
        <textarea
          data-testid="chat-input"
          rows={2}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message — Enter to send, Shift+Enter for a new line"
        />
        <button data-testid="send-button" onClick={handleSend} disabled={pending || !input.trim()}>
          Send
        </button>
      </footer>
    </div>
  );
}
