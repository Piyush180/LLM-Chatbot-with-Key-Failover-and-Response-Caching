// Base URL of the backend, baked in at build time (VITE_API_URL).
const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/+$/, "");

export function newSessionId() {
  const rand =
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : Math.random().toString(36).slice(2) + Date.now().toString(36);
  return `s-${rand}`;
}

// Turn a FastAPI error body into one readable sentence.
function describeError(status, body) {
  const detail = body && body.detail;
  if (status === 422 && Array.isArray(detail)) {
    const parts = detail.map((d) => {
      const field = Array.isArray(d.loc) ? d.loc.filter((p) => p !== "body").join(".") : "";
      return field ? `${field}: ${d.msg}` : d.msg;
    });
    return `Invalid request — ${parts.join("; ")}`;
  }
  if (status === 503) {
    return "All providers are unavailable right now. Please try again in a moment.";
  }
  if (typeof detail === "string") return `Error ${status}: ${detail}`;
  return `Unexpected error (HTTP ${status}).`;
}

export async function sendChat({ sessionId, message, systemPrompt }) {
  const payload = { session_id: sessionId, message };
  if (systemPrompt && systemPrompt.trim()) payload.system_prompt = systemPrompt.trim();

  let res;
  try {
    res = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    return { ok: false, error: "Could not reach the server. Check your connection and try again." };
  }

  let body = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }

  if (!res.ok || !body) return { ok: false, error: describeError(res.status, body) };
  return { ok: true, data: body };
}
