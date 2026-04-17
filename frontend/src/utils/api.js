const BASE = '/api/v1'

async function req(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

export const api = {
  query: (query, session_number) =>
    req('/query', { method: 'POST', body: JSON.stringify({ query, session_number }) }),

  feedback: (session_id, score) =>
    req('/feedback', { method: 'POST', body: JSON.stringify({ session_id, score }) }),

  sessions: (limit = 20) => req(`/sessions?limit=${limit}`),
  session: (id) => req(`/sessions/${id}`),
  benchmarks: () => req('/benchmarks'),
  memory: () => req('/memory'),
  stats: () => req('/stats'),
}
