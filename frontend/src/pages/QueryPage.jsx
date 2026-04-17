import { useState } from 'react'
import { api } from '../utils/api.js'
import { Search, Loader2, ExternalLink, ThumbsUp, ThumbsDown, ChevronDown, ChevronUp, Zap, Target, Brain } from 'lucide-react'

const EXAMPLE_QUERIES = [
  "What are the best techniques for reducing hallucination in RAG pipelines?",
  "How does episodic memory improve LLM agents?",
  "What is MemRL and how does it work?",
  "Compare GraphRAG vs standard RAG for multi-hop reasoning",
  "What are the latest advances in agentic AI systems?",
]

const strategyColor = {
  semantic: 'var(--blue)',
  graph: 'var(--purple)',
  hybrid: 'var(--teal)',
  aggressive_rewrite: 'var(--amber)',
}

export default function QueryPage() {
  const [query, setQuery] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showSources, setShowSources] = useState(false)
  const [feedbackSent, setFeedbackSent] = useState(false)

  async function submit(q) {
    const text = (q || query).trim()
    if (!text) return
    setQuery(text)
    setLoading(true)
    setError(null)
    setResult(null)
    setShowSources(false)
    setFeedbackSent(false)
    try {
      const res = await api.query(text)
      setResult(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function sendFeedback(score) {
    if (!result || feedbackSent) return
    await api.feedback(result.session_id, score).catch(() => {})
    setFeedbackSent(true)
  }

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: '32px 24px' }}>
      {/* Query input */}
      <div style={{ marginBottom: 24 }}>
        <div style={{
          display: 'flex', gap: 10,
          background: 'var(--bg2)',
          border: '1px solid var(--border2)',
          borderRadius: 12,
          padding: '4px 4px 4px 16px',
          transition: 'border-color 0.2s',
        }}>
          <Search size={16} color="var(--text3)" style={{ marginTop: 13, flexShrink: 0 }} />
          <textarea
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() } }}
            placeholder="Ask a research question about AI papers…"
            rows={2}
            style={{
              flex: 1, background: 'transparent', border: 'none', outline: 'none',
              color: 'var(--text)', fontSize: 14, resize: 'none', paddingTop: 10, lineHeight: 1.5,
            }}
          />
          <button onClick={() => submit()} disabled={loading || !query.trim()} style={{
            background: loading || !query.trim() ? 'var(--border2)' : 'var(--teal)',
            color: loading || !query.trim() ? 'var(--text3)' : '#000',
            border: 'none', borderRadius: 8, padding: '8px 20px',
            fontWeight: 600, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6,
            transition: 'all 0.15s', alignSelf: 'flex-end', marginBottom: 4,
          }}>
            {loading ? <Loader2 size={14} style={{ animation: 'spin 1s linear infinite' }} /> : <Zap size={14} />}
            {loading ? 'Thinking…' : 'Ask'}
          </button>
        </div>

        {/* Example queries */}
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 10 }}>
          {EXAMPLE_QUERIES.slice(0, 3).map(q => (
            <button key={q} onClick={() => submit(q)} style={{
              background: 'var(--bg3)', border: '1px solid var(--border)',
              borderRadius: 20, padding: '4px 12px',
              fontSize: 11, color: 'var(--text2)', cursor: 'pointer',
            }}>
              {q.length > 55 ? q.slice(0, 55) + '…' : q}
            </button>
          ))}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{
          background: 'var(--red-dim)', border: '1px solid var(--red)',
          borderRadius: 10, padding: '14px 16px', color: 'var(--red)', marginBottom: 20
        }}>
          ⚠ {error}
        </div>
      )}

      {/* Loading skeleton */}
      {loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {[1, 2, 3].map(i => (
            <div key={i} style={{
              height: i === 1 ? 80 : 40,
              background: 'var(--bg2)',
              borderRadius: 8,
              animation: 'pulse 1.5s ease-in-out infinite',
              opacity: 0.6,
            }} />
          ))}
          <style>{`@keyframes pulse { 0%,100%{opacity:.4} 50%{opacity:.8} } @keyframes spin { to{transform:rotate(360deg)} }`}</style>
        </div>
      )}

      {/* Result */}
      {result && !loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Meta row */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
            <Chip label="Session" value={`#${result.session_number}`} color="var(--text2)" />
            <Chip label="Strategy" value={result.retrieval_strategy} color={strategyColor[result.retrieval_strategy] || 'var(--text2)'} />
            <Chip label="Confidence" value={`${(result.confidence * 100).toFixed(0)}%`}
              color={result.confidence > 0.7 ? 'var(--teal)' : result.confidence > 0.4 ? 'var(--amber)' : 'var(--red)'} />
            <Chip label="Hallucination" value={`${(result.hallucination_score * 100).toFixed(0)}%`}
              color={result.hallucination_score < 0.2 ? 'var(--teal)' : result.hallucination_score < 0.4 ? 'var(--amber)' : 'var(--red)'} />
            <Chip label="Rewrites" value={result.rewrite_count} color="var(--text2)" />
            <Chip label="Papers" value={result.papers_retrieved} color="var(--text2)" />
            <Chip label="Duration" value={`${result.duration_ms}ms`} color="var(--text3)" />
          </div>

          {/* Intent */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 8,
            padding: '8px 14px', background: 'var(--bg2)', borderRadius: 8,
            border: '1px solid var(--border)',
          }}>
            <Brain size={13} color="var(--purple)" />
            <span style={{ fontSize: 12, color: 'var(--text3)' }}>Intent:</span>
            <span style={{ fontSize: 12, color: 'var(--purple)', fontWeight: 500 }}>{result.intent}</span>
            <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--text3)' }}>Q-value: {result.q_value_after}</span>
          </div>

          {/* Answer */}
          <div style={{
            background: 'var(--bg2)', border: '1px solid var(--border)',
            borderRadius: 12, padding: '20px 22px',
          }}>
            <div style={{ fontSize: 13, color: 'var(--text)', lineHeight: 1.8, whiteSpace: 'pre-wrap' }}>
              {result.answer}
            </div>

            {/* Feedback */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 16, paddingTop: 14, borderTop: '1px solid var(--border)' }}>
              <span style={{ fontSize: 12, color: 'var(--text3)', marginRight: 4 }}>Helpful?</span>
              <button onClick={() => sendFeedback(1.0)} disabled={feedbackSent} style={{
                background: feedbackSent ? 'var(--bg3)' : 'var(--bg3)',
                border: '1px solid var(--border)', borderRadius: 6,
                padding: '4px 10px', color: feedbackSent ? 'var(--teal)' : 'var(--text2)',
                display: 'flex', alignItems: 'center', gap: 4, fontSize: 12,
              }}>
                <ThumbsUp size={12} /> Yes
              </button>
              <button onClick={() => sendFeedback(0.0)} disabled={feedbackSent} style={{
                background: 'var(--bg3)', border: '1px solid var(--border)',
                borderRadius: 6, padding: '4px 10px',
                color: 'var(--text2)', display: 'flex', alignItems: 'center', gap: 4, fontSize: 12,
              }}>
                <ThumbsDown size={12} /> No
              </button>
            </div>
          </div>

          {/* Sources toggle */}
          {result.sources?.length > 0 && (
            <div>
              <button onClick={() => setShowSources(!showSources)} style={{
                background: 'none', border: 'none',
                color: 'var(--text2)', fontSize: 13,
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '4px 0',
              }}>
                {showSources ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                {result.sources.length} source papers
              </button>

              {showSources && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 10 }}>
                  {result.sources.map((s, i) => (
                    <div key={i} style={{
                      background: 'var(--bg2)', border: '1px solid var(--border)',
                      borderRadius: 10, padding: '14px 16px',
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
                        <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--text)', lineHeight: 1.4 }}>{s.title}</div>
                        <a href={s.url} target="_blank" rel="noreferrer" style={{ flexShrink: 0 }}>
                          <ExternalLink size={13} color="var(--teal)" />
                        </a>
                      </div>
                      {s.authors?.length > 0 && (
                        <div style={{ fontSize: 11, color: 'var(--text3)', marginTop: 4 }}>
                          {s.authors.join(', ')}
                        </div>
                      )}
                      <div style={{ fontSize: 12, color: 'var(--text2)', marginTop: 8, lineHeight: 1.6 }}>
                        {s.abstract_snippet}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {!result && !loading && !error && (
        <div style={{ textAlign: 'center', padding: '60px 0', color: 'var(--text3)' }}>
          <Target size={36} style={{ margin: '0 auto 16px', opacity: 0.4 }} />
          <div style={{ fontSize: 15, fontWeight: 500, marginBottom: 8, color: 'var(--text2)' }}>
            Ask a research question
          </div>
          <div style={{ fontSize: 13, maxWidth: 400, margin: '0 auto', lineHeight: 1.7 }}>
            GhostMind fetches papers from arXiv, retrieves the most relevant ones,
            generates an answer, and learns from the outcome to improve next time.
          </div>
        </div>
      )}
    </div>
  )
}

function Chip({ label, value, color }) {
  return (
    <div style={{
      display: 'flex', gap: 5, alignItems: 'center',
      background: 'var(--bg2)', border: '1px solid var(--border)',
      borderRadius: 20, padding: '3px 10px', fontSize: 11,
    }}>
      <span style={{ color: 'var(--text3)' }}>{label}</span>
      <span style={{ color, fontWeight: 500 }}>{value}</span>
    </div>
  )
}
