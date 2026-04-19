import { useState, useEffect, useRef } from 'react'
import { api } from '../utils/api.js'
import {
  Search, ExternalLink, ThumbsUp, ThumbsDown,
  ChevronDown, ChevronUp, Zap, Target, Brain,
  Database, TrendingUp, Cpu, BookOpen
} from 'lucide-react'

const EXAMPLE_QUERIES = [
  "What are the best techniques for reducing hallucination in RAG systems?",
  "How does episodic memory improve LLM agents?",
  "What is MemRL and how does it work?",
  "Compare GraphRAG vs standard RAG for multi-hop reasoning",
  "What are the latest advances in agentic AI systems?",
  "How does proximal policy optimization (PPO) work?",
  "What is the difference between dense and sparse retrieval in RAG?",
]

const STRATEGY_COLOR = {
  semantic:           '#60a5fa',
  graph:              '#a78bfa',
  hybrid:             '#2dd4a4',
  aggressive_rewrite: '#fbbf24',
}

const STRATEGY_LABEL = {
  semantic:           'Semantic',
  graph:              'Graph',
  hybrid:             'Hybrid',
  aggressive_rewrite: 'Aggressive Rewrite',
}

// Thinking steps that animate while loading
const THINKING_STEPS = [
  { icon: Brain,    label: 'Classifying intent…'       },
  { icon: Database, label: 'Fetching arXiv papers…'    },
  { icon: Search,   label: 'Retrieving documents…'     },
  { icon: Cpu,      label: 'Generating answer…'        },
  { icon: TrendingUp, label: 'Updating memory…'        },
]

function ThinkingPanel({ strategy }) {
  const [step, setStep] = useState(0)
  const [dots, setDots] = useState('')

  useEffect(() => {
    const stepTimer = setInterval(() => {
      setStep(s => Math.min(s + 1, THINKING_STEPS.length - 1))
    }, 1800)
    const dotTimer = setInterval(() => {
      setDots(d => d.length >= 3 ? '' : d + '.')
    }, 400)
    return () => { clearInterval(stepTimer); clearInterval(dotTimer) }
  }, [])

  const strategyColor = strategy ? (STRATEGY_COLOR[strategy] || '#8b90a8') : '#2dd4a4'
  const strategyLabel = strategy ? (STRATEGY_LABEL[strategy] || strategy) : '…'

  return (
    <div style={{
      background: 'var(--bg2)', border: '1px solid var(--border2)',
      borderRadius: 14, padding: '24px 24px 20px', marginTop: 8,
    }}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 22 }}>
        <div style={{
          width: 8, height: 8, borderRadius: '50%',
          background: strategyColor,
          boxShadow: `0 0 8px ${strategyColor}`,
          animation: 'ghostPulse 1.2s ease-in-out infinite',
        }} />
        <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text)' }}>
          GhostMind is thinking{dots}
        </span>
        {strategy && (
          <span style={{
            marginLeft: 'auto', fontSize: 11, fontWeight: 500,
            color: strategyColor, background: `${strategyColor}18`,
            border: `1px solid ${strategyColor}40`,
            borderRadius: 20, padding: '2px 10px',
          }}>
            {strategyLabel} strategy
          </span>
        )}
      </div>

      {/* Steps */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {THINKING_STEPS.map((s, i) => {
          const Icon = s.icon
          const done = i < step
          const active = i === step
          return (
            <div key={i} style={{
              display: 'flex', alignItems: 'center', gap: 10,
              opacity: i > step ? 0.25 : 1,
              transition: 'opacity 0.4s',
            }}>
              <div style={{
                width: 28, height: 28, borderRadius: 8,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                background: done ? '#2dd4a415' : active ? `${strategyColor}20` : 'var(--bg3)',
                border: `1px solid ${done ? '#2dd4a440' : active ? `${strategyColor}50` : 'var(--border)'}`,
                flexShrink: 0,
                transition: 'all 0.4s',
              }}>
                <Icon size={13} color={done ? '#2dd4a4' : active ? strategyColor : 'var(--text3)'} />
              </div>
              <span style={{
                fontSize: 12,
                color: done ? 'var(--text2)' : active ? 'var(--text)' : 'var(--text3)',
                fontWeight: active ? 500 : 400,
                transition: 'color 0.4s',
              }}>
                {s.label}
              </span>
              {done && (
                <span style={{ marginLeft: 'auto', fontSize: 11, color: '#2dd4a4' }}>✓</span>
              )}
              {active && (
                <span style={{ marginLeft: 'auto', fontSize: 11, color: strategyColor }}>
                  <span style={{ animation: 'spinDot 1s linear infinite', display: 'inline-block' }}>⟳</span>
                </span>
              )}
            </div>
          )
        })}
      </div>

      <style>{`
        @keyframes ghostPulse { 0%,100%{opacity:.5;transform:scale(1)} 50%{opacity:1;transform:scale(1.3)} }
        @keyframes spinDot { to{transform:rotate(360deg)} }
        @keyframes spin { to{transform:rotate(360deg)} }
      `}</style>
    </div>
  )
}

function MetricBar({ label, value, color, inverse = false }) {
  // value 0-1
  const pct = Math.round(value * 100)
  const barColor = inverse
    ? (value < 0.2 ? '#2dd4a4' : value < 0.4 ? '#fbbf24' : '#f87171')
    : (value > 0.7 ? '#2dd4a4' : value > 0.4 ? '#fbbf24' : '#f87171')

  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: 11, color: 'var(--text3)' }}>{label}</span>
        <span style={{ fontSize: 11, fontWeight: 600, color: barColor }}>{pct}%</span>
      </div>
      <div style={{ height: 4, background: 'var(--bg3)', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${pct}%`, background: barColor,
          borderRadius: 2, transition: 'width 0.8s ease',
        }} />
      </div>
    </div>
  )
}

function MemRLPanel({ debug, qValueAfter }) {
  const [open, setOpen] = useState(false)
  if (!debug) return null

  const qValues = debug.memory_after?.q_values || {}
  const sorted = Object.entries(qValues).sort((a, b) => b[1].q - a[1].q)
  const bestStrategy = debug.memory_after?.best_strategy
  const deltaPositive = debug.q_delta > 0

  return (
    <div style={{
      background: 'var(--bg2)', border: '1px solid var(--border)',
      borderRadius: 12, overflow: 'hidden',
    }}>
      <button onClick={() => setOpen(!open)} style={{
        width: '100%', background: 'none', border: 'none',
        padding: '10px 16px', display: 'flex', alignItems: 'center', gap: 8,
        color: 'var(--text2)', fontSize: 12,
      }}>
        <Brain size={13} color="var(--purple)" />
        <span style={{ fontWeight: 500, color: 'var(--text)' }}>MemRL Memory</span>
        <span style={{
          marginLeft: 4, fontSize: 11, padding: '1px 8px',
          borderRadius: 20, fontWeight: 500,
          background: deltaPositive ? '#2dd4a415' : '#f8717115',
          color: deltaPositive ? '#2dd4a4' : '#f87171',
          border: `1px solid ${deltaPositive ? '#2dd4a440' : '#f8717140'}`,
        }}>
          Q {deltaPositive ? '↑' : '↓'} {debug.q_delta > 0 ? '+' : ''}{debug.q_delta}
        </span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--text3)' }}>
          bucket: {debug.intent_bucket}
        </span>
        {open ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
      </button>

      {open && (
        <div style={{ padding: '0 16px 16px', borderTop: '1px solid var(--border)' }}>
          {/* Strategy Q-value bars */}
          <div style={{ marginTop: 14 }}>
            <div style={{ fontSize: 11, color: 'var(--text3)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Strategy Q-values
            </div>
            {sorted.length > 0 ? sorted.map(([strat, data]) => {
              const isBest = strat === bestStrategy
              const color = STRATEGY_COLOR[strat] || '#8b90a8'
              return (
                <div key={strat} style={{ marginBottom: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3, alignItems: 'center' }}>
                    <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                      <span style={{
                        width: 8, height: 8, borderRadius: 2,
                        background: color, display: 'inline-block', flexShrink: 0,
                      }} />
                      <span style={{ fontSize: 11, color: isBest ? 'var(--text)' : 'var(--text2)', fontWeight: isBest ? 500 : 400 }}>
                        {STRATEGY_LABEL[strat] || strat}
                        {isBest && <span style={{ marginLeft: 5, fontSize: 10, color: '#2dd4a4' }}>★ best</span>}
                      </span>
                    </div>
                    <div style={{ display: 'flex', gap: 8, fontSize: 11 }}>
                      <span style={{ color, fontWeight: 600 }}>{data.q.toFixed(3)}</span>
                      <span style={{ color: 'var(--text3)' }}>{data.visits}v</span>
                    </div>
                  </div>
                  <div style={{ height: 3, background: 'var(--bg3)', borderRadius: 2, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${data.q * 100}%`, background: color, borderRadius: 2 }} />
                  </div>
                </div>
              )
            }) : (
              <div style={{ fontSize: 11, color: 'var(--text3)' }}>No memory yet for this topic.</div>
            )}
          </div>

          {/* Debug row */}
          <div style={{
            marginTop: 14, display: 'flex', gap: 6, flexWrap: 'wrap',
          }}>
            {[
              { k: 'exploring', v: debug.was_exploring ? 'yes' : 'no', c: debug.was_exploring ? '#fbbf24' : '#2dd4a4' },
              { k: 'code_quality', v: debug.code_quality?.toFixed(3), c: '#60a5fa' },
              { k: 'visit_count', v: debug.visit_count, c: 'var(--text2)' },
              { k: 'q_after', v: qValueAfter, c: '#a78bfa' },
            ].map(({ k, v, c }) => (
              <div key={k} style={{
                fontSize: 10, background: 'var(--bg3)',
                border: '1px solid var(--border)', borderRadius: 6,
                padding: '2px 8px', display: 'flex', gap: 4,
              }}>
                <span style={{ color: 'var(--text3)' }}>{k}</span>
                <span style={{ color: c, fontWeight: 500 }}>{v}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default function QueryPage() {
  const [query, setQuery] = useState('')
  const [pendingStrategy, setPendingStrategy] = useState(null)
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
    setPendingStrategy(null)
    try {
      const res = await api.query(text)
      setResult(res)
      setPendingStrategy(res.retrieval_strategy)
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

  const strategyColor = result ? (STRATEGY_COLOR[result.retrieval_strategy] || '#8b90a8') : 'var(--teal)'

  return (
    <div style={{ maxWidth: 820, margin: '0 auto', padding: '32px 24px' }}>

      {/* Search bar */}
      <div style={{ marginBottom: 20 }}>
        <div style={{
          display: 'flex', gap: 10,
          background: 'var(--bg2)',
          border: `1px solid ${loading ? 'var(--border2)' : 'var(--border2)'}`,
          borderRadius: 14, padding: '6px 6px 6px 16px',
        }}>
          <Search size={15} color="var(--text3)" style={{ marginTop: 12, flexShrink: 0 }} />
          <textarea
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() } }}
            placeholder="Ask a research question about AI…"
            rows={2}
            style={{
              flex: 1, background: 'transparent', border: 'none', outline: 'none',
              color: 'var(--text)', fontSize: 14, resize: 'none',
              paddingTop: 8, lineHeight: 1.6,
            }}
          />
          <button
            onClick={() => submit()}
            disabled={loading || !query.trim()}
            style={{
              background: loading || !query.trim() ? 'var(--bg3)' : 'var(--teal)',
              color: loading || !query.trim() ? 'var(--text3)' : '#000',
              border: 'none', borderRadius: 10, padding: '0 22px',
              fontWeight: 600, fontSize: 13,
              display: 'flex', alignItems: 'center', gap: 6,
              alignSelf: 'stretch', transition: 'all 0.15s',
              whiteSpace: 'nowrap',
            }}
          >
            {loading
              ? <><span style={{ animation: 'spin 1s linear infinite', display: 'inline-block' }}>⟳</span> Thinking</>
              : <><Zap size={13} /> Ask</>
            }
          </button>
        </div>

        {/* Suggestion chips */}
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 10 }}>
          {EXAMPLE_QUERIES.slice(0, 4).map(q => (
            <button
              key={q}
              onClick={() => submit(q)}
              disabled={loading}
              style={{
                background: 'var(--bg3)', border: '1px solid var(--border)',
                borderRadius: 20, padding: '4px 12px',
                fontSize: 11, color: 'var(--text2)',
                transition: 'border-color 0.15s',
              }}
            >
              {q.length > 52 ? q.slice(0, 52) + '…' : q}
            </button>
          ))}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{
          background: 'var(--red-dim)', border: '1px solid var(--red)',
          borderRadius: 10, padding: '14px 16px', color: 'var(--red)', marginBottom: 20,
          fontSize: 13,
        }}>
          <strong>Error:</strong> {error}
          {error.includes('providers') && (
            <div style={{ marginTop: 6, fontSize: 12, color: 'var(--text2)' }}>
              Groq hit its per-minute rate limit. Wait ~60 seconds and try again.
            </div>
          )}
        </div>
      )}

      {/* Animated thinking state */}
      {loading && <ThinkingPanel strategy={pendingStrategy} />}

      {/* Result */}
      {result && !loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

          {/* Metric cards row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
            <MetricCard
              label="Confidence"
              value={`${(result.confidence * 100).toFixed(0)}%`}
              color={result.confidence > 0.7 ? '#2dd4a4' : result.confidence > 0.4 ? '#fbbf24' : '#f87171'}
              sub={result.confidence > 0.7 ? 'High' : result.confidence > 0.4 ? 'Medium' : 'Low'}
            />
            <MetricCard
              label="Hallucination"
              value={`${(result.hallucination_score * 100).toFixed(0)}%`}
              color={result.hallucination_score < 0.2 ? '#2dd4a4' : result.hallucination_score < 0.4 ? '#fbbf24' : '#f87171'}
              sub={result.hallucination_score < 0.2 ? 'Low' : result.hallucination_score < 0.4 ? 'Medium' : 'High'}
              inverse
            />
            <MetricCard
              label="Strategy"
              value={STRATEGY_LABEL[result.retrieval_strategy] || result.retrieval_strategy}
              color={STRATEGY_COLOR[result.retrieval_strategy] || '#8b90a8'}
              sub={`Session #${result.session_number}`}
            />
            <MetricCard
              label="Q-value"
              value={result.q_value_after}
              color="#a78bfa"
              sub={`${result.papers_retrieved} papers · ${result.duration_ms}ms`}
            />
          </div>

          {/* Intent + quality bars */}
          <div style={{
            background: 'var(--bg2)', border: '1px solid var(--border)',
            borderRadius: 12, padding: '14px 18px',
            display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 32px',
          }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 12 }}>
                <Brain size={12} color="#a78bfa" />
                <span style={{ fontSize: 11, color: 'var(--text3)' }}>Detected intent</span>
                <span style={{ fontSize: 12, color: '#a78bfa', fontWeight: 500, marginLeft: 2 }}>
                  {result.intent}
                </span>
              </div>
              <MetricBar label="Confidence" value={result.confidence} />
              <MetricBar label="Answer quality" value={result.outcome_quality || result.confidence * (1 - result.hallucination_score)} />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 12 }}>
                <TrendingUp size={12} color={strategyColor} />
                <span style={{ fontSize: 11, color: 'var(--text3)' }}>Memory bucket</span>
                <span style={{ fontSize: 11, color: strategyColor, fontWeight: 500, marginLeft: 2 }}>
                  {result.memrl_debug?.intent_bucket || '—'}
                </span>
              </div>
              <MetricBar label="Retrieval precision" value={result.relevance_score || 0} />
              <MetricBar label="Hallucination" value={result.hallucination_score} inverse />
            </div>
          </div>

          {/* MemRL panel */}
          <MemRLPanel debug={result.memrl_debug} qValueAfter={result.q_value_after} />

          {/* Answer */}
          <div style={{
            background: 'var(--bg2)', border: '1px solid var(--border)',
            borderRadius: 12, padding: '20px 22px',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 14 }}>
              <BookOpen size={13} color="var(--teal)" />
              <span style={{ fontSize: 12, fontWeight: 500, color: 'var(--text2)' }}>Answer</span>
              {result.rewrite_count > 0 && (
                <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--amber)' }}>
                  {result.rewrite_count} query rewrite{result.rewrite_count > 1 ? 's' : ''}
                </span>
              )}
            </div>
            <div style={{ fontSize: 13.5, color: 'var(--text)', lineHeight: 1.85, whiteSpace: 'pre-wrap' }}>
              {result.answer}
            </div>

            {/* Feedback */}
            <div style={{
              display: 'flex', justifyContent: 'flex-end', gap: 8,
              marginTop: 18, paddingTop: 14, borderTop: '1px solid var(--border)',
              alignItems: 'center',
            }}>
              <span style={{ fontSize: 12, color: 'var(--text3)', marginRight: 2 }}>Was this helpful?</span>
              <button onClick={() => sendFeedback(1.0)} disabled={feedbackSent} style={{
                background: feedbackSent ? 'var(--teal-dim)' : 'var(--bg3)',
                border: `1px solid ${feedbackSent ? 'var(--teal)' : 'var(--border)'}`,
                borderRadius: 7, padding: '4px 12px',
                color: feedbackSent ? 'var(--teal)' : 'var(--text2)',
                display: 'flex', alignItems: 'center', gap: 4, fontSize: 12,
              }}>
                <ThumbsUp size={12} /> Yes
              </button>
              <button onClick={() => sendFeedback(0.0)} disabled={feedbackSent} style={{
                background: 'var(--bg3)', border: '1px solid var(--border)',
                borderRadius: 7, padding: '4px 12px',
                color: 'var(--text2)', display: 'flex', alignItems: 'center', gap: 4, fontSize: 12,
              }}>
                <ThumbsDown size={12} /> No
              </button>
            </div>
          </div>

          {/* Sources */}
          {result.sources?.length > 0 && (
            <div>
              <button onClick={() => setShowSources(!showSources)} style={{
                background: 'none', border: 'none',
                color: 'var(--text2)', fontSize: 13,
                display: 'flex', alignItems: 'center', gap: 6, padding: '4px 0',
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
                      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
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
        <div style={{ textAlign: 'center', padding: '64px 0', color: 'var(--text3)' }}>
          <Target size={38} style={{ margin: '0 auto 18px', opacity: 0.3 }} />
          <div style={{ fontSize: 16, fontWeight: 500, marginBottom: 10, color: 'var(--text2)' }}>
            Ask a research question
          </div>
          <div style={{ fontSize: 13, maxWidth: 420, margin: '0 auto', lineHeight: 1.8 }}>
            GhostMind fetches papers from arXiv, retrieves the most relevant ones using its learned strategy,
            generates a grounded answer, and updates its memory to improve next time.
          </div>
        </div>
      )}

      <style>{`@keyframes spin { to{transform:rotate(360deg)} }`}</style>
    </div>
  )
}

function MetricCard({ label, value, color, sub, inverse }) {
  return (
    <div style={{
      background: 'var(--bg2)', border: '1px solid var(--border)',
      borderRadius: 10, padding: '12px 14px',
    }}>
      <div style={{ fontSize: 10, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ fontSize: 20, fontWeight: 600, color, lineHeight: 1, marginBottom: 4 }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: 10, color: 'var(--text3)' }}>{sub}</div>}
    </div>
  )
}