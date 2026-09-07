import { useState, useEffect } from 'react'
import { api } from '../utils/api.js'
import { Database, ChevronDown, ChevronUp, Loader2, ExternalLink } from 'lucide-react'

const strategyColor = {
  semantic: 'var(--blue)',
  graph: 'var(--purple)',
  hybrid: 'var(--teal)',
  aggressive_rewrite: 'var(--amber)',
}

const COLS = '36px 1fr 92px 64px 64px 70px 56px 76px 28px'

export default function SessionsPage() {
  const [sessions, setSessions] = useState([])
  const [selected, setSelected] = useState(null)
  const [detail, setDetail] = useState(null)
  const [loading, setLoading] = useState(true)
  const [detailLoading, setDetailLoading] = useState(false)

  useEffect(() => {
    api.sessions(50).then(setSessions).catch(() => {}).finally(() => setLoading(false))
  }, [])

  async function loadDetail(id) {
    if (selected === id) { setSelected(null); setDetail(null); return }
    setSelected(id); setDetailLoading(true)
    try { setDetail(await api.session(id)) } catch { setDetail(null) }
    finally { setDetailLoading(false) }
  }

  if (loading) return <Spinner />

  return (
    <div style={{ maxWidth: 1080, margin: '0 auto', padding: '32px 24px' }}>
      <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 4 }}>Research Sessions</h2>
      <p style={{ color: 'var(--text3)', fontSize: 13, marginBottom: 28 }}>
        Measured faithfulness, verified citations and real retrieval scores — no flat metrics.
      </p>

      {sessions.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '60px 0', color: 'var(--text3)' }}>
          <Database size={32} style={{ margin: '0 auto 12px', opacity: 0.3 }} />
          <div>No sessions yet. Run a query to get started.</div>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ display: 'grid', gridTemplateColumns: COLS, gap: 8,
                        padding: '6px 14px', fontSize: 11, color: 'var(--text3)' }}>
            <span>#</span><span>Query</span><span>Strategy</span>
            <span>Quality</span><span>Conf.</span><span>Halluc.</span>
            <span>Sources</span><span>Duration</span><span></span>
          </div>

          {sessions.map(s => (
            <div key={s.id}>
              <div onClick={() => loadDetail(s.id)}
                style={{ display: 'grid', gridTemplateColumns: COLS, gap: 8, padding: '10px 14px',
                  background: selected === s.id ? 'var(--bg3)' : 'var(--bg2)',
                  border: `1px solid ${selected === s.id ? 'var(--border2)' : 'var(--border)'}`,
                  borderRadius: selected === s.id ? '10px 10px 0 0' : 10,
                  cursor: 'pointer', alignItems: 'center', fontSize: 13, transition: 'background 0.15s' }}>
                <span style={{ color: 'var(--text3)', fontSize: 12 }}>{s.session_number}</span>
                <span style={{ color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {s.query}
                </span>
                <span style={{ color: strategyColor[s.retrieval_strategy] || 'var(--text2)', fontSize: 11, fontWeight: 500 }}>
                  {s.retrieval_strategy}
                </span>
                <Meter value={s.outcome_quality} good />
                <Meter value={s.confidence} good />
                <Meter value={s.hallucination_score} good={false} />
                <span style={{ color: 'var(--text2)', fontSize: 12, textAlign: 'center' }}>
                  {s.sources_count ?? 0}
                </span>
                <span style={{ color: 'var(--text3)', fontSize: 12 }}>{s.duration_ms}ms</span>
                {selected === s.id ? <ChevronUp size={14} color="var(--text3)" /> : <ChevronDown size={14} color="var(--text3)" />}
              </div>

              {selected === s.id && (
                <div style={{ background: 'var(--bg3)', border: '1px solid var(--border2)',
                              borderTop: 'none', borderRadius: '0 0 10px 10px', padding: '16px 18px' }}>
                  {detailLoading ? <Spinner small /> : detail ? (
                    <div>
                      <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap', fontSize: 11,
                                    color: 'var(--text3)', marginBottom: 10 }}>
                        <span>Intent: <b style={{ color: 'var(--purple)' }}>{detail.intent}</b></span>
                        <span>Faithfulness: <b style={{ color: 'var(--teal)' }}>
                          {((detail.eval_breakdown?.grounded_claims ?? 0))}/{(detail.eval_breakdown?.claims ?? 0)} claims grounded</b></span>
                        <span>Citations verified: <b style={{ color: 'var(--blue)' }}>
                          {detail.eval_breakdown?.verified_citations ?? 0}/{detail.eval_breakdown?.citations ?? 0}</b></span>
                        <span>Judge: <b>{((detail.eval_breakdown?.judge_score ?? 0) * 100).toFixed(0)}%</b></span>
                        <span>Retrieval: <b>{((detail.relevance_score ?? 0) * 100).toFixed(0)}%</b></span>
                      </div>

                      {detail.eval_breakdown?.ungrounded_claims?.length > 0 && (
                        <div style={{ fontSize: 11, color: 'var(--amber)', marginBottom: 10 }}>
                          ⚠ Ungrounded: {detail.eval_breakdown.ungrounded_claims.slice(0, 2).join(' · ')}
                        </div>
                      )}

                      <div style={{ fontSize: 13, color: 'var(--text)', lineHeight: 1.7,
                                    whiteSpace: 'pre-wrap', marginBottom: 12 }}>
                        {detail.answer}
                      </div>

                      {detail.sources?.length > 0 && (
                        <div style={{ fontSize: 12, color: 'var(--text3)' }}>
                          <b>Sources:</b>
                          <ul style={{ margin: '6px 0 0', paddingLeft: 18 }}>
                            {detail.sources.map((src, i) => (
                              <li key={i} style={{ marginBottom: 3 }}>
                                <a href={src.url} target="_blank" rel="noreferrer"
                                   style={{ color: 'var(--blue)', textDecoration: 'none' }}>
                                  {src.title} <ExternalLink size={10} style={{ verticalAlign: 'middle' }} />
                                </a>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div style={{ color: 'var(--text3)', fontSize: 13 }}>Failed to load session details.</div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function Meter({ value, good }) {
  const v = value ?? 0
  const pct = (v * 100).toFixed(0)
  let color = 'var(--text2)'
  if (good) color = v > 0.7 ? 'var(--teal)' : v > 0.4 ? 'var(--amber)' : 'var(--red)'
  else color = v < 0.2 ? 'var(--teal)' : v < 0.4 ? 'var(--amber)' : 'var(--red)'
  return <span style={{ color, fontWeight: 500, fontSize: 12 }}>{pct}%</span>
}

function Spinner({ small }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'center', padding: small ? '20px 0' : '80px 0' }}>
      <Loader2 size={small ? 18 : 24} color="var(--teal)" style={{ animation: 'spin 1s linear infinite' }} />
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}
