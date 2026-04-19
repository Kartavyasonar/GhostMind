import { useState, useEffect } from 'react'
import { api } from '../utils/api.js'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { Brain, AlertCircle, GitBranch, Loader2 } from 'lucide-react'

export default function MemoryPage() {
  const [memory, setMemory] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.memory()
      .then(setMemory)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <Spinner />

  const strategies = memory?.strategy_summary || []
  const failures = memory?.recent_failures || []
  const graph = memory?.graph_stats || {}

  return (
    <div style={{ maxWidth: 960, margin: '0 auto', padding: '32px 24px' }}>
      <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 4 }}>Episodic Memory & MemRL</h2>
      <p style={{ color: 'var(--text3)', fontSize: 13, marginBottom: 28 }}>
        The agent stores intent → strategy → outcome triplets and uses Q-value updates to learn which retrieval strategies work best.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 24 }}>
        {/* Strategy Q-values */}
        <div style={{
          background: 'var(--bg2)', border: '1px solid var(--border)',
          borderRadius: 12, padding: '18px 20px',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
            <Brain size={15} color="var(--purple)" />
            <span style={{ fontSize: 14, fontWeight: 500 }}>Strategy Q-Values</span>
          </div>
          {strategies.length === 0 ? (
            <div style={{ color: 'var(--text3)', fontSize: 13, padding: '20px 0', textAlign: 'center' }}>
              No memory yet — run some queries first.
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={strategies} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="strategy" tick={{ fill: 'var(--text3)', fontSize: 11 }} />
                <YAxis domain={[0, 1]} tick={{ fill: 'var(--text3)', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 8 }}
                  formatter={(v, n) => [v.toFixed(3), n]}
                />
                <Bar dataKey="avg_q_value" name="Avg Q-Value" fill="#a78bfa" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}

          {/* Strategy table */}
          {strategies.length > 0 && (
            <div style={{ marginTop: 16 }}>
              {strategies.map(s => (
                <div key={s.strategy} style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  padding: '8px 0', borderBottom: '1px solid var(--border)', fontSize: 12,
                }}>
                  <span style={{ color: 'var(--text)', fontWeight: 500 }}>{s.strategy}</span>
                  <div style={{ display: 'flex', gap: 16, color: 'var(--text3)' }}>
                    <span>Q: <span style={{ color: 'var(--purple)' }}>{s.avg_q_value}</span></span>
                    <span>Visits: {s.total_visits}</span>
                    <span>Intents: {s.num_intents}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Graph stats */}
        <div style={{
          background: 'var(--bg2)', border: '1px solid var(--border)',
          borderRadius: 12, padding: '18px 20px',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
            <GitBranch size={15} color="var(--teal)" />
            <span style={{ fontSize: 14, fontWeight: 500 }}>Citation Graph</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            {[
              { label: 'Nodes (papers)', value: graph.nodes || 0, color: 'var(--teal)' },
              { label: 'Edges (citations)', value: graph.edges || 0, color: 'var(--blue)' },
              { label: 'Connected', value: graph.is_connected ? 'Yes' : 'No', color: graph.is_connected ? 'var(--teal)' : 'var(--text3)' },
              { label: 'Avg Degree', value: graph.avg_degree ? graph.avg_degree.toFixed(2) : '0.00', color: 'var(--amber)' },
            ].map(({ label, value, color }) => (
              <div key={label} style={{
                background: 'var(--bg3)', borderRadius: 8, padding: '12px 14px',
              }}>
                <div style={{ fontSize: 20, fontWeight: 600, color }}>{value}</div>
                <div style={{ fontSize: 11, color: 'var(--text3)', marginTop: 3 }}>{label}</div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 16, padding: '12px', background: 'var(--bg3)', borderRadius: 8, fontSize: 12, color: 'var(--text2)', lineHeight: 1.6 }}>
            The citation graph powers GraphRAG and hybrid retrieval strategies. Papers are nodes; citation edges are added as the agent learns relationships between papers.
          </div>
        </div>
      </div>

      {/* Failure log */}
      <div style={{
        background: 'var(--bg2)', border: '1px solid var(--border)',
        borderRadius: 12, padding: '18px 20px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
          <AlertCircle size={15} color="var(--red)" />
          <span style={{ fontSize: 14, fontWeight: 500 }}>Failure Log</span>
          <span style={{ fontSize: 11, color: 'var(--text3)', marginLeft: 4 }}>
            Where the agent got it wrong — and what it changed
          </span>
        </div>

        {failures.length === 0 ? (
          <div style={{ color: 'var(--text3)', fontSize: 13, padding: '20px 0', textAlign: 'center' }}>
            No failures recorded yet. This is a good sign! 🎉
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {failures.map(f => (
              <div key={f.id} style={{
                background: 'var(--bg3)', borderRadius: 8, padding: '12px 14px',
                border: '1px solid var(--border)',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <span style={{
                      fontSize: 10, padding: '2px 8px', borderRadius: 20, fontWeight: 500,
                      background: 'var(--red-dim)', color: 'var(--red)', marginRight: 8,
                    }}>
                      {f.failure_type}
                    </span>
                    <span style={{ fontSize: 12, color: 'var(--text2)' }}>{f.description}</span>
                  </div>
                  <span style={{
                    fontSize: 11, color: f.delta_q >= 0 ? 'var(--teal)' : 'var(--red)',
                    whiteSpace: 'nowrap', marginLeft: 12,
                  }}>
                    ΔQ: {f.delta_q >= 0 ? '+' : ''}{f.delta_q}
                  </span>
                </div>
                <div style={{ marginTop: 8, display: 'flex', gap: 8, fontSize: 11, color: 'var(--text3)', alignItems: 'center' }}>
                  <span style={{ background: 'var(--border)', padding: '2px 8px', borderRadius: 4 }}>{f.strategy_before}</span>
                  <span>→</span>
                  <span style={{ background: 'var(--teal-dim)', color: 'var(--teal)', padding: '2px 8px', borderRadius: 4 }}>{f.strategy_after}</span>
                  <span style={{ marginLeft: 'auto' }}>{f.created_at ? new Date(f.created_at).toLocaleString() : ''}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function Spinner() {
  return (
    <div style={{ display: 'flex', justifyContent: 'center', padding: '80px 0' }}>
      <Loader2 size={24} color="var(--teal)" style={{ animation: 'spin 1s linear infinite' }} />
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}
