import { useState, useEffect } from 'react'
import { api } from '../utils/api.js'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, AreaChart, Area
} from 'recharts'
import { TrendingUp, TrendingDown, Loader2 } from 'lucide-react'

export default function DashboardPage() {
  const [benchmarks, setBenchmarks] = useState([])
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([api.benchmarks(), api.stats()])
      .then(([b, s]) => { setBenchmarks(b); setStats(s) })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <Spinner />

  const chartData = benchmarks.map(b => ({
    session: `S${b.session_number}`,
    quality: +(b.answer_quality * 100).toFixed(1),
    confidence: +(b.avg_confidence * 100).toFixed(1),
    hallucination: +(b.avg_hallucination * 100).toFixed(1),
    precision: +(b.retrieval_precision * 100).toFixed(1),
  }))

  return (
    <div style={{ maxWidth: 960, margin: '0 auto', padding: '32px 24px' }}>
      <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 4 }}>Benchmark Dashboard</h2>
      <p style={{ color: 'var(--text3)', fontSize: 13, marginBottom: 28 }}>
        Performance metrics across all research sessions. The system improves as it learns from experience.
      </p>

      {/* Stats cards */}
      {stats && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12, marginBottom: 32 }}>
          <StatCard label="Total Sessions" value={stats.total_sessions} />
          <StatCard label="Memory Triplets" value={stats.total_triplets} color="var(--purple)" />
          <StatCard label="Failure Logs" value={stats.total_failures} color="var(--red)" />
          <StatCard label="Avg Confidence" value={`${(stats.avg_confidence * 100).toFixed(0)}%`} color="var(--teal)" />
          <StatCard label="Avg Hallucination" value={`${(stats.avg_hallucination * 100).toFixed(0)}%`} color="var(--amber)" />
        </div>
      )}

      {chartData.length === 0 ? (
        <EmptyState message="Run some queries to see benchmark data here." />
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          {/* Answer quality + hallucination */}
          <ChartCard title="Answer Quality & Hallucination Rate" subtitle="How the system improves over sessions">
            <ResponsiveContainer width="100%" height={240}>
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="qualGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#2dd4a4" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#2dd4a4" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="hallGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f87171" stopOpacity={0.15} />
                    <stop offset="95%" stopColor="#f87171" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="session" tick={{ fill: 'var(--text3)', fontSize: 11 }} />
                <YAxis domain={[0, 100]} tick={{ fill: 'var(--text3)', fontSize: 11 }} tickFormatter={v => `${v}%`} />
                <Tooltip contentStyle={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 8 }}
                  labelStyle={{ color: 'var(--text2)' }} formatter={v => `${v}%`} />
                <Legend wrapperStyle={{ fontSize: 12, color: 'var(--text2)' }} />
                <Area type="monotone" dataKey="quality" name="Answer Quality" stroke="#2dd4a4" fill="url(#qualGrad)" strokeWidth={2} dot={{ r: 3 }} />
                <Area type="monotone" dataKey="hallucination" name="Hallucination %" stroke="#f87171" fill="url(#hallGrad)" strokeWidth={2} dot={{ r: 3 }} />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>

          {/* Confidence + retrieval precision */}
          <ChartCard title="Confidence & Retrieval Precision" subtitle="Self-evaluation scores over time">
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="session" tick={{ fill: 'var(--text3)', fontSize: 11 }} />
                <YAxis domain={[0, 100]} tick={{ fill: 'var(--text3)', fontSize: 11 }} tickFormatter={v => `${v}%`} />
                <Tooltip contentStyle={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 8 }}
                  labelStyle={{ color: 'var(--text2)' }} formatter={v => `${v}%`} />
                <Legend wrapperStyle={{ fontSize: 12, color: 'var(--text2)' }} />
                <Line type="monotone" dataKey="confidence" name="Confidence" stroke="#60a5fa" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="precision" name="Retrieval Precision" stroke="#a78bfa" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>
      )}
    </div>
  )
}

function StatCard({ label, value, color = 'var(--text)' }) {
  return (
    <div style={{
      background: 'var(--bg2)', border: '1px solid var(--border)',
      borderRadius: 12, padding: '16px 18px',
    }}>
      <div style={{ fontSize: 22, fontWeight: 600, color }}>{value}</div>
      <div style={{ fontSize: 11, color: 'var(--text3)', marginTop: 4 }}>{label}</div>
    </div>
  )
}

function ChartCard({ title, subtitle, children }) {
  return (
    <div style={{
      background: 'var(--bg2)', border: '1px solid var(--border)',
      borderRadius: 12, padding: '20px 22px',
    }}>
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 14, fontWeight: 500 }}>{title}</div>
        {subtitle && <div style={{ fontSize: 12, color: 'var(--text3)', marginTop: 2 }}>{subtitle}</div>}
      </div>
      {children}
    </div>
  )
}

function EmptyState({ message }) {
  return (
    <div style={{ textAlign: 'center', padding: '60px 0', color: 'var(--text3)' }}>
      <TrendingUp size={32} style={{ margin: '0 auto 12px', opacity: 0.3 }} />
      <div style={{ fontSize: 14 }}>{message}</div>
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
