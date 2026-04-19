import { useState } from 'react'
import QueryPage from './pages/QueryPage.jsx'
import DashboardPage from './pages/DashboardPage.jsx'
import MemoryPage from './pages/MemoryPage.jsx'
import SessionsPage from './pages/SessionsPage.jsx'
import { Brain, BarChart2, Database, Search, Zap } from 'lucide-react'

const tabs = [
  { id: 'query', label: 'Research', icon: Search },
  { id: 'dashboard', label: 'Benchmarks', icon: BarChart2 },
  { id: 'memory', label: 'Memory', icon: Brain },
  { id: 'sessions', label: 'Sessions', icon: Database },
]

export default function App() {
  const [tab, setTab] = useState('query')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      {/* Header */}
      <header style={{
        background: 'var(--bg2)',
        borderBottom: '1px solid var(--border)',
        padding: '0 24px',
        display: 'flex',
        alignItems: 'center',
        height: 56,
        gap: 12,
        position: 'sticky',
        top: 0,
        zIndex: 100,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginRight: 24 }}>
          <Zap size={18} color="var(--teal)" />
          <span style={{ fontWeight: 600, fontSize: 15, color: 'var(--text)' }}>GhostMind</span>
          <span style={{
            fontSize: 10, background: 'var(--teal-dim)', color: 'var(--teal)',
            padding: '2px 7px', borderRadius: 20, fontWeight: 500, letterSpacing: '0.05em'
          }}>BETA</span>
        </div>

        <nav style={{ display: 'flex', gap: 4 }}>
          {tabs.map(({ id, label, icon: Icon }) => (
            <button key={id} onClick={() => setTab(id)} style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '6px 14px', borderRadius: 8, border: 'none',
              background: tab === id ? 'var(--bg3)' : 'transparent',
              color: tab === id ? 'var(--text)' : 'var(--text2)',
              fontSize: 13, fontWeight: tab === id ? 500 : 400,
              transition: 'all 0.15s',
            }}>
              <Icon size={14} />
              {label}
            </button>
          ))}
        </nav>

        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 12, color: 'var(--text3)' }}>Self-evolving research agent</span>
        </div>
      </header>

      {/* Main content */}
      <main style={{ flex: 1, overflow: 'auto' }}>
        {tab === 'query' && <QueryPage />}
        {tab === 'dashboard' && <DashboardPage />}
        {tab === 'memory' && <MemoryPage />}
        {tab === 'sessions' && <SessionsPage />}
      </main>
    </div>
  )
}
