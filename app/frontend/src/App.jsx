import React, { useState, useEffect } from 'react'
import IdentifyPage from './pages/IdentifyPage'
import VerifyPage from './pages/VerifyPage'
import MetricsPage from './pages/MetricsPage'
import StatusPage from './pages/StatusPage'
import { getStatus } from './services/api'

const TRADITIONS = ['Ikat','Jamdani','Banarasi','Patola','Kanjivaram','Chanderi']

const TABS = [
  { id: 'identify', label: '🔍 Identify',  icon: '🔍' },
  { id: 'verify',   label: '⚖️ Verify',    icon: '⚖️' },
  { id: 'metrics',  label: '📈 Metrics',   icon: '📈' },
  { id: 'status',   label: '⚙️ Status',    icon: '⚙️' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('identify')
  const [apiOk, setApiOk] = useState(null)

  useEffect(() => {
    getStatus()
      .then(() => setApiOk(true))
      .catch(() => setApiOk(false))
  }, [])

  function render() {
    switch (activeTab) {
      case 'identify': return <IdentifyPage />
      case 'verify':   return <VerifyPage />
      case 'metrics':  return <MetricsPage />
      case 'status':   return <StatusPage />
      default:         return <IdentifyPage />
    }
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <a href="/" className="header-logo">
          <div className="logo-dot" />
          DeepLure · Handloom ID
        </a>
        <nav className="header-nav">
          {TABS.map(t => (
            <button
              key={t.id}
              id={`tab-${t.id}`}
              className={`nav-btn${activeTab === t.id ? ' active' : ''}`}
              onClick={() => setActiveTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      {/* Hero */}
      <section className="hero">
        <div className="hero-orb hero-orb-1" />
        <div className="hero-orb hero-orb-2" />
        <div className="hero-badge">🧵 Handloom Saree · Color-Invariant Recognition</div>
        <h1>Identify Handloom Designs<br />Across Any Colour Palette</h1>
        <p>
          Metric-learning system for Indian handloom weaves — Ikat, Jamdani,
          Banarasi, Patola, Kanjivaram &amp; Chanderi. Same motif, any palette.
        </p>
        <div style={{ display:'flex', gap:'.5rem', justifyContent:'center', flexWrap:'wrap', marginBottom:'1.5rem' }}>
          {TRADITIONS.map(t => (
            <span key={t} style={{
              background:'var(--clr-glass)', border:'1px solid var(--clr-border2)',
              borderRadius:'999px', padding:'.2rem .7rem',
              fontSize:'.72rem', fontWeight:600, color:'var(--clr-accent2)',
              letterSpacing:'.04em'
            }}>{t}</span>
          ))}
        </div>
        <div className="hero-status">
          <div className={`status-dot${apiOk === false ? ' error' : ''}`} />
          {apiOk === null && 'Connecting to API…'}
          {apiOk === true && 'API connected · Backend ready'}
          {apiOk === false && 'API offline — start backend (uvicorn main:app)'}
        </div>
      </section>

      {/* Tab content */}
      <main className="main">
        <div className="tabs">
          {TABS.map(t => (
            <button
              key={t.id}
              className={`tab${activeTab === t.id ? ' active' : ''}`}
              onClick={() => setActiveTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
        {render()}
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>
          DeepLure · Color-Invariant Saree Design Recognition ·{' '}
          <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer">API Docs</a>
          {' · '}
          <a href="https://deeplure.org" target="_blank" rel="noreferrer">deeplure.org</a>
        </p>
      </footer>
    </div>
  )
}
