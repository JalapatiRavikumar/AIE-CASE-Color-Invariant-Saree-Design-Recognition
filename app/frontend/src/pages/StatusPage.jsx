import React, { useEffect } from 'react'
import { getStatus, getDownloads, downloadUrl } from '../services/api'
import { useAsync } from '../hooks/useUpload'
import { Spinner, Alert } from '../components/UI'

function StatusRow({ label, value, ok }) {
  return (
    <div className="status-row">
      <span className="status-key">{label}</span>
      <span className={`status-val${ok === true ? ' ok' : ok === 'demo' ? ' demo' : ''}`}>{value ?? '—'}</span>
    </div>
  )
}

function fmt(n) {
  if (n === null || n === undefined) return '—'
  if (typeof n === 'number') return n.toLocaleString()
  return String(n)
}

function fmtSize(bytes) {
  if (!bytes) return '0 B'
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(2) + ' MB'
}

export default function StatusPage() {
  const status = useAsync(getStatus)
  const downloads = useAsync(getDownloads)

  useEffect(() => {
    status.run()
    downloads.run()
  }, [])

  const d = status.data
  const dlFiles = downloads.data?.files || []

  return (
    <div className="fade-in">
      <div className="grid-2">
        {/* System Status */}
        <div className="card">
          <div className="card-title" style={{ justifyContent: 'space-between' }}>
            <span><span className="icon">⚙️</span> System Status</span>
            <button id="refresh-status-btn" className="btn btn-secondary" onClick={status.run} disabled={status.loading}>
              {status.loading ? <Spinner /> : '🔄'}
            </button>
          </div>
          {status.error && <Alert type="error">{status.error}</Alert>}
          {d && (
            <>
              <StatusRow label="Mode" value={d.mode} ok={d.mode === 'trained' ? true : 'demo'} />
              <StatusRow label="Device" value={d.device} />
              <StatusRow label="Backbone" value={d.backbone} />
              <StatusRow label="Embedding Dim" value={fmt(d.embedding_dim)} />
              <StatusRow label="Checkpoint" value={d.checkpoint ? '✓ Loaded' : 'None (demo)'} ok={!!d.checkpoint} />
              <StatusRow label="Gallery Images" value={fmt(d.gallery_count)} ok={d.gallery_count > 0} />
              <StatusRow label="FAISS Index" value={d.faiss ? '✓ Active (FAISS)' : 'FAISS: unavailable · Fallback: NumPy'} ok={d.faiss} />
            </>
          )}
          {!d && !status.loading && (
            <Alert type="error">Cannot reach backend at localhost:8000</Alert>
          )}
        </div>

        {/* Model Meta */}
        <div className="card">
          <div className="card-title"><span className="icon">🧠</span> Model Info</div>
          {d?.meta ? (
            Object.entries(d.meta).map(([k, v]) => (
              <StatusRow key={k} label={k} value={typeof v === 'object' ? JSON.stringify(v) : String(v)} />
            ))
          ) : (
            <p style={{ color: 'var(--clr-text3)', fontSize: '.875rem' }}>No checkpoint metadata available.</p>
          )}
        </div>
      </div>

      {/* Downloadable Artifacts */}
      <div className="card" style={{ marginTop: '1.5rem' }}>
        <div className="card-title" style={{ justifyContent: 'space-between' }}>
          <span><span className="icon">📦</span> Downloadable Artifacts & Models</span>
          <button className="btn btn-secondary" onClick={downloads.run} disabled={downloads.loading}>
            {downloads.loading ? <Spinner /> : '🔄 Refresh'}
          </button>
        </div>
        {downloads.error && <Alert type="error">{downloads.error}</Alert>}
        {dlFiles.length === 0 && !downloads.loading ? (
          <p style={{ color: 'var(--clr-text3)', fontSize: '.875rem' }}>No downloadable artifacts found in /downloads.</p>
        ) : (
          <div style={{ marginTop: '1rem' }}>
            {dlFiles.map(file => (
              <div key={file.name} className="download-item">
                <div className="download-left">
                  <span className="download-icon">
                    {file.name.endsWith('.zip') ? '🗜️' : file.name.endsWith('.json') ? '📊' : '📄'}
                  </span>
                  <div>
                    <div className="download-name">{file.name}</div>
                    <div className="download-size">{fmtSize(file.size_bytes)}</div>
                  </div>
                </div>
                <a
                  href={downloadUrl(file.name)}
                  download={file.name}
                  className="btn btn-primary download-btn"
                  target="_blank"
                  rel="noreferrer"
                >
                  ⬇️ Download
                </a>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

