import React, { useState, useCallback } from 'react'
import { identify } from '../services/api'
import { useAsync, useImageUpload } from '../hooks/useUpload'
import { UploadZone, SimBar, Spinner, Alert } from '../components/UI'

export default function IdentifyPage() {
  const [file, setFile] = useState(null)
  const [topK, setTopK] = useState(5)
  const { loading, data, error, run } = useAsync(identify)

  async function handleSubmit() {
    if (!file) return
    await run(file, topK)
  }

  return (
    <div className="fade-in">
      <div className="grid-2" style={{ marginBottom: '1.5rem' }}>
        <div className="card">
          <div className="card-title"><span className="icon">🔍</span> Query Image</div>
          <UploadZone id="identify-upload" onFile={setFile} label="Drop query saree image" />
          <div style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <label style={{ fontSize: '.85rem', color: 'var(--clr-text2)', whiteSpace: 'nowrap' }}>
              Top-K results
            </label>
            <input
              type="range" min="1" max="20" value={topK}
              onChange={e => setTopK(Number(e.target.value))}
              style={{ flex: 1 }}
              id="topk-slider"
            />
            <span style={{
              fontFamily: 'var(--font-heading)', fontWeight: 700,
              color: 'var(--clr-accent)', minWidth: '2ch', textAlign: 'right'
            }}>{topK}</span>
          </div>
          <button
            id="identify-btn"
            className="btn btn-primary btn-full"
            style={{ marginTop: '1rem' }}
            onClick={handleSubmit}
            disabled={!file || loading}
          >
            {loading ? <><Spinner /> Matching…</> : '🔍 Find Matches'}
          </button>
          {error && <Alert type="error">{error}</Alert>}
        </div>

        <div className="card">
          <div className="card-title"><span className="icon">📊</span> Ranked Matches</div>
          {!data && !loading && (
            <div style={{ textAlign: 'center', color: 'var(--clr-text3)', padding: '3rem 1rem' }}>
              <div style={{ fontSize: '2.5rem', marginBottom: '.5rem' }}>🎯</div>
              <p>Upload a saree image and click <strong>Find Matches</strong></p>
            </div>
          )}
          {loading && (
            <div style={{ textAlign: 'center', color: 'var(--clr-text2)', padding: '3rem 1rem' }}>
              <Spinner /> <span style={{ marginLeft: '.5rem' }}>Computing embeddings…</span>
            </div>
          )}
          {data && (
            <>
              <p style={{ fontSize: '.78rem', color: 'var(--clr-text3)', marginBottom: '1rem' }}>
                Mode: <strong style={{ color: 'var(--clr-accent2)' }}>{data.mode}</strong> ·{' '}
                Top-{data.top_k} results {data.gallery_total ? `(gallery size: ${data.gallery_total} images)` : ''}
              </p>
              <div className="result-list">
                {(data.results || data.matches || []).map(m => (
                  <div className="result-item fade-in" key={m.rank} style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                    <div className="result-rank">#{m.rank}</div>
                    {m.image_url && (
                      <img
                        src={m.image_url}
                        alt={m.image}
                        style={{ width: '54px', height: '54px', objectFit: 'cover', borderRadius: '6px', border: '1px solid var(--clr-border)', background: 'var(--clr-surface2)' }}
                        onError={e => { e.target.style.display = 'none' }}
                      />
                    )}
                    <div className="result-info" style={{ flex: 1 }}>
                      <div className="result-name">{m.image || m.filename}</div>
                      {m.class && <div style={{ fontSize:'.72rem', color:'var(--clr-accent2)', marginBottom:'.2rem' }}>{m.class.toUpperCase()}</div>}
                      <div className="result-sim">Cosine similarity: {(m.similarity * 100).toFixed(1)}%</div>
                      <SimBar value={m.similarity} />
                    </div>
                  </div>
                ))}
              </div>
              {(data.results || data.matches || []).length === 0 && (
                <Alert type="info">Gallery is empty — run: python scripts/generate_embeddings.py</Alert>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
