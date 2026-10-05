import React, { useState } from 'react'
import { verify } from '../services/api'
import { useAsync } from '../hooks/useUpload'
import { UploadZone, Spinner, Alert, SimBar } from '../components/UI'

export default function VerifyPage() {
  const [fileA, setFileA] = useState(null)
  const [fileB, setFileB] = useState(null)
  const { loading, data, error, run } = useAsync(verify)

  async function handleSubmit() {
    if (!fileA || !fileB) return
    await run(fileA, fileB)
  }

  const isMatch = data?.same_design

  return (
    <div className="fade-in">
      <div className="grid-2" style={{ marginBottom: '1.5rem' }}>
        <div className="card">
          <div className="card-title"><span className="icon">🅰️</span> Image A</div>
          <UploadZone id="verify-upload-a" onFile={setFileA} label="Drop first saree image" />
        </div>
        <div className="card">
          <div className="card-title"><span className="icon">🅱️</span> Image B</div>
          <UploadZone id="verify-upload-b" onFile={setFileB} label="Drop second saree image" />
        </div>
      </div>

      <div className="card">
        <div className="card-title"><span className="icon">⚖️</span> Verification Result</div>

        <button
          id="verify-btn"
          className="btn btn-primary btn-full"
          onClick={handleSubmit}
          disabled={!fileA || !fileB || loading}
          style={{ marginBottom: '1.25rem' }}
        >
          {loading ? <><Spinner /> Comparing…</> : '⚖️ Compare Designs'}
        </button>

        {error && <Alert type="error">{error}</Alert>}

        {data && (
          <div className="fade-in">
            <div className={`verify-badge ${isMatch ? 'match' : 'no-match'}`}>
              <span className="badge-icon">{isMatch ? '✅' : '❌'}</span>
              <div>
                <div>{isMatch ? 'Same Design' : 'Different Designs'}</div>
                <div className="badge-sim">
                  Cosine similarity: {(data.similarity * 100).toFixed(2)}%
                  {' · '}Threshold: {(data.threshold * 100).toFixed(1)}%
                  {data.confidence !== undefined && ` · Confidence: ${(data.confidence * 100).toFixed(1)}%`}
                </div>
              </div>
            </div>
            <div style={{ marginTop: '1rem' }}>
              <p style={{ fontSize: '.8rem', color: 'var(--clr-text3)', marginBottom: '.4rem' }}>
                Similarity score
              </p>
              <SimBar value={data.similarity} />
            </div>
            <p style={{ fontSize: '.78rem', color: 'var(--clr-text3)', marginTop: '.75rem' }}>
              Mode: <strong style={{ color: 'var(--clr-accent2)' }}>{data.mode}</strong>
            </p>
          </div>
        )}

        {!data && !loading && (
          <div style={{ textAlign: 'center', color: 'var(--clr-text3)', padding: '2rem 1rem' }}>
            <div style={{ fontSize: '2.5rem', marginBottom: '.5rem' }}>⚖️</div>
            <p>Upload two images and click <strong>Compare Designs</strong></p>
          </div>
        )}
      </div>
    </div>
  )
}
