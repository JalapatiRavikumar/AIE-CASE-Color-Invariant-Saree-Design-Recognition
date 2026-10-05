import React from 'react'
import { useImageUpload } from '../hooks/useUpload'

export function UploadZone({ onFile, label = 'Drop image here', id }) {
  const { file, preview, dragging, handleFile, onDrop, onDragOver, onDragLeave } = useImageUpload()

  function onChange(e) {
    const f = e.target.files[0]
    handleFile(f)
    if (f && onFile) onFile(f)
  }

  function handleDrop(e) {
    onDrop(e)
    const f = e.dataTransfer.files[0]
    if (f && onFile) onFile(f)
  }

  return (
    <div
      className={`upload-zone${dragging ? ' drag-over' : ''}`}
      onDrop={handleDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      id={id}
    >
      <input type="file" accept="image/*" onChange={onChange} />
      {preview ? (
        <>
          <img src={preview} alt="preview" className="upload-preview" />
          <p className="upload-hint" style={{ marginTop: '.5rem' }}>
            {file?.name} · {(file?.size / 1024).toFixed(0)} KB
          </p>
        </>
      ) : (
        <>
          <div className="upload-icon">🖼️</div>
          <p className="upload-text">{label}</p>
          <p className="upload-hint">PNG, JPG, WEBP · max 8 MB</p>
        </>
      )}
    </div>
  )
}

export function SimBar({ value }) {
  const pct = Math.max(0, Math.min(1, value)) * 100
  return (
    <div className="sim-bar">
      <div className="sim-fill" style={{ width: `${pct}%` }} />
    </div>
  )
}

export function Spinner() {
  return <span className="spinner" />
}

export function Alert({ type = 'error', children }) {
  return (
    <div className={`alert alert-${type}`}>
      <span>{type === 'error' ? '⚠️' : 'ℹ️'}</span>
      {children}
    </div>
  )
}

export function MetricCard({ value, label }) {
  let displayValue = value
  if (value === null || value === undefined) {
    displayValue = '—'
  } else if (typeof value === 'number') {
    const isThreshold = /threshold/i.test(label)
    const isRateOrPct = /recall|map|auc|eer|f1|accuracy|rejection/i.test(label)
    if (isThreshold) {
      displayValue = value.toFixed(3)
    } else if (isRateOrPct && value <= 1.0) {
      displayValue = (value * 100).toFixed(1) + '%'
    } else if (value < 1.0 && value > 0) {
      displayValue = (value * 100).toFixed(1) + '%'
    } else {
      displayValue = value.toLocaleString()
    }
  }
  return (
    <div className="metric-card">
      <div className="metric-value">{displayValue}</div>
      <div className="metric-label">{label}</div>
    </div>
  )
}
