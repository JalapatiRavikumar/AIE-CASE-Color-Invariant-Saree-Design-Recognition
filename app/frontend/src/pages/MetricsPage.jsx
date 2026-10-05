import React, { useEffect } from 'react'
import { getMetrics } from '../services/api'
import { useAsync } from '../hooks/useUpload'
import { Spinner, Alert, MetricCard } from '../components/UI'

function Section({ title, children }) {
  return (
    <div style={{ marginBottom: '1.5rem' }}>
      <h3 style={{
        fontFamily: 'var(--font-heading)', fontSize: '.85rem', fontWeight: 700,
        textTransform: 'uppercase', letterSpacing: '.08em',
        color: 'var(--clr-accent2)', marginBottom: '1rem',
      }}>{title}</h3>
      {children}
    </div>
  )
}

export default function MetricsPage() {
  const { loading, data, error, run } = useAsync(getMetrics)
  useEffect(() => { run() }, [])

  return (
    <div className="fade-in">
      <div className="card">
        <div className="card-title" style={{ justifyContent: 'space-between' }}>
          <span><span className="icon">📈</span> Evaluation Metrics</span>
          <button id="refresh-metrics-btn" className="btn btn-secondary" onClick={run} disabled={loading}>
            {loading ? <Spinner /> : '🔄 Refresh'}
          </button>
        </div>

        {error && <Alert type="error">{error}</Alert>}

        {loading && (
          <div style={{ textAlign: 'center', padding: '3rem' }}>
            <Spinner />
          </div>
        )}

        {data && !data.available && (
          <Alert type="info">
            No metrics computed yet. Run <code>python scripts/evaluate.py</code> to generate results.
          </Alert>
        )}

        {data && data.available && (
          <div className="fade-in">
            {/* Meta */}
            <div style={{
              display: 'flex', gap: '.75rem', flexWrap: 'wrap', marginBottom: '1.5rem',
            }}>
              {[
                ['Backbone', data.backbone],
                ['Embedding Dim', data.embedding_dim],
                ['Split', data.split],
                ['Images', data.n_images],
                ['Classes', data.n_classes],
                ['Color Mode', data.color_mode],
              ].map(([k, v]) => (
                <span key={k} style={{
                  background: 'var(--clr-surface2)', border: '1px solid var(--clr-border)',
                  borderRadius: '6px', padding: '.25rem .65rem', fontSize: '.78rem',
                }}>
                  <span style={{ color: 'var(--clr-text3)' }}>{k}: </span>
                  <strong>{v ?? '—'}</strong>
                </span>
              ))}
            </div>

            {/* Identification */}
            {data.identification && (
              <Section title="Identification (Retrieval)">
                <div className="metric-grid">
                  {Object.entries(data.identification).map(([k, v]) => (
                    <MetricCard key={k} value={v} label={k.replace('@', ' @ ')} />
                  ))}
                </div>
              </Section>
            )}

            {/* Verification */}
            {data.verification && (
              <Section title="Verification (Pair Classification)">
                <div className="metric-grid">
                  {[
                    ['ROC-AUC', data.verification.roc_auc],
                    ['EER', data.verification.eer],
                    ['EER Threshold', data.verification.eer_threshold],
                    ['Best F1', data.verification.best_f1],
                    ['F1 Threshold', data.verification.best_f1_threshold],
                    ['Pairs Tested', data.verification.n_pairs],
                  ].map(([label, val]) => (
                    <MetricCard key={label} value={val} label={label} />
                  ))}
                </div>
              </Section>
            )}

            {/* Color Invariance */}
            {data.color_invariance && (
              <Section title="Color Invariance & Robustness">
                <div className="metric-grid">
                  {[
                    ['Colorway Recall @ 1', data.color_invariance['cross_recall@1'] ?? data.color_invariance['mean_perturbed_recall@1']],
                    ['Colorway Recall @ 5', data.color_invariance['cross_recall@5'] ?? data.color_invariance['mean_perturbed_recall@5']],
                    ['Colorway Recall @ 10', data.color_invariance['cross_recall@10'] ?? data.color_invariance['mean_perturbed_recall@10']],
                    ['Color Trap Accuracy', data.color_invariance.color_trap_accuracy],
                    ['Same-Design Diff-Color Recall', data.color_invariance.same_design_diff_color_recall],
                    ['Diff-Design Same-Color Rejection', data.color_invariance.diff_design_same_color_rejection],
                  ].filter(([_, val]) => val !== undefined && val !== null).map(([label, val]) => (
                    <MetricCard key={label} value={val} label={label} />
                  ))}
                </div>
              </Section>
            )}

            {/* Efficiency */}
            {data.efficiency && (
              <Section title="Efficiency">
                <div className="metric-grid">
                  {[
                    ['Total Params', data.efficiency.n_params_total],
                    ['Trainable', data.efficiency.n_params_trainable],
                    ['Emb Size (bytes)', data.efficiency.embedding_size_bytes],
                  ].map(([label, val]) => (
                    <MetricCard key={label} value={val} label={label} />
                  ))}
                </div>
              </Section>
            )}

            <p style={{ fontSize: '.75rem', color: 'var(--clr-text3)', marginTop: '1rem' }}>
              Source: <code>{data.checkpoint}</code>
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
