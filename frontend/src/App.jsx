import { useState, useRef, useCallback, useEffect } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api/v1'
const MAX_FILE_SIZE = 50 * 1024 * 1024 // 50 MB

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

/* ── Skeleton placeholder ── */
function Skeleton({ width = '100%', height = '1rem', rounded = false }) {
  return (
    <div
      className="skeleton"
      style={{
        width,
        height,
        borderRadius: rounded ? '9999px' : 'var(--radius-md)',
      }}
    />
  )
}

/* ── Card Skeleton for dashboard ── */
function CardSkeleton() {
  return (
    <div className="card card--skeleton">
      <Skeleton width="60%" height="0.75rem" />
      <Skeleton width="40%" height="1.25rem" />
    </div>
  )
}

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [prompt, setPrompt] = useState('')
  const [modelVersion, setModelVersion] = useState('blip-image-captioning-base')
  const [health, setHealth] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [datasets, setDatasets] = useState([])
  const [trainingJobs, setTrainingJobs] = useState([])
  const [history, setHistory] = useState([])
  const [loadingDashboard, setLoadingDashboard] = useState(false)
  const [dashboardError, setDashboardError] = useState(null)
  const [serverOnline, setServerOnline] = useState(null) // null = unknown, true/false
  const inputRef = useRef(null)

  const fetchJSON = useCallback(async (url, options) => {
    const res = await fetch(url, options)
    if (!res.ok) {
      throw new Error(`Server responded with ${res.status}`)
    }
    return res.json()
  }, [])

  const loadDashboard = useCallback(async () => {
    setLoadingDashboard(true)
    setDashboardError(null)
    try {
      const [healthData, metricsData, datasetsData, jobsData, historyData] = await Promise.all([
        fetchJSON(`${API_BASE.replace('/api/v1', '')}/health`),
        fetchJSON(`${API_BASE}/metrics/summary`),
        fetchJSON(`${API_BASE}/datasets`),
        fetchJSON(`${API_BASE}/training/jobs`),
        fetchJSON(`${API_BASE}/inference/history?limit=8`),
      ])

      setHealth(healthData)
      setMetrics(metricsData)
      setDatasets(datasetsData.datasets || [])
      setTrainingJobs(jobsData.jobs || [])
      setHistory(historyData.inferences || [])
      setServerOnline(true)
    } catch (err) {
      setDashboardError(err.message || 'Failed to load dashboard data.')
      setServerOnline(false)
    } finally {
      setLoadingDashboard(false)
    }
  }, [fetchJSON])

  useEffect(() => {
    loadDashboard()
  }, [loadDashboard])

  const handleFile = useCallback((f) => {
    if (!f) return
    setError(null)
    setResult(null)

    const allowed = ['image/png', 'image/jpeg', 'image/webp', 'image/tiff', 'image/bmp']
    if (!allowed.includes(f.type)) {
      setError('Unsupported format. Please upload PNG, JPEG, WebP, TIFF, or BMP.')
      return
    }

    if (f.size > MAX_FILE_SIZE) {
      setError(`File too large (${formatBytes(f.size)}). Maximum allowed size is 50 MB.`)
      return
    }

    setFile(f)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }, [])

  const onDrop = useCallback(
    (e) => {
      e.preventDefault()
      setDragActive(false)
      const dropped = e.dataTransfer?.files?.[0]
      handleFile(dropped)
    },
    [handleFile]
  )

  const onDragOver = useCallback((e) => {
    e.preventDefault()
    setDragActive(true)
  }, [])

  const onDragLeave = useCallback((e) => {
    e.preventDefault()
    setDragActive(false)
  }, [])

  const onInputChange = useCallback(
    (e) => {
      handleFile(e.target.files?.[0])
    },
    [handleFile]
  )

  const removeFile = useCallback(() => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
    if (inputRef.current) inputRef.current.value = ''
  }, [])

  const handleUpload = useCallback(async () => {
    if (!file) return
    setUploading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)
      if (prompt.trim()) {
        formData.append('prompt', prompt.trim())
      }
      formData.append('model_version', modelVersion)

      const data = await fetchJSON(`${API_BASE}/inference/run`, {
        method: 'POST',
        body: formData,
      })

      setResult(data)
      if (data?.inference_id) {
        const detail = await fetchJSON(`${API_BASE}/inference/${data.inference_id}`)
        setResult((prev) => ({ ...prev, detail }))
      }
      loadDashboard()
    } catch (err) {
      setError(err.message || 'Upload failed. Make sure the server is running.')
    } finally {
      setUploading(false)
    }
  }, [file, prompt, modelVersion, fetchJSON, loadDashboard])

  const dismissError = useCallback(() => {
    setError(null)
  }, [])

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header" id="app-header">
        <div className="header__brand">
          <div className="header__logo" aria-hidden="true">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
              <circle cx="12" cy="12" r="3" />
            </svg>
          </div>
          <h1 className="header__title">VLM Project</h1>
          <span
            className={`header__status ${
              serverOnline === null
                ? 'header__status--unknown'
                : serverOnline
                  ? 'header__status--online'
                  : 'header__status--offline'
            }`}
            title={
              serverOnline === null
                ? 'Checking server…'
                : serverOnline
                  ? 'Server online'
                  : 'Server offline'
            }
          />
        </div>
        <span className="header__badge">Medical AI</span>
      </header>

      {/* ── Main ── */}
      <main className="main" id="upload-section">
        <div className="upload-container">
          <div className="upload-container__heading">
            <h2 className="upload-container__title">Upload Medical Image</h2>
            <p className="upload-container__subtitle">
              Drag &amp; drop a medical image or click to browse. Supported formats: PNG, JPEG, WebP, TIFF, BMP.
            </p>
          </div>

          {/* Controls */}
          <div className="controls">
            <label className="controls__field">
              <span className="controls__label">Model version</span>
              <input
                className="controls__input"
                type="text"
                value={modelVersion}
                onChange={(e) => setModelVersion(e.target.value)}
                placeholder="blip-image-captioning-base"
              />
            </label>
            <label className="controls__field">
              <span className="controls__label">Optional prompt</span>
              <input
                className="controls__input"
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="e.g., Describe findings succinctly"
              />
            </label>
          </div>

          {/* Drop Zone */}
          <div
            id="dropzone"
            className={`dropzone${dragActive ? ' dropzone--active' : ''}`}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            aria-label="Upload medical image"
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click()
            }}
          >
            <div className="dropzone__icon">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
              </svg>
            </div>
            <p className="dropzone__label">
              Drop your image here or <span className="dropzone__label-accent">browse</span>
            </p>
            <p className="dropzone__hint">PNG, JPEG, WebP, TIFF, BMP — max 50 MB</p>
            <input
              ref={inputRef}
              id="file-input"
              className="dropzone__input"
              type="file"
              accept="image/png,image/jpeg,image/webp,image/tiff,image/bmp"
              onChange={onInputChange}
            />
          </div>

          {/* Preview */}
          {preview && file && (
            <div className={`preview${uploading ? ' preview--analyzing' : ''}`} id="image-preview">
              <div className="preview__image-wrap">
                <img className="preview__image" src={preview} alt="Medical image preview" />
                {uploading && (
                  <div className="preview__overlay">
                    <div className="preview__overlay-spinner" />
                    <span className="preview__overlay-text">Analyzing…</span>
                  </div>
                )}
              </div>
              <div className="preview__info">
                <div>
                  <p className="preview__name">{file.name}</p>
                  <p className="preview__size">{formatBytes(file.size)}</p>
                </div>
                <button
                  className="preview__remove"
                  onClick={(e) => {
                    e.stopPropagation()
                    removeFile()
                  }}
                  aria-label="Remove image"
                  id="remove-image-btn"
                  disabled={uploading}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>
          )}

          {/* Upload Button */}
          <button
            id="upload-btn"
            className={`upload-btn${uploading ? ' upload-btn--loading' : ''}`}
            disabled={!file || uploading}
            onClick={handleUpload}
          >
            {uploading ? (
              <>
                <span className="upload-btn__spinner" />
                Analyzing…
              </>
            ) : (
              <>
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
                </svg>
                Upload &amp; Analyze
              </>
            )}
          </button>

          {/* Result */}
          {result && (
            <div className="result" id="result-card">
              <div className="result__header">
                <span className="result__dot" />
                <span className="result__label">Analysis Result</span>
              </div>
              <p className="result__text">{result.caption || result.prompt || 'Inference received.'}</p>
              {result.inference_id && (
                <div className="result__meta">
                  <span>Inference ID: {result.inference_id}</span>
                  <span>Model: {result.model_version}</span>
                </div>
              )}
              {result.detail?.generated_text && (
                <p className="result__text">{result.detail.generated_text}</p>
              )}
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="error-card" id="error-card">
              <svg className="error-card__icon" xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z" />
              </svg>
              <p className="error-card__text">{error}</p>
              <div className="error-card__actions">
                {file && (
                  <button className="error-card__retry" onClick={handleUpload} id="retry-btn">
                    Retry
                  </button>
                )}
                <button className="error-card__dismiss" onClick={dismissError} aria-label="Dismiss error" id="dismiss-error-btn">
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>
          )}

          {/* ── Live Data Dashboard ── */}
          <section className="dashboard" id="dashboard">
            <div className="dashboard__header">
              <h3 className="dashboard__title">Live API Status</h3>
              <button
                className="dashboard__refresh"
                onClick={loadDashboard}
                disabled={loadingDashboard}
              >
                {loadingDashboard ? (
                  <>
                    <span className="dashboard__refresh-spinner" />
                    Refreshing…
                  </>
                ) : (
                  'Refresh'
                )}
              </button>
            </div>

            {/* Dashboard Error */}
            {dashboardError && !loadingDashboard && (
              <div className="dashboard__error">
                <div className="dashboard__error-content">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126Z" />
                  </svg>
                  <div>
                    <p className="dashboard__error-title">Unable to connect to server</p>
                    <p className="dashboard__error-text">{dashboardError}</p>
                  </div>
                </div>
                <button className="dashboard__error-retry" onClick={loadDashboard}>
                  Try Again
                </button>
              </div>
            )}

            {/* Dashboard Loading Skeletons */}
            {loadingDashboard && (
              <div className="dashboard__grid">
                <CardSkeleton />
                <CardSkeleton />
                <CardSkeleton />
                <CardSkeleton />
              </div>
            )}

            {/* Dashboard Data */}
            {!loadingDashboard && !dashboardError && (
              <>
                <div className="dashboard__grid">
                  <div className="card">
                    <h4 className="card__title">Health</h4>
                    <p className={`card__body card__body--status ${health?.status === 'ok' ? 'card__body--ok' : ''}`}>
                      {health?.status || 'unknown'}
                    </p>
                  </div>
                  <div className="card">
                    <h4 className="card__title">Metrics</h4>
                    <p className="card__body">
                      {metrics?.status ? metrics.status : 'not available'}
                    </p>
                  </div>
                  <div className="card">
                    <h4 className="card__title">Datasets</h4>
                    <p className="card__body">
                      {datasets.length ? `${datasets.length} available` : 'none found'}
                    </p>
                  </div>
                  <div className="card">
                    <h4 className="card__title">Training Jobs</h4>
                    <p className="card__body">
                      {trainingJobs.length ? `${trainingJobs.length} active` : 'none queued'}
                    </p>
                  </div>
                </div>

                <div className="card card--full">
                  <h4 className="card__title">Recent Inferences</h4>
                  {history.length === 0 ? (
                    <div className="empty-state">
                      <svg className="empty-state__icon" xmlns="http://www.w3.org/2000/svg" width="32" height="32" fill="none" viewBox="0 0 24 24" strokeWidth={1} stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
                      </svg>
                      <p className="empty-state__title">No inferences yet</p>
                      <p className="empty-state__text">Upload a medical image above to run your first analysis.</p>
                    </div>
                  ) : (
                    <ul className="history">
                      {history.map((item) => (
                        <li key={item.id} className="history__item">
                          <div>
                            <p className="history__title">#{item.id} · {item.model_version}</p>
                            <p className="history__meta">{item.image_path}</p>
                          </div>
                          <span className="history__status">{item.generated_text}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </>
            )}

            {/* Loading skeletons for history card */}
            {loadingDashboard && (
              <div className="card card--full card--skeleton">
                <Skeleton width="35%" height="0.75rem" />
                <div style={{ display: 'grid', gap: '0.75rem', marginTop: '1rem' }}>
                  <Skeleton height="2.5rem" />
                  <Skeleton height="2.5rem" />
                  <Skeleton height="2.5rem" />
                </div>
              </div>
            )}
          </section>
        </div>
      </main>

      {/* ── Footer ── */}
      <footer className="footer" id="app-footer">
        VLM Project — AI-Powered Medical Image Analysis
      </footer>
    </div>
  )
}

export default App
