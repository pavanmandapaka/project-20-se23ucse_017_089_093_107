import { useState, useRef, useCallback } from 'react'
import './App.css'

const API_BASE = 'http://localhost:8000/api/v1'

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  const handleFile = useCallback((f) => {
    if (!f) return
    setError(null)
    setResult(null)

    const allowed = ['image/png', 'image/jpeg', 'image/webp', 'image/tiff', 'image/bmp']
    if (!allowed.includes(f.type)) {
      setError('Unsupported format. Please upload PNG, JPEG, WebP, TIFF, or BMP.')
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

      const res = await fetch(`${API_BASE}/inference/run`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        throw new Error(`Server responded with ${res.status}`)
      }

      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Upload failed. Make sure the server is running.')
    } finally {
      setUploading(false)
    }
  }, [file])

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header" id="app-header">
        <div className="header__brand">
          <div className="header__logo" aria-hidden="true">G</div>
          <h1 className="header__title">Genni</h1>
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
            <div className="preview" id="image-preview">
              <div className="preview__image-wrap">
                <img className="preview__image" src={preview} alt="Medical image preview" />
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
              <p className="result__text">
                {result.caption || result.prompt || JSON.stringify(result)}
              </p>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="error-card" id="error-card">
              <svg className="error-card__icon" xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z" />
              </svg>
              <p className="error-card__text">{error}</p>
            </div>
          )}
        </div>
      </main>

      {/* ── Footer ── */}
      <footer className="footer" id="app-footer">
        Genni — AI-Powered Medical Image Analysis
      </footer>
    </div>
  )
}

export default App
