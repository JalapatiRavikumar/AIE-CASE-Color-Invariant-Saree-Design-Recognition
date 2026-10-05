import { useState, useCallback } from 'react'

export function useImageUpload() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [dragging, setDragging] = useState(false)

  const handleFile = useCallback((f) => {
    if (!f || !f.type.startsWith('image/')) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
  }, [])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }, [handleFile])

  const onDragOver = useCallback((e) => {
    e.preventDefault(); setDragging(true)
  }, [])

  const onDragLeave = useCallback(() => setDragging(false), [])

  const reset = useCallback(() => {
    setFile(null); setPreview(null)
  }, [])

  return { file, preview, dragging, handleFile, onDrop, onDragOver, onDragLeave, reset }
}

export function useAsync(asyncFn) {
  const [loading, setLoading] = useState(false)
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  const run = useCallback(async (...args) => {
    setLoading(true); setError(null); setData(null)
    try {
      const result = await asyncFn(...args)
      setData(result)
      return result
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Unknown error'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }, [asyncFn])

  return { loading, data, error, run }
}
