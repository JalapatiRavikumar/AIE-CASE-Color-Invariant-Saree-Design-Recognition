import axios from 'axios'

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const api = axios.create({ baseURL: BASE, timeout: 60_000 })

export const getStatus = () => api.get('/api/status').then(r => r.data)
export const getMetrics = () => api.get('/api/metrics').then(r => r.data)
export const getDownloads = () => api.get('/api/downloads').then(r => r.data)

export const identify = (file, topK = 5) => {
  const form = new FormData()
  form.append('file', file)
  return api.post(`/api/identify?top_k=${topK}`, form).then(r => r.data)
}

export const verify = (fileA, fileB) => {
  const form = new FormData()
  form.append('file_a', fileA)
  form.append('file_b', fileB)
  return api.post('/api/verify', form).then(r => r.data)
}

export const embed = (file) => {
  const form = new FormData()
  form.append('file', file)
  return api.post('/api/embed', form).then(r => r.data)
}

export const indexGallery = (galleryPath) => {
  const form = new FormData()
  form.append('gallery_path', galleryPath)
  return api.post('/api/index', form).then(r => r.data)
}

export const downloadUrl = (filename) => `${BASE}/api/download/${filename}`

export default api
