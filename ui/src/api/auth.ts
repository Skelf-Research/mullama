export function getStoredApiKey(): string | null {
  try {
    const direct = localStorage.getItem('mullama-api-key')
    if (direct && direct.trim()) {
      return direct.trim()
    }

    const rawSettings = localStorage.getItem('mullama-settings')
    if (!rawSettings) {
      return null
    }

    const settings = JSON.parse(rawSettings) as { apiKey?: string }
    if (settings.apiKey && settings.apiKey.trim()) {
      return settings.apiKey.trim()
    }
  } catch {
    // Ignore storage parsing errors.
  }

  return null
}

export function buildHeaders(extra?: HeadersInit): HeadersInit {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  }

  const apiKey = getStoredApiKey()
  if (apiKey) {
    headers.Authorization = `Bearer ${apiKey}`
    headers['X-API-Key'] = apiKey
  }

  const merged = new Headers(extra)
  for (const [key, value] of Object.entries(headers)) {
    merged.set(key, value)
  }
  return merged
}
