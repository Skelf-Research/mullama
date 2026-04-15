import { buildHeaders } from './auth'

const BASE_URL = ''

export async function fetchApi<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: buildHeaders(options?.headers),
  })

  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ error: { message: response.statusText } }))
    throw new Error(error.error?.message || `HTTP ${response.status}`)
  }

  return response.json()
}
