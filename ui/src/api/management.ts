import { buildHeaders } from './auth'
import { fetchApi } from './http'
import type {
  DefaultModel,
  DefaultsResponse,
  ModelDetails,
  PullProgress,
  SystemStatus,
} from './types'

export const management = {
  async status(): Promise<SystemStatus> {
    return fetchApi<SystemStatus>('/api/system/status')
  },

  async listModels(): Promise<ModelDetails[]> {
    const response = await fetchApi<{
      models: ModelDetails[]
      available_aliases: string[]
      total_cached: number
    }>('/api/models')
    return response.models
  },

  async getModel(name: string): Promise<ModelDetails> {
    return fetchApi<ModelDetails>(`/api/models/${encodeURIComponent(name)}`)
  },

  async pullModel(
    name: string,
    onProgress?: (progress: PullProgress) => void
  ): Promise<void> {
    const response = await fetch('/api/models/pull', {
      method: 'POST',
      headers: buildHeaders(),
      body: JSON.stringify({ name }),
    })

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ error: { message: response.statusText } }))
      throw new Error(error.error?.message || `HTTP ${response.status}`)
    }

    const reader = response.body?.getReader()
    if (reader && onProgress) {
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.trim()) {
            try {
              const progress = JSON.parse(line) as PullProgress
              onProgress(progress)
            } catch {
              // Ignore parse errors.
            }
          }
        }
      }
    }
  },

  async deleteModel(name: string): Promise<void> {
    await fetchApi<void>(`/api/models/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    })
  },

  async loadModel(
    name: string,
    options?: { gpu_layers?: number; context_size?: number }
  ): Promise<{ success: boolean; message: string; model?: unknown }> {
    return fetchApi<{ success: boolean; message: string; model?: unknown }>(
      '/api/models/load',
      {
        method: 'POST',
        body: JSON.stringify({
          name,
          gpu_layers: options?.gpu_layers,
          context_size: options?.context_size,
        }),
      }
    )
  },

  async unloadModel(name: string): Promise<{ success: boolean; message: string }> {
    return fetchApi<{ success: boolean; message: string }>(
      `/api/models/${encodeURIComponent(name)}/unload`,
      {
        method: 'POST',
      }
    )
  },

  async listDefaults(): Promise<DefaultModel[]> {
    const response = await fetchApi<DefaultsResponse>('/api/defaults')
    return response.models
  },

  async useDefault(
    name: string
  ): Promise<{ success: boolean; message: string; model?: unknown }> {
    return fetchApi<{ success: boolean; message: string; model?: unknown }>(
      `/api/defaults/${encodeURIComponent(name)}/use`,
      {
        method: 'POST',
      }
    )
  },
}
