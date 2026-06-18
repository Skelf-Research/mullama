import { buildHeaders } from './auth'
import { fetchApi } from './http'
import { processDataLines } from './sse'
import type { ChatRequest, ChatResponse, Model, ModelsResponse } from './types'

export const openai = {
  async listModels(): Promise<Model[]> {
    const res = await fetchApi<ModelsResponse>('/v1/models')
    return res.data
  },

  async chat(request: ChatRequest): Promise<ChatResponse> {
    return fetchApi<ChatResponse>('/v1/chat/completions', {
      method: 'POST',
      body: JSON.stringify({ ...request, stream: false }),
    })
  },

  async chatStream(
    request: ChatRequest,
    onChunk: (content: string, thinking?: string) => void
  ): Promise<void> {
    const response = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: buildHeaders(),
      body: JSON.stringify({ ...request, stream: true }),
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    const reader = response.body?.getReader()
    if (!reader) throw new Error('No response body')

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()

      if (done) {
        if (buffer.trim()) {
          processDataLines(buffer, onChunk)
        }
        break
      }

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim()
          if (data === '[DONE]') {
            return
          }
          if (data) {
            try {
              const parsed = JSON.parse(data)
              const content = parsed.choices?.[0]?.delta?.content
              const thinking = parsed.choices?.[0]?.delta?.thinking
              if (content || thinking) {
                onChunk(content || '', thinking)
              }
            } catch {
              // Ignore parse errors.
            }
          }
        }
      }
    }
  },
}
