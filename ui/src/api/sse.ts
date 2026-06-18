export function processDataLines(
  buffer: string,
  onChunk: (content: string, thinking?: string) => void
) {
  const lines = buffer.split('\n')
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = line.slice(6).trim()
      if (data && data !== '[DONE]') {
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
