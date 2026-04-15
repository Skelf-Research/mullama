import { computed, ref, type Ref, watch } from 'vue'

export function usePlayground(selectedModel: Ref<string | null>) {
  const apiType = ref<'openai' | 'anthropic'>('openai')
  const requestBody = ref('')
  const response = ref('')
  const responseTime = ref<number | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  const updateTemplate = () => {
    if (apiType.value === 'openai') {
      requestBody.value = JSON.stringify(
        {
          model: selectedModel.value || 'default',
          messages: [{ role: 'user', content: 'Hello!' }],
          max_tokens: 256,
          temperature: 0.7,
          stream: false,
        },
        null,
        2
      )
    } else {
      requestBody.value = JSON.stringify(
        {
          model: selectedModel.value || 'default',
          max_tokens: 256,
          messages: [{ role: 'user', content: 'Hello!' }],
        },
        null,
        2
      )
    }
  }

  const endpoint = computed(() => {
    return apiType.value === 'openai' ? '/v1/chat/completions' : '/v1/messages'
  })

  const storedApiKey = computed(() => {
    try {
      const rawSettings = localStorage.getItem('mullama-settings')
      if (rawSettings) {
        const parsed = JSON.parse(rawSettings) as { apiKey?: string }
        if (parsed.apiKey && parsed.apiKey.trim()) {
          return parsed.apiKey.trim()
        }
      }
      const direct = localStorage.getItem('mullama-api-key')
      return direct?.trim() || ''
    } catch {
      return ''
    }
  })

  const curlCommand = computed(() => {
    const body = requestBody.value.replace(/\n/g, ' ').replace(/\s+/g, ' ')
    const authLine = storedApiKey.value
      ? ` \\\n  -H "Authorization: Bearer ${storedApiKey.value}" \\\n  -H "X-API-Key: ${storedApiKey.value}"`
      : ''

    return `curl -X POST ${window.location.origin}${endpoint.value} \\
  -H "Content-Type: application/json" \\
${authLine}
  -d '${body}'`
  })

  const sendRequest = async () => {
    loading.value = true
    error.value = null
    response.value = ''

    const start = performance.now()

    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      if (storedApiKey.value) {
        headers.Authorization = `Bearer ${storedApiKey.value}`
        headers['X-API-Key'] = storedApiKey.value
      }

      const res = await fetch(endpoint.value, {
        method: 'POST',
        headers,
        body: requestBody.value,
      })

      responseTime.value = Math.round(performance.now() - start)

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ error: { message: res.statusText } }))
        throw new Error(errData.error?.message || `HTTP ${res.status}`)
      }

      response.value = JSON.stringify(await res.json(), null, 2)
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Request failed'
    } finally {
      loading.value = false
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  watch([apiType, selectedModel], updateTemplate, { immediate: true })

  return {
    apiType,
    requestBody,
    response,
    responseTime,
    loading,
    error,
    endpoint,
    curlCommand,
    updateTemplate,
    sendRequest,
    copyToClipboard,
  }
}
