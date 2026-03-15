<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useModels } from '@/composables/useModels'

const { models, selectedModel, fetchModels } = useModels()

const apiType = ref<'openai' | 'anthropic'>('openai')
const requestBody = ref('')
const response = ref('')
const responseTime = ref<number | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

onMounted(() => {
  fetchModels()
  updateTemplate()
})

const updateTemplate = () => {
  if (apiType.value === 'openai') {
    requestBody.value = JSON.stringify({
      model: selectedModel.value || 'default',
      messages: [
        { role: 'user', content: 'Hello!' }
      ],
      max_tokens: 256,
      temperature: 0.7,
      stream: false
    }, null, 2)
  } else {
    requestBody.value = JSON.stringify({
      model: selectedModel.value || 'default',
      max_tokens: 256,
      messages: [
        { role: 'user', content: 'Hello!' }
      ]
    }, null, 2)
  }
}

const endpoint = computed(() => {
  return apiType.value === 'openai' ? '/v1/chat/completions' : '/v1/messages'
})

const curlCommand = computed(() => {
  const body = requestBody.value.replace(/\n/g, ' ').replace(/\s+/g, ' ')
  return `curl -X POST ${window.location.origin}${endpoint.value} \\
  -H "Content-Type: application/json" \\
  -d '${body}'`
})

const sendRequest = async () => {
  loading.value = true
  error.value = null
  response.value = ''

  const start = performance.now()

  try {
    const res = await fetch(endpoint.value, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: requestBody.value,
    })

    responseTime.value = Math.round(performance.now() - start)

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ error: { message: res.statusText } }))
      throw new Error(errData.error?.message || `HTTP ${res.status}`)
    }

    const data = await res.json()
    response.value = JSON.stringify(data, null, 2)
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Request failed'
  } finally {
    loading.value = false
  }
}

const copyToClipboard = (text: string) => {
  navigator.clipboard.writeText(text)
}
</script>

<template>
  <div class="p-6 max-w-7xl mx-auto">
    <h1 class="text-2xl font-bold text-gray-900 dark:text-white mb-6">API Playground</h1>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Request Panel -->
      <div class="card p-6">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-lg font-semibold text-gray-900 dark:text-white">Request</h2>
          <div class="flex items-center gap-2">
            <select v-model="apiType" @change="updateTemplate" class="input py-1 px-2 text-sm">
              <option value="openai">OpenAI API</option>
              <option value="anthropic">Anthropic API</option>
            </select>
          </div>
        </div>

        <!-- Endpoint -->
        <div class="mb-4">
          <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Endpoint
          </label>
          <div class="flex items-center gap-2">
            <span class="px-3 py-2 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 text-sm font-medium rounded-lg">
              POST
            </span>
            <code class="flex-1 px-3 py-2 bg-gray-100 dark:bg-gray-700 rounded-lg text-sm font-mono">
              {{ endpoint }}
            </code>
          </div>
        </div>

        <!-- Request Body -->
        <div class="mb-4">
          <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Request Body
          </label>
          <textarea
            v-model="requestBody"
            class="input font-mono text-sm"
            rows="12"
          />
        </div>

        <!-- Actions -->
        <div class="flex gap-3">
          <button
            @click="sendRequest"
            class="btn btn-primary flex-1"
            :disabled="loading"
          >
            {{ loading ? 'Sending...' : 'Send Request' }}
          </button>
          <button
            @click="updateTemplate"
            class="btn btn-secondary"
          >
            Reset
          </button>
        </div>
      </div>

      <!-- Response Panel -->
      <div class="card p-6">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-lg font-semibold text-gray-900 dark:text-white">Response</h2>
          <div v-if="responseTime !== null" class="text-sm text-gray-500 dark:text-gray-400">
            {{ responseTime }}ms
          </div>
        </div>

        <!-- Error -->
        <div v-if="error" class="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg mb-4">
          <p class="text-red-700 dark:text-red-400 text-sm">{{ error }}</p>
        </div>

        <!-- Response Body -->
        <div class="relative">
          <pre class="p-4 bg-gray-900 text-gray-100 rounded-lg overflow-x-auto text-sm min-h-[300px]"><code>{{ response || '// Response will appear here' }}</code></pre>
          <button
            v-if="response"
            @click="copyToClipboard(response)"
            class="absolute top-2 right-2 p-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
            title="Copy response"
          >
            <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
            </svg>
          </button>
        </div>
      </div>
    </div>

    <!-- cURL Command -->
    <div class="card p-6 mt-6">
      <div class="flex items-center justify-between mb-4">
        <h2 class="text-lg font-semibold text-gray-900 dark:text-white">cURL Command</h2>
        <button
          @click="copyToClipboard(curlCommand)"
          class="btn btn-secondary text-sm"
        >
          <span class="flex items-center gap-2">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
            </svg>
            Copy
          </span>
        </button>
      </div>
      <pre class="p-4 bg-gray-900 text-gray-100 rounded-lg overflow-x-auto text-sm"><code>{{ curlCommand }}</code></pre>
    </div>
  </div>
</template>
