<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { management, metrics } from '@/api/client'
import StatusIndicator from '@/components/StatusIndicator.vue'

interface SystemStatus {
  uptime_secs: number
  version: string
  models_loaded: number
  http_endpoint?: string
}

const status = ref<SystemStatus | null>(null)
const metricsData = ref<string>('')
const loading = ref(true)
const error = ref<string | null>(null)
let refreshInterval: number | null = null

const formatUptime = (seconds: number): string => {
  const days = Math.floor(seconds / 86400)
  const hours = Math.floor((seconds % 86400) / 3600)
  const mins = Math.floor((seconds % 3600) / 60)

  if (days > 0) return `${days}d ${hours}h ${mins}m`
  if (hours > 0) return `${hours}h ${mins}m`
  return `${mins}m`
}

const fetchStatus = async () => {
  try {
    status.value = await management.status()
    error.value = null
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to fetch status'
  } finally {
    loading.value = false
  }
}

const fetchMetrics = async () => {
  try {
    metricsData.value = await metrics.get()
  } catch {
    // Metrics endpoint may not be available
  }
}

onMounted(() => {
  fetchStatus()
  fetchMetrics()
  refreshInterval = window.setInterval(() => {
    fetchStatus()
    fetchMetrics()
  }, 5000)
})

onUnmounted(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
  }
})

const parseMetrics = (text: string): Record<string, string> => {
  const result: Record<string, string> = {}
  const lines = text.split('\n')
  for (const line of lines) {
    if (line && !line.startsWith('#')) {
      const [key, value] = line.split(' ')
      if (key && value) {
        result[key] = value
      }
    }
  }
  return result
}
</script>

<template>
  <div class="p-6 max-w-7xl mx-auto">
    <h1 class="text-2xl font-bold text-gray-900 dark:text-white mb-6">Dashboard</h1>

    <!-- Status Cards -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
      <!-- Status -->
      <div class="card p-6">
        <div class="flex items-center justify-between mb-2">
          <span class="text-sm text-gray-500 dark:text-gray-400">Status</span>
          <StatusIndicator :status="status ? 'online' : 'offline'" />
        </div>
        <p class="text-2xl font-semibold text-gray-900 dark:text-white">
          {{ status ? 'Online' : loading ? 'Loading...' : 'Offline' }}
        </p>
      </div>

      <!-- Version -->
      <div class="card p-6">
        <div class="flex items-center justify-between mb-2">
          <span class="text-sm text-gray-500 dark:text-gray-400">Version</span>
          <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
          </svg>
        </div>
        <p class="text-2xl font-semibold text-gray-900 dark:text-white">
          {{ status?.version || '-' }}
        </p>
      </div>

      <!-- Uptime -->
      <div class="card p-6">
        <div class="flex items-center justify-between mb-2">
          <span class="text-sm text-gray-500 dark:text-gray-400">Uptime</span>
          <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <p class="text-2xl font-semibold text-gray-900 dark:text-white">
          {{ status ? formatUptime(status.uptime_secs) : '-' }}
        </p>
      </div>

      <!-- Models Loaded -->
      <div class="card p-6">
        <div class="flex items-center justify-between mb-2">
          <span class="text-sm text-gray-500 dark:text-gray-400">Models Loaded</span>
          <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
        </div>
        <p class="text-2xl font-semibold text-gray-900 dark:text-white">
          {{ status?.models_loaded ?? 0 }}
        </p>
      </div>
    </div>

    <!-- Error Alert -->
    <div v-if="error" class="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
      <div class="flex items-center gap-3">
        <svg class="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <p class="text-red-700 dark:text-red-400">{{ error }}</p>
      </div>
    </div>

    <!-- Quick Actions -->
    <div class="card p-6 mb-8">
      <h2 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">Quick Actions</h2>
      <div class="flex flex-wrap gap-3">
        <RouterLink to="/chat" class="btn btn-primary">
          <span class="flex items-center gap-2">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
            Start Chat
          </span>
        </RouterLink>
        <RouterLink to="/models" class="btn btn-secondary">
          <span class="flex items-center gap-2">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Pull Model
          </span>
        </RouterLink>
        <RouterLink to="/playground" class="btn btn-secondary">
          <span class="flex items-center gap-2">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
            </svg>
            API Playground
          </span>
        </RouterLink>
      </div>
    </div>

    <!-- Connection Info -->
    <div class="card p-6">
      <h2 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">Connection Info</h2>
      <div class="space-y-3">
        <div class="flex items-center justify-between py-2 border-b border-gray-100 dark:border-gray-700">
          <span class="text-gray-600 dark:text-gray-400">HTTP Endpoint</span>
          <code class="text-sm bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
            {{ status?.http_endpoint || 'http://localhost:8080' }}
          </code>
        </div>
        <div class="flex items-center justify-between py-2 border-b border-gray-100 dark:border-gray-700">
          <span class="text-gray-600 dark:text-gray-400">OpenAI API</span>
          <code class="text-sm bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
            /v1/chat/completions
          </code>
        </div>
        <div class="flex items-center justify-between py-2 border-b border-gray-100 dark:border-gray-700">
          <span class="text-gray-600 dark:text-gray-400">Anthropic API</span>
          <code class="text-sm bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
            /v1/messages
          </code>
        </div>
        <div class="flex items-center justify-between py-2">
          <span class="text-gray-600 dark:text-gray-400">Metrics</span>
          <code class="text-sm bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
            /metrics
          </code>
        </div>
      </div>
    </div>
  </div>
</template>
