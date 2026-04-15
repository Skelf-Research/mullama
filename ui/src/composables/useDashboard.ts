import { ref, onMounted, onUnmounted } from 'vue'
import { management, metrics } from '@/api/client'

export interface SystemStatus {
  uptime_secs: number
  version: string
  models_loaded: number
  http_endpoint?: string
}

export function useDashboard() {
  const status = ref<SystemStatus | null>(null)
  const metricsData = ref('')
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
      // Metrics endpoint may not be available.
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

  return {
    status,
    metricsData,
    loading,
    error,
    formatUptime,
    fetchStatus,
    fetchMetrics,
  }
}
