<script setup lang="ts">
import StatusIndicator from './StatusIndicator.vue'

defineProps<{
  name: string
  status: 'loaded' | 'available' | 'downloading'
  ownedBy?: string
  size?: string
  parameters?: string
}>()

const emit = defineEmits<{
  (e: 'delete'): void
  (e: 'select'): void
}>()
</script>

<template>
  <div class="card p-4 hover:shadow-md transition-shadow">
    <div class="flex items-start justify-between mb-3">
      <div class="flex-1 min-w-0">
        <h3 class="font-medium text-gray-900 dark:text-white truncate">{{ name }}</h3>
        <p v-if="ownedBy" class="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
          {{ ownedBy }}
        </p>
      </div>
      <StatusIndicator
        :status="status === 'loaded' ? 'online' : status === 'downloading' ? 'loading' : 'offline'"
      />
    </div>

    <div v-if="size || parameters" class="flex gap-4 text-sm text-gray-600 dark:text-gray-400 mb-3">
      <span v-if="size">{{ size }}</span>
      <span v-if="parameters">{{ parameters }}</span>
    </div>

    <div class="flex gap-2">
      <button
        @click="emit('select')"
        class="btn btn-secondary text-sm flex-1"
      >
        Select
      </button>
      <button
        @click="emit('delete')"
        class="p-2 text-gray-400 hover:text-red-500 transition-colors"
        title="Delete model"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
        </svg>
      </button>
    </div>
  </div>
</template>
