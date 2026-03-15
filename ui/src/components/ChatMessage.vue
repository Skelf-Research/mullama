<script setup lang="ts">
import { computed, ref } from 'vue'
import { useMarkdown } from '@/composables/useMarkdown'

const props = defineProps<{
  role: 'user' | 'assistant' | 'system'
  content: string
  thinking?: string
  timestamp?: string
  isStreaming?: boolean
}>()

const { renderMarkdown } = useMarkdown()
const showThinking = ref(false)

const isUser = computed(() => props.role === 'user')

// Render content with markdown
const formattedContent = computed(() => {
  if (!props.content) return ''
  return renderMarkdown(props.content)
})

// Render thinking content
const formattedThinking = computed(() => {
  if (!props.thinking) return ''
  return renderMarkdown(props.thinking)
})
</script>

<template>
  <div
    :class="[
      'flex gap-3',
      isUser ? 'flex-row-reverse' : ''
    ]"
  >
    <!-- Avatar -->
    <div
      :class="[
        'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0',
        isUser ? 'bg-primary-100 dark:bg-primary-900' : 'bg-gray-200 dark:bg-gray-700'
      ]"
    >
      <svg v-if="isUser" class="w-4 h-4 text-primary-600 dark:text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
      </svg>
      <svg v-else class="w-4 h-4 text-gray-600 dark:text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    </div>

    <!-- Message -->
    <div
      :class="[
        'max-w-[80%] rounded-2xl px-4 py-2',
        isUser
          ? 'bg-primary-600 text-white'
          : 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700'
      ]"
    >
      <!-- Thinking block (collapsible) -->
      <div v-if="thinking" class="thinking-block mb-3">
        <button
          @click="showThinking = !showThinking"
          class="flex items-center gap-2 text-sm font-medium text-amber-600 dark:text-amber-400"
        >
          <svg
            :class="['w-4 h-4 transition-transform', showThinking ? 'rotate-90' : '']"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
          </svg>
          Thinking...
        </button>
        <div
          v-show="showThinking"
          class="mt-2 pl-6 text-sm text-gray-600 dark:text-gray-400 italic prose prose-sm dark:prose-invert"
          v-html="formattedThinking"
        />
      </div>

      <!-- Main content -->
      <div
        v-if="content"
        :class="[
          'prose prose-sm dark:prose-invert max-w-none',
          isUser ? 'prose-invert' : ''
        ]"
        v-html="formattedContent"
      />
      <div v-else-if="isStreaming" class="text-sm text-gray-500 dark:text-gray-400">
        <span class="loading-dots">Thinking</span>
      </div>

      <div
        v-if="timestamp"
        :class="[
          'text-xs mt-1',
          isUser ? 'text-primary-200' : 'text-gray-400 dark:text-gray-500'
        ]"
      >
        {{ timestamp }}
      </div>
    </div>
  </div>
</template>
