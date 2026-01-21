<script setup lang="ts">
import { ref, computed, nextTick, onMounted, watch } from 'vue'
import { useChat, type Conversation } from '@/composables/useChat'
import { useModels } from '@/composables/useModels'
import ChatMessage from '@/components/ChatMessage.vue'

const {
  conversations,
  activeConversationId,
  isGenerating,
  activeConversation,
  createConversation,
  deleteConversation,
  sendMessage,
  clearConversation,
} = useChat()

const { models, selectedModel, fetchModels } = useModels()

const inputText = ref('')
const messagesContainer = ref<HTMLElement | null>(null)
const inputRef = ref<HTMLTextAreaElement | null>(null)

onMounted(() => {
  fetchModels()
})

const currentConversation = computed(() => activeConversation())

const handleNewChat = () => {
  if (selectedModel.value) {
    createConversation(selectedModel.value)
    inputText.value = ''
    inputRef.value?.focus()
  }
}

const handleSend = async () => {
  if (!inputText.value.trim() || isGenerating.value) return

  let convId = activeConversationId.value
  if (!convId && selectedModel.value) {
    const conv = createConversation(selectedModel.value)
    convId = conv.id
  }

  if (!convId) return

  const text = inputText.value.trim()
  inputText.value = ''

  await sendMessage(convId, text, { stream: true })

  // Scroll to bottom after message
  await nextTick()
  scrollToBottom()
}

const scrollToBottom = () => {
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

// Auto-scroll when messages change
watch(
  () => currentConversation.value?.messages.length,
  () => {
    nextTick(scrollToBottom)
  }
)

const handleKeydown = (e: KeyboardEvent) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    handleSend()
  }
}

const formatTime = (timestamp: number) => {
  return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}
</script>

<template>
  <div class="flex h-full">
    <!-- Sidebar - Conversations -->
    <div class="w-64 border-r border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 flex flex-col">
      <div class="p-4 border-b border-gray-200 dark:border-gray-700">
        <button @click="handleNewChat" class="btn btn-primary w-full">
          <span class="flex items-center justify-center gap-2">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
            </svg>
            New Chat
          </span>
        </button>
      </div>

      <div class="flex-1 overflow-y-auto p-2">
        <div
          v-for="conv in conversations"
          :key="conv.id"
          :class="[
            'p-3 rounded-lg cursor-pointer mb-1 transition-colors',
            activeConversationId === conv.id
              ? 'bg-primary-50 dark:bg-primary-900/20'
              : 'hover:bg-gray-100 dark:hover:bg-gray-700'
          ]"
          @click="activeConversationId = conv.id"
        >
          <div class="flex items-start justify-between">
            <div class="flex-1 min-w-0">
              <p class="text-sm font-medium text-gray-900 dark:text-white truncate">
                {{ conv.title }}
              </p>
              <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
                {{ conv.model }}
              </p>
            </div>
            <button
              @click.stop="deleteConversation(conv.id)"
              class="p-1 text-gray-400 hover:text-red-500 transition-colors"
            >
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        <div v-if="conversations.length === 0" class="p-4 text-center text-gray-500 dark:text-gray-400 text-sm">
          No conversations yet
        </div>
      </div>
    </div>

    <!-- Main Chat Area -->
    <div class="flex-1 flex flex-col">
      <!-- Header -->
      <div class="h-14 border-b border-gray-200 dark:border-gray-700 px-4 flex items-center justify-between bg-white dark:bg-gray-800">
        <div class="flex items-center gap-3">
          <span class="text-gray-900 dark:text-white font-medium">
            {{ currentConversation?.title || 'New Chat' }}
          </span>
        </div>

        <div class="flex items-center gap-2">
          <select
            v-model="selectedModel"
            class="input py-1 px-2 text-sm w-48"
          >
            <option v-for="model in models" :key="model.id" :value="model.id">
              {{ model.id }}
            </option>
          </select>

          <button
            v-if="currentConversation"
            @click="clearConversation(currentConversation.id)"
            class="p-2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
            title="Clear conversation"
          >
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>
        </div>
      </div>

      <!-- Messages -->
      <div ref="messagesContainer" class="flex-1 overflow-y-auto p-4 space-y-4">
        <template v-if="currentConversation?.messages.length">
          <ChatMessage
            v-for="message in currentConversation.messages"
            :key="message.id"
            :role="message.role"
            :content="message.content"
            :thinking="message.thinking"
            :timestamp="formatTime(message.timestamp)"
            :is-streaming="message.isStreaming"
          />
        </template>
        <div v-else class="flex flex-col items-center justify-center h-full text-gray-500 dark:text-gray-400">
          <svg class="w-16 h-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
          <p class="text-lg font-medium">Start a conversation</p>
          <p class="text-sm mt-1">Type a message below to begin</p>
        </div>
      </div>

      <!-- Input -->
      <div class="border-t border-gray-200 dark:border-gray-700 p-4 bg-white dark:bg-gray-800">
        <div class="flex gap-3">
          <textarea
            ref="inputRef"
            v-model="inputText"
            @keydown="handleKeydown"
            class="input resize-none"
            :class="{ 'opacity-50': isGenerating }"
            rows="1"
            placeholder="Type your message..."
            :disabled="isGenerating || models.length === 0"
          />
          <button
            @click="handleSend"
            class="btn btn-primary px-4"
            :disabled="!inputText.trim() || isGenerating || models.length === 0"
          >
            <svg v-if="isGenerating" class="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <svg v-else class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </button>
        </div>
        <p v-if="models.length === 0" class="text-xs text-yellow-600 dark:text-yellow-400 mt-2">
          No models available. Start the daemon with a model first.
        </p>
      </div>
    </div>
  </div>
</template>
