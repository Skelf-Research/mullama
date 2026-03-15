import { ref, reactive } from 'vue'
import { openai, type ChatMessage, type ChatResponse } from '@/api/client'

export interface Message extends ChatMessage {
  id: string
  timestamp: number
  isStreaming?: boolean
  thinking?: string
}

export interface Conversation {
  id: string
  title: string
  messages: Message[]
  model: string
  createdAt: number
  updatedAt: number
}

const conversations = ref<Conversation[]>([])
const activeConversationId = ref<string | null>(null)
const isGenerating = ref(false)

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
}

export function useChat() {
  const activeConversation = () => {
    return conversations.value.find(c => c.id === activeConversationId.value)
  }

  const createConversation = (model: string): Conversation => {
    const conversation: Conversation = {
      id: generateId(),
      title: 'New Chat',
      messages: [],
      model,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    }
    conversations.value.unshift(conversation)
    activeConversationId.value = conversation.id
    return conversation
  }

  const deleteConversation = (id: string) => {
    const index = conversations.value.findIndex(c => c.id === id)
    if (index !== -1) {
      conversations.value.splice(index, 1)
      if (activeConversationId.value === id) {
        activeConversationId.value = conversations.value[0]?.id ?? null
      }
    }
  }

  const addMessage = (
    conversationId: string,
    role: Message['role'],
    content: string
  ): Message => {
    const conversation = conversations.value.find(c => c.id === conversationId)
    if (!conversation) throw new Error('Conversation not found')

    const message: Message = {
      id: generateId(),
      role,
      content,
      timestamp: Date.now(),
    }
    conversation.messages.push(message)
    conversation.updatedAt = Date.now()

    // Update title from first user message
    if (role === 'user' && conversation.messages.filter(m => m.role === 'user').length === 1) {
      conversation.title = content.slice(0, 50) + (content.length > 50 ? '...' : '')
    }

    return message
  }

  const sendMessage = async (
    conversationId: string,
    content: string,
    options?: {
      stream?: boolean
      maxTokens?: number
      temperature?: number
    }
  ) => {
    const conversation = conversations.value.find(c => c.id === conversationId)
    if (!conversation) throw new Error('Conversation not found')

    // Add user message
    addMessage(conversationId, 'user', content)

    // Create assistant message placeholder
    const assistantMessage = addMessage(conversationId, 'assistant', '')
    assistantMessage.isStreaming = true

    isGenerating.value = true

    try {
      const messages = conversation.messages
        .filter(m => m.id !== assistantMessage.id)
        .map(m => ({ role: m.role, content: m.content }))

      if (options?.stream !== false) {
        // Streaming response with callback
        const messageId = assistantMessage.id

        await openai.chatStream(
          {
            model: conversation.model,
            messages,
            max_tokens: options?.maxTokens ?? 1024,
            temperature: options?.temperature ?? 0.7,
            stream: true,
          },
          (chunk, thinking) => {
            // Re-find conversation and message to ensure Vue reactivity
            const conv = conversations.value.find(c => c.id === conversationId)
            if (conv) {
              const msg = conv.messages.find(m => m.id === messageId)
              if (msg) {
                if (thinking) {
                  msg.thinking = (msg.thinking || '') + thinking
                } else if (chunk) {
                  msg.content += chunk
                }
              }
            }
          }
        )
      } else {
        // Non-streaming response
        const response = await openai.chat({
          model: conversation.model,
          messages,
          max_tokens: options?.maxTokens ?? 1024,
          temperature: options?.temperature ?? 0.7,
          stream: false,
        })

        assistantMessage.content = response.choices[0]?.message?.content ?? ''
      }
    } catch (error) {
      assistantMessage.content = `Error: ${error instanceof Error ? error.message : 'Unknown error'}`
    } finally {
      assistantMessage.isStreaming = false
      isGenerating.value = false
    }
  }

  const clearConversation = (conversationId: string) => {
    const conversation = conversations.value.find(c => c.id === conversationId)
    if (conversation) {
      conversation.messages = []
      conversation.updatedAt = Date.now()
    }
  }

  return {
    conversations,
    activeConversationId,
    isGenerating,
    activeConversation,
    createConversation,
    deleteConversation,
    addMessage,
    sendMessage,
    clearConversation,
  }
}
