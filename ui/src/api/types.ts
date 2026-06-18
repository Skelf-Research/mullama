export interface Model {
  id: string
  object: string
  created: number
  owned_by: string
}

export interface ModelsResponse {
  object: string
  data: Model[]
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export interface ChatRequest {
  model: string
  messages: ChatMessage[]
  stream?: boolean
  max_tokens?: number
  temperature?: number
}

export interface ChatChoice {
  index: number
  message: ChatMessage
  finish_reason: string
}

export interface ChatResponse {
  id: string
  object: string
  created: number
  model: string
  choices: ChatChoice[]
  usage?: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
  }
}

export interface SystemStatus {
  uptime_secs: number
  version: string
  models_loaded: number
  http_endpoint?: string
}

export interface ModelDetails {
  name: string
  filename: string
  path: string
  repo_id?: string
  size: number
  size_formatted: string
  loaded: boolean
  downloaded?: string
  source?: string
}

export interface PullProgress {
  status: string
  progress?: number
  total?: number
  speed?: string
}

export interface DefaultModel {
  name: string
  description: string
  size_hint: string
  tags: string[]
  from: string
  has_thinking: boolean
  has_vision: boolean
  has_tools: boolean
}

export interface DefaultsResponse {
  models: DefaultModel[]
}
