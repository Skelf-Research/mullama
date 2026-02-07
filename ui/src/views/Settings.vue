<script setup lang="ts">
import { ref, onMounted } from 'vue'

const settings = ref({
  defaultModel: '',
  temperature: 0.7,
  maxTokens: 1024,
  contextSize: 4096,
  gpuLayers: 0,
  theme: 'system' as 'light' | 'dark' | 'system',
})

const saved = ref(false)

onMounted(() => {
  // Load settings from localStorage
  const stored = localStorage.getItem('mullama-settings')
  if (stored) {
    try {
      const parsed = JSON.parse(stored)
      settings.value = { ...settings.value, ...parsed }
    } catch {
      // Ignore parse errors
    }
  }

  // Check current theme
  if (document.documentElement.classList.contains('dark')) {
    settings.value.theme = 'dark'
  } else {
    settings.value.theme = 'light'
  }
})

const saveSettings = () => {
  localStorage.setItem('mullama-settings', JSON.stringify(settings.value))

  // Apply theme
  if (settings.value.theme === 'dark') {
    document.documentElement.classList.add('dark')
  } else if (settings.value.theme === 'light') {
    document.documentElement.classList.remove('dark')
  } else {
    // System preference
    if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }

  saved.value = true
  setTimeout(() => {
    saved.value = false
  }, 2000)
}

const resetSettings = () => {
  settings.value = {
    defaultModel: '',
    temperature: 0.7,
    maxTokens: 1024,
    contextSize: 4096,
    gpuLayers: 0,
    theme: 'system',
  }
  localStorage.removeItem('mullama-settings')
}
</script>

<template>
  <div class="p-6 max-w-3xl mx-auto">
    <h1 class="text-2xl font-bold text-gray-900 dark:text-white mb-6">Settings</h1>

    <!-- Success Message -->
    <div
      v-if="saved"
      class="mb-6 p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg"
    >
      <p class="text-green-700 dark:text-green-400">Settings saved successfully!</p>
    </div>

    <!-- Appearance -->
    <div class="card p-6 mb-6">
      <h2 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">Appearance</h2>

      <div class="mb-4">
        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Theme
        </label>
        <select v-model="settings.theme" class="input">
          <option value="system">System</option>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
      </div>
    </div>

    <!-- Generation Defaults -->
    <div class="card p-6 mb-6">
      <h2 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">Generation Defaults</h2>

      <div class="space-y-4">
        <div>
          <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Default Model
          </label>
          <input
            v-model="settings.defaultModel"
            type="text"
            class="input"
            placeholder="e.g., llama3.2:1b"
          />
          <p class="text-xs text-gray-500 mt-1">Leave empty to use the first available model</p>
        </div>

        <div>
          <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Temperature: {{ settings.temperature }}
          </label>
          <input
            v-model.number="settings.temperature"
            type="range"
            min="0"
            max="2"
            step="0.1"
            class="w-full"
          />
          <div class="flex justify-between text-xs text-gray-500">
            <span>Deterministic (0)</span>
            <span>Creative (2)</span>
          </div>
        </div>

        <div>
          <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Max Tokens
          </label>
          <input
            v-model.number="settings.maxTokens"
            type="number"
            min="1"
            max="32768"
            class="input"
          />
        </div>
      </div>
    </div>

    <!-- Model Settings -->
    <div class="card p-6 mb-6">
      <h2 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">Model Settings</h2>

      <div class="space-y-4">
        <div>
          <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Context Size
          </label>
          <select v-model.number="settings.contextSize" class="input">
            <option :value="2048">2048</option>
            <option :value="4096">4096</option>
            <option :value="8192">8192</option>
            <option :value="16384">16384</option>
            <option :value="32768">32768</option>
          </select>
        </div>

        <div>
          <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            GPU Layers
          </label>
          <input
            v-model.number="settings.gpuLayers"
            type="number"
            min="0"
            max="100"
            class="input"
          />
          <p class="text-xs text-gray-500 mt-1">0 = CPU only, higher = more GPU offloading</p>
        </div>
      </div>
    </div>

    <!-- Actions -->
    <div class="flex gap-3">
      <button @click="saveSettings" class="btn btn-primary">
        Save Settings
      </button>
      <button @click="resetSettings" class="btn btn-secondary">
        Reset to Defaults
      </button>
    </div>

    <!-- Info -->
    <div class="mt-8 p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
      <h3 class="font-medium text-gray-900 dark:text-white mb-2">About Mullama</h3>
      <p class="text-sm text-gray-600 dark:text-gray-400 mb-2">
        Mullama is a multi-model LLM server with OpenAI and Anthropic-compatible APIs.
      </p>
      <div class="flex gap-4 text-sm">
        <a
          href="https://github.com/skelf-research/mullama"
          target="_blank"
          class="text-primary-600 hover:text-primary-700 dark:text-primary-400"
        >
          GitHub
        </a>
        <a
          href="https://docs.skelfresearch.com/mullama"
          target="_blank"
          class="text-primary-600 hover:text-primary-700 dark:text-primary-400"
        >
          Documentation
        </a>
      </div>
    </div>
  </div>
</template>
