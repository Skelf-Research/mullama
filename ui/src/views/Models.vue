<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useModels } from '@/composables/useModels'
import ModelCard from '@/components/ModelCard.vue'

const { models, localModels, loading, error, fetchModels, fetchLocalModels, pullModel, deleteModel, loadModel, unloadModel } = useModels()

const pullModelName = ref('')
const pullLoading = ref(false)
const pullProgress = ref<string | null>(null)
const pullError = ref<string | null>(null)
const showPullModal = ref(false)
const deleteConfirm = ref<string | null>(null)
const loadingModel = ref<string | null>(null)
const loadError = ref<string | null>(null)

const popularModels = [
  { name: 'llama3.2:1b', description: 'Lightweight, fast model for basic tasks' },
  { name: 'llama3.2:3b', description: 'Balanced performance and quality' },
  { name: 'qwen2.5:7b', description: 'Strong multilingual capabilities' },
  { name: 'deepseek-r1:7b', description: 'Advanced reasoning model' },
  { name: 'mistral:7b', description: 'Excellent for coding tasks' },
  { name: 'phi3:mini', description: 'Microsoft\'s small language model' },
]

onMounted(() => {
  fetchModels()
  fetchLocalModels()
})

const handlePull = async () => {
  if (!pullModelName.value.trim()) return

  pullLoading.value = true
  pullError.value = null
  pullProgress.value = 'Starting download...'

  try {
    await pullModel(pullModelName.value.trim(), (progress) => {
      if (progress.progress && progress.total) {
        const percent = Math.round((progress.progress / progress.total) * 100)
        pullProgress.value = `${progress.status}: ${percent}%`
      } else {
        pullProgress.value = progress.status
      }
    })
    pullModelName.value = ''
    showPullModal.value = false
  } catch (e) {
    pullError.value = e instanceof Error ? e.message : 'Failed to pull model'
  } finally {
    pullLoading.value = false
    pullProgress.value = null
  }
}

const handleDelete = async (name: string) => {
  try {
    await deleteModel(name)
    deleteConfirm.value = null
  } catch (e) {
    // Handle error
  }
}

const quickPull = (name: string) => {
  pullModelName.value = name
  showPullModal.value = true
}

const handleLoad = async (name: string) => {
  loadingModel.value = name
  loadError.value = null
  try {
    await loadModel(name)
  } catch (e) {
    loadError.value = e instanceof Error ? e.message : 'Failed to load model'
  } finally {
    loadingModel.value = null
  }
}

const handleUnload = async (name: string) => {
  loadingModel.value = name
  loadError.value = null
  try {
    await unloadModel(name)
  } catch (e) {
    loadError.value = e instanceof Error ? e.message : 'Failed to unload model'
  } finally {
    loadingModel.value = null
  }
}
</script>

<template>
  <div class="p-6 max-w-7xl mx-auto">
    <div class="flex items-center justify-between mb-6">
      <h1 class="text-2xl font-bold text-gray-900 dark:text-white">Models</h1>
      <button @click="showPullModal = true" class="btn btn-primary">
        <span class="flex items-center gap-2">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          Pull Model
        </span>
      </button>
    </div>

    <!-- Error Alert -->
    <div v-if="error" class="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
      <p class="text-red-700 dark:text-red-400">{{ error }}</p>
    </div>

    <!-- Load Error Alert -->
    <div v-if="loadError" class="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg flex items-center justify-between">
      <p class="text-red-700 dark:text-red-400">{{ loadError }}</p>
      <button @click="loadError = null" class="text-red-500 hover:text-red-700">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>

    <!-- Downloaded/Cached Models -->
    <div class="mb-8">
      <h2 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        Downloaded Models
        <span v-if="localModels.length > 0" class="text-sm font-normal text-gray-500 ml-2">
          ({{ localModels.length }} models)
        </span>
      </h2>
      <div v-if="loading" class="text-gray-500 dark:text-gray-400">Loading...</div>
      <div v-else-if="localModels.length === 0" class="card p-8 text-center">
        <svg class="w-12 h-12 mx-auto text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
        </svg>
        <p class="text-gray-600 dark:text-gray-400">No models downloaded yet</p>
        <p class="text-sm text-gray-500 mt-2">Pull a model from the list below to get started</p>
      </div>
      <div v-else class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div
          v-for="model in localModels"
          :key="model.name"
          class="card p-4"
        >
          <div class="flex items-start justify-between">
            <div class="flex-1 min-w-0">
              <h3 class="font-medium text-gray-900 dark:text-white truncate" :title="model.name">
                {{ model.name }}
              </h3>
              <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">
                {{ model.size_formatted }}
              </p>
              <p v-if="model.repo_id" class="text-xs text-gray-400 dark:text-gray-500 mt-1 truncate" :title="model.repo_id">
                {{ model.repo_id }}
              </p>
            </div>
            <div class="flex items-center gap-1 ml-2">
              <!-- Load/Unload Button -->
              <button
                v-if="model.loaded"
                @click="handleUnload(model.name)"
                :disabled="loadingModel === model.name"
                class="px-2 py-1 text-xs font-medium rounded bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400 hover:bg-orange-200 dark:hover:bg-orange-900/50 transition-colors disabled:opacity-50"
                title="Unload model"
              >
                {{ loadingModel === model.name ? 'Unloading...' : 'Unload' }}
              </button>
              <button
                v-else
                @click="handleLoad(model.name)"
                :disabled="loadingModel === model.name"
                class="px-2 py-1 text-xs font-medium rounded bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 hover:bg-green-200 dark:hover:bg-green-900/50 transition-colors disabled:opacity-50"
                title="Load model"
              >
                {{ loadingModel === model.name ? 'Loading...' : 'Load' }}
              </button>
              <!-- Delete Button -->
              <button
                @click="deleteConfirm = model.name"
                class="p-1.5 text-gray-400 hover:text-red-500 transition-colors"
                title="Delete model"
              >
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Loaded Models (Active in Daemon) -->
    <div v-if="models.length > 0" class="mb-8">
      <h2 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">Active Models</h2>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <ModelCard
          v-for="model in models"
          :key="model.id"
          :name="model.id"
          :status="'loaded'"
          :owned-by="model.owned_by"
        />
      </div>
    </div>

    <!-- Popular Models -->
    <div class="mb-8">
      <h2 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">Popular Models</h2>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div
          v-for="model in popularModels"
          :key="model.name"
          class="card p-4 hover:border-primary-300 dark:hover:border-primary-700 cursor-pointer transition-colors"
          @click="quickPull(model.name)"
        >
          <div class="flex items-start justify-between">
            <div>
              <h3 class="font-medium text-gray-900 dark:text-white">{{ model.name }}</h3>
              <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">{{ model.description }}</p>
            </div>
            <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
          </div>
        </div>
      </div>
    </div>

    <!-- Pull Modal -->
    <Teleport to="body">
      <div v-if="showPullModal" class="fixed inset-0 bg-black/50 flex items-center justify-center z-50" @click.self="showPullModal = false">
        <div class="card p-6 w-full max-w-md mx-4">
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">Pull Model</h3>

          <div class="mb-4">
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Model Name
            </label>
            <input
              v-model="pullModelName"
              type="text"
              class="input"
              placeholder="e.g., llama3.2:1b or hf:meta-llama/Llama-3.2-1B-Instruct-GGUF"
              :disabled="pullLoading"
            />
            <p class="text-xs text-gray-500 mt-1">
              Use short names (llama3.2:1b) or full HuggingFace paths (hf:repo/model)
            </p>
          </div>

          <div v-if="pullProgress" class="mb-4 p-3 bg-gray-100 dark:bg-gray-700 rounded-lg">
            <p class="text-sm text-gray-600 dark:text-gray-400">{{ pullProgress }}</p>
          </div>

          <div v-if="pullError" class="mb-4 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
            <p class="text-sm text-red-600 dark:text-red-400">{{ pullError }}</p>
          </div>

          <div class="flex gap-3">
            <button
              @click="showPullModal = false"
              class="btn btn-secondary flex-1"
              :disabled="pullLoading"
            >
              Cancel
            </button>
            <button
              @click="handlePull"
              class="btn btn-primary flex-1"
              :disabled="pullLoading || !pullModelName.trim()"
            >
              {{ pullLoading ? 'Pulling...' : 'Pull' }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>

    <!-- Delete Confirmation Modal -->
    <Teleport to="body">
      <div v-if="deleteConfirm" class="fixed inset-0 bg-black/50 flex items-center justify-center z-50" @click.self="deleteConfirm = null">
        <div class="card p-6 w-full max-w-md mx-4">
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">Delete Model</h3>
          <p class="text-gray-600 dark:text-gray-400 mb-6">
            Are you sure you want to delete <strong>{{ deleteConfirm }}</strong>? This action cannot be undone.
          </p>
          <div class="flex gap-3">
            <button @click="deleteConfirm = null" class="btn btn-secondary flex-1">
              Cancel
            </button>
            <button @click="handleDelete(deleteConfirm!)" class="btn btn-danger flex-1">
              Delete
            </button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>
