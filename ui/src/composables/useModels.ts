import { ref, computed } from 'vue'
import { openai, management, type Model, type ModelDetails } from '@/api/client'

const models = ref<Model[]>([])
const localModels = ref<ModelDetails[]>([])
const loading = ref(false)
const error = ref<string | null>(null)
const selectedModel = ref<string | null>(null)

export function useModels() {
  const fetchModels = async () => {
    loading.value = true
    error.value = null
    try {
      models.value = await openai.listModels()
      if (models.value.length > 0 && !selectedModel.value) {
        selectedModel.value = models.value[0].id
      }
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to fetch models'
    } finally {
      loading.value = false
    }
  }

  const fetchLocalModels = async () => {
    loading.value = true
    error.value = null
    try {
      localModels.value = await management.listModels()
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to fetch local models'
    } finally {
      loading.value = false
    }
  }

  const pullModel = async (name: string, onProgress?: (p: any) => void) => {
    try {
      await management.pullModel(name, onProgress)
      await fetchLocalModels()
    } catch (e) {
      throw e
    }
  }

  const deleteModel = async (name: string) => {
    try {
      await management.deleteModel(name)
      await fetchLocalModels()
    } catch (e) {
      throw e
    }
  }

  const loadModel = async (name: string, options?: { gpu_layers?: number; context_size?: number }) => {
    try {
      const result = await management.loadModel(name, options)
      await fetchModels()
      await fetchLocalModels()
      return result
    } catch (e) {
      throw e
    }
  }

  const unloadModel = async (name: string) => {
    try {
      const result = await management.unloadModel(name)
      await fetchModels()
      await fetchLocalModels()
      return result
    } catch (e) {
      throw e
    }
  }

  const modelNames = computed(() => models.value.map(m => m.id))

  return {
    models,
    localModels,
    loading,
    error,
    selectedModel,
    modelNames,
    fetchModels,
    fetchLocalModels,
    pullModel,
    deleteModel,
    loadModel,
    unloadModel,
  }
}
