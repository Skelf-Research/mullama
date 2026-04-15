import { onMounted, ref } from 'vue'
import { management, type DefaultModel } from '@/api/client'
import { useModels } from '@/composables/useModels'

export function useModelManagement() {
  const {
    models,
    localModels,
    loading,
    error,
    fetchModels,
    fetchLocalModels,
    pullModel,
    deleteModel,
    loadModel,
    unloadModel,
  } = useModels()

  const pullModelName = ref('')
  const pullLoading = ref(false)
  const pullProgress = ref<string | null>(null)
  const pullError = ref<string | null>(null)
  const showPullModal = ref(false)
  const deleteConfirm = ref<string | null>(null)
  const loadingModel = ref<string | null>(null)
  const loadError = ref<string | null>(null)

  const defaultModels = ref<DefaultModel[]>([])
  const defaultsLoading = ref(false)
  const usingDefault = ref<string | null>(null)

  const fetchDefaults = async () => {
    defaultsLoading.value = true
    try {
      defaultModels.value = await management.listDefaults()
    } catch (e) {
      console.error('Failed to fetch defaults:', e)
    } finally {
      defaultsLoading.value = false
    }
  }

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
    } catch {
      // Let the shared models error state surface the failure.
    }
  }

  const handleUseDefault = async (name: string) => {
    usingDefault.value = name
    loadError.value = null
    try {
      const result = await management.useDefault(name)
      if (!result.success) {
        loadError.value = result.message
      } else {
        await fetchModels()
        await fetchLocalModels()
      }
    } catch (e) {
      loadError.value = e instanceof Error ? e.message : 'Failed to load model'
    } finally {
      usingDefault.value = null
    }
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

  onMounted(() => {
    fetchModels()
    fetchLocalModels()
    fetchDefaults()
  })

  return {
    models,
    localModels,
    loading,
    error,
    pullModelName,
    pullLoading,
    pullProgress,
    pullError,
    showPullModal,
    deleteConfirm,
    loadingModel,
    loadError,
    defaultModels,
    defaultsLoading,
    usingDefault,
    handlePull,
    handleDelete,
    handleUseDefault,
    handleLoad,
    handleUnload,
  }
}
