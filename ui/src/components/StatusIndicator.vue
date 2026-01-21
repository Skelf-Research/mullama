<script setup lang="ts">
defineProps<{
  status: 'online' | 'offline' | 'loading'
}>()
</script>

<template>
  <div class="flex items-center gap-2">
    <span
      :class="[
        'w-2 h-2 rounded-full',
        status === 'online' && 'bg-green-500',
        status === 'offline' && 'bg-red-500',
        status === 'loading' && 'bg-yellow-500 animate-pulse',
      ]"
    />
    <span class="text-xs text-gray-500 dark:text-gray-400 capitalize">
      {{ status }}
    </span>
  </div>
</template>
