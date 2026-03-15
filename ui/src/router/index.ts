import { createRouter, createWebHistory } from 'vue-router'
import Dashboard from '@/views/Dashboard.vue'
import Models from '@/views/Models.vue'
import Chat from '@/views/Chat.vue'
import Playground from '@/views/Playground.vue'
import Settings from '@/views/Settings.vue'

const router = createRouter({
  history: createWebHistory('/ui/'),
  routes: [
    {
      path: '/',
      name: 'dashboard',
      component: Dashboard,
    },
    {
      path: '/models',
      name: 'models',
      component: Models,
    },
    {
      path: '/chat',
      name: 'chat',
      component: Chat,
    },
    {
      path: '/playground',
      name: 'playground',
      component: Playground,
    },
    {
      path: '/settings',
      name: 'settings',
      component: Settings,
    },
  ],
})

export default router
