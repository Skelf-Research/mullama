import { buildHeaders } from './auth'

export const metrics = {
  async get(): Promise<string> {
    const response = await fetch('/metrics', {
      headers: buildHeaders(),
    })
    return response.text()
  },
}
