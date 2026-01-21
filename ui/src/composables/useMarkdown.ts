import { marked } from 'marked'
import hljs from 'highlight.js'
import DOMPurify from 'dompurify'

// Configure marked with highlight.js for syntax highlighting
const renderer = new marked.Renderer()

renderer.code = function({ text, lang }: { text: string; lang?: string }) {
  const language = lang && hljs.getLanguage(lang) ? lang : 'plaintext'
  const highlighted = hljs.highlight(text, { language }).value
  return `<pre><code class="hljs language-${language}">${highlighted}</code></pre>`
}

marked.use({
  renderer,
  breaks: true,  // Convert \n to <br>
  gfm: true,     // GitHub Flavored Markdown
})

export function useMarkdown() {
  const renderMarkdown = (content: string): string => {
    if (!content) return ''

    // Parse markdown
    const html = marked.parse(content, { async: false }) as string

    // Sanitize to prevent XSS
    return DOMPurify.sanitize(html, {
      ALLOWED_TAGS: [
        'p', 'br', 'strong', 'em', 'code', 'pre', 'ul', 'ol', 'li',
        'blockquote', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'span',
        'table', 'thead', 'tbody', 'tr', 'th', 'td', 'hr', 'del', 'sup', 'sub'
      ],
      ALLOWED_ATTR: ['href', 'class', 'target', 'rel'],
    })
  }

  return { renderMarkdown }
}
