---
title: TUI Chat
description: ratatui-based terminal UI for interactive chat with local models
---

# TUI Chat

Mullama includes a feature-rich terminal UI (TUI) for interactive chat, built with [ratatui](https://ratatui.rs/). The TUI provides a full-featured chat experience directly in your terminal with streaming responses, multi-model switching, and conversation management.

## Overview

The TUI connects to the Mullama daemon via IPC for low-latency communication and provides:

- **Streaming output** -- Tokens appear as they are generated
- **Multi-model switching** -- Switch between loaded models mid-conversation
- **Conversation history** -- Maintain and navigate multiple conversations
- **Session persistence** -- Resume previous conversations
- **Markdown rendering** -- Rich text formatting in the terminal
- **Keyboard-driven** -- Efficient navigation with keyboard shortcuts
- **Command mode** -- In-chat commands for model and session management

## Launching the TUI

```bash
# Launch with default model
mullama chat

# Launch with specific model
mullama chat --model deepseek-r1:7b

# With custom system prompt
mullama chat --model qwen2.5:7b --system "You are a helpful coding assistant"

# Connect to non-default daemon
mullama chat --socket ipc:///var/run/mullama.sock

# With connection timeout
mullama chat --timeout 15
```

!!! tip "Auto-Spawn"
    If the daemon is not running, `mullama chat` will automatically start it before connecting.

## Interface Layout

```
+------------------------------------------------------------------+
|  Mullama TUI v0.1.1  |  Model: llama3.2:1b  |  GPU: 35 layers   |
+------------------------------------------------------------------+
|                                                                    |
|  User:                                                             |
|  What is the capital of France?                                    |
|                                                                    |
|  Assistant:                                                        |
|  The capital of France is Paris. It is the largest city in         |
|  France and serves as the country's political, economic, and       |
|  cultural center.                                                  |
|                                                                    |
|  User:                                                             |
|  Tell me more about its history.                                   |
|                                                                    |
|  Assistant:                                                        |
|  Paris has a rich history spanning over 2,000 years...             |
|  [streaming...]                                                    |
|                                                                    |
+------------------------------------------------------------------+
|  > Type your message...                                    [Enter] |
+------------------------------------------------------------------+
|  /help  |  /model  |  /new  |  /quit       Tokens: 245  |  35t/s |
+------------------------------------------------------------------+
```

The interface consists of:

1. **Header bar** -- Daemon version, current model, GPU status
2. **Chat area** -- Scrollable conversation history with streaming responses
3. **Input area** -- Multi-line text input
4. **Status bar** -- Available commands, token count, generation speed

---

## Keyboard Shortcuts

### Navigation

| Key | Action |
|-----|--------|
| ++up++ / ++down++ | Scroll through chat history |
| ++page-up++ / ++page-down++ | Scroll one page at a time |
| ++home++ / ++end++ | Jump to top/bottom of conversation |
| ++tab++ | Switch between input and chat panels |

### Input

| Key | Action |
|-----|--------|
| ++enter++ | Send message |
| ++shift+enter++ | New line in input (multi-line mode) |
| ++ctrl+u++ | Clear current input |
| ++ctrl+w++ | Delete word before cursor |
| ++ctrl+a++ / ++ctrl+e++ | Move cursor to start/end of line |
| ++left++ / ++right++ | Move cursor within input |

### Commands

| Key | Action |
|-----|--------|
| ++ctrl+n++ | New conversation |
| ++ctrl+s++ | Save current session |
| ++ctrl+l++ | Clear screen |
| ++ctrl+c++ | Stop current generation |
| ++ctrl+d++ / ++esc++ | Quit TUI |

### Model

| Key | Action |
|-----|--------|
| ++ctrl+m++ | Open model selector |
| ++ctrl+p++ | Toggle parameter panel |

---

## Command Mode

Type `/` at the beginning of the input to enter command mode. Available commands:

### General Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/quit` or `/exit` | Exit the TUI |
| `/clear` | Clear the current conversation |
| `/new` | Start a new conversation |

### Model Commands

| Command | Description |
|---------|-------------|
| `/model` | Show current model |
| `/model <name>` | Switch to a different model |
| `/models` | List all loaded models |
| `/load <spec>` | Load a new model into the daemon |
| `/unload <name>` | Unload a model |

### Session Commands

| Command | Description |
|---------|-------------|
| `/save [name]` | Save current conversation |
| `/load-session <name>` | Load a saved conversation |
| `/sessions` | List saved sessions |
| `/delete-session <name>` | Delete a saved session |
| `/export [path]` | Export conversation as markdown |

### Parameter Commands

| Command | Description |
|---------|-------------|
| `/temp <value>` | Set temperature (0.0-2.0) |
| `/max-tokens <n>` | Set max tokens |
| `/top-p <value>` | Set top_p |
| `/system <prompt>` | Set or change system prompt |
| `/params` | Show current parameters |

### Utility Commands

| Command | Description |
|---------|-------------|
| `/status` | Show daemon status |
| `/stats` | Show generation statistics |
| `/copy` | Copy last response to clipboard |
| `/regenerate` | Regenerate the last response |
| `/edit` | Edit the last user message and regenerate |

**Examples:**

```
/model deepseek-r1:7b
/temp 0.9
/system You are a creative writing assistant
/save my-conversation
/export ./chat-history.md
```

---

## Model Switching

Switch between loaded models without leaving the TUI:

```
/model qwen2.5:7b
```

Or use ++ctrl+m++ to open the model selector popup:

```
+---------------------------+
|  Select Model             |
|                           |
|  > llama3.2:1b   [1.2G]  |
|    qwen2.5:7b    [7.6G]  |
|    deepseek-r1:7b [4.9G] |
|                           |
|  [Enter] Select  [Esc] Cancel
+---------------------------+
```

!!! note "Conversation Continuity"
    When switching models, the conversation history is preserved. The new model will have context of the full conversation so far.

---

## Session Persistence

### Auto-Save

By default, conversations are automatically saved when you exit the TUI. They are stored in:

| Platform | Path |
|----------|------|
| Linux | `~/.mullama/sessions/` |
| macOS | `~/.mullama/sessions/` |
| Windows | `%USERPROFILE%\.mullama\sessions\` |

### Manual Save/Load

```
# Save with a name
/save project-discussion

# List available sessions
/sessions

# Load a previous session
/load-session project-discussion

# Delete a session
/delete-session old-conversation
```

### Export

Export conversations as markdown files:

```
# Export to default path (~/.mullama/exports/)
/export

# Export to specific path
/export ./my-conversation.md
```

**Exported format:**

```markdown
# Chat Session - 2025-01-23 14:30

**Model:** llama3.2:1b
**System:** You are a helpful assistant.

---

**User:** What is the capital of France?

**Assistant:** The capital of France is Paris.

---

**User:** Tell me more about it.

**Assistant:** Paris is the largest city in France...
```

---

## Streaming Display

Responses stream token-by-token with visual feedback:

- **Cursor animation** while generating
- **Token counter** updates in real-time in the status bar
- **Speed indicator** shows tokens/second
- **Stop button** (++ctrl+c++) halts generation immediately

For models with thinking tokens (e.g., DeepSeek-R1), the thinking process is displayed in a dimmed/collapsed section:

```
Assistant:
  [Thinking...] (press Tab to expand)
  Let me work through this step by step...
  First, I need to consider...

  The answer is 42.
```

---

## Multi-Line Input

For longer messages, use ++shift+enter++ to create new lines:

```
> Write a Python function that:
> - Takes a list of numbers
> - Returns the top 3 largest
> - Handles edge cases
[Enter to send]
```

---

## Configuration

### TUI-Specific Settings

The TUI respects the following configuration options (via config file or environment):

| Setting | Default | Description |
|---------|---------|-------------|
| `tui.theme` | `auto` | Color theme (auto, dark, light) |
| `tui.show_thinking` | `true` | Show thinking tokens for reasoning models |
| `tui.auto_save` | `true` | Auto-save sessions on exit |
| `tui.history_limit` | `1000` | Maximum messages per session |
| `tui.scroll_speed` | `3` | Lines to scroll per mouse wheel tick |
| `tui.word_wrap` | `true` | Wrap long lines in chat display |
| `tui.code_highlight` | `true` | Syntax highlight code blocks |
| `tui.timestamp` | `false` | Show timestamps on messages |

### Color Themes

The TUI adapts to your terminal's color scheme by default. Override with:

```bash
# Force dark theme
mullama chat --theme dark

# Force light theme
mullama chat --theme light
```

---

## Tips and Tricks

### Efficient Workflow

1. **Start with a system prompt** to set context:
   ```
   /system You are an expert Python developer. Provide concise, production-ready code.
   ```

2. **Use model switching** for different tasks:
   ```
   /model deepseek-r1:7b    # For reasoning problems
   /model qwen2.5-coder:7b  # For code generation
   /model llama3.2:1b       # For quick questions
   ```

3. **Save important conversations** for reference:
   ```
   /save architecture-discussion
   ```

4. **Regenerate** if the response is not satisfactory:
   ```
   /regenerate
   ```

### Terminal Compatibility

The TUI works best in terminals that support:

- 256 colors or true color
- Unicode characters
- Mouse input (optional, for scrolling)

**Recommended terminals:**

- **Linux:** Alacritty, Kitty, WezTerm, GNOME Terminal
- **macOS:** iTerm2, Alacritty, Terminal.app
- **Windows:** Windows Terminal, WezTerm

### Troubleshooting

!!! warning "Screen Rendering Issues"
    If the TUI renders incorrectly, try:

    - Resize your terminal window
    - Ensure your terminal supports UTF-8: `echo $LANG`
    - Set `TERM=xterm-256color` if colors are missing
    - Use `--theme dark` or `--theme light` to force a compatible theme

!!! info "Connection Issues"
    If the TUI cannot connect to the daemon:

    - Check if the daemon is running: `mullama daemon status`
    - Verify the socket path: `ls /tmp/mullama.sock`
    - Try increasing the timeout: `mullama chat --timeout 30`
    - Start the daemon manually: `mullama daemon start`
