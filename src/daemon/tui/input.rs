use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::InputMode;
use super::app::TuiApp;

impl TuiApp {
    pub(crate) fn handle_key(&mut self, key: KeyEvent) {
        // Global shortcuts
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('c') => {
                    if self.generating {
                        self.status_message = "Cancelled".to_string();
                        self.generating = false;
                    } else {
                        self.should_quit = true;
                    }
                    return;
                }
                KeyCode::Char('q') => {
                    self.should_quit = true;
                    return;
                }
                KeyCode::Char('l') => {
                    self.messages.clear();
                    self.add_system_message("Chat cleared.");
                    return;
                }
                KeyCode::Char('m') => {
                    self.input_mode = InputMode::ModelSelect;
                    self.model_select_index = 0;
                    return;
                }
                _ => {}
            }
        }

        // Help toggle
        if key.code == KeyCode::Char('?') && self.input_mode == InputMode::Normal {
            self.show_help = !self.show_help;
            return;
        }

        // Mode-specific handling
        match self.input_mode {
            InputMode::Insert => self.handle_insert_key(key),
            InputMode::Normal => self.handle_normal_key(key),
            InputMode::Command => self.handle_command_key(key),
            InputMode::ModelSelect => self.handle_model_select_key(key),
        }
    }

    fn handle_insert_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter => {
                if !self.input.is_empty() {
                    self.submit_input();
                }
            }
            KeyCode::Char(c) => {
                self.input.insert(self.cursor_pos, c);
                self.cursor_pos += 1;
            }
            KeyCode::Backspace => {
                if self.cursor_pos > 0 {
                    self.cursor_pos -= 1;
                    self.input.remove(self.cursor_pos);
                }
            }
            KeyCode::Delete => {
                if self.cursor_pos < self.input.len() {
                    self.input.remove(self.cursor_pos);
                }
            }
            KeyCode::Left => {
                self.cursor_pos = self.cursor_pos.saturating_sub(1);
            }
            KeyCode::Right => {
                self.cursor_pos = (self.cursor_pos + 1).min(self.input.len());
            }
            KeyCode::Home => self.cursor_pos = 0,
            KeyCode::End => self.cursor_pos = self.input.len(),
            KeyCode::Up => {
                if !self.input_history.is_empty() {
                    let idx = match self.history_index {
                        Some(i) => i.saturating_sub(1),
                        None => self.input_history.len() - 1,
                    };
                    self.history_index = Some(idx);
                    self.input = self.input_history[idx].clone();
                    self.cursor_pos = self.input.len();
                }
            }
            KeyCode::Down => {
                if let Some(idx) = self.history_index {
                    if idx + 1 < self.input_history.len() {
                        self.history_index = Some(idx + 1);
                        self.input = self.input_history[idx + 1].clone();
                    } else {
                        self.history_index = None;
                        self.input.clear();
                    }
                    self.cursor_pos = self.input.len();
                }
            }
            KeyCode::Esc => {
                self.input_mode = InputMode::Normal;
            }
            KeyCode::PageUp => {
                self.messages_scroll = self.messages_scroll.saturating_sub(5);
            }
            KeyCode::PageDown => {
                self.messages_scroll += 5;
            }
            _ => {}
        }
    }

    fn handle_normal_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('i') | KeyCode::Char('a') => {
                self.input_mode = InputMode::Insert;
            }
            KeyCode::Char(':') => {
                self.input_mode = InputMode::Command;
                self.input.clear();
                self.cursor_pos = 0;
            }
            KeyCode::Char('j') | KeyCode::Down => {
                self.messages_scroll += 1;
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.messages_scroll = self.messages_scroll.saturating_sub(1);
            }
            KeyCode::Char('g') => {
                self.messages_scroll = 0;
            }
            KeyCode::Char('G') => {
                self.messages_scroll = self.messages.len().saturating_sub(1);
            }
            KeyCode::Char('q') => {
                self.should_quit = true;
            }
            _ => {}
        }
    }

    fn handle_command_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter => {
                self.execute_command();
                self.input_mode = InputMode::Insert;
            }
            KeyCode::Esc => {
                self.input.clear();
                self.input_mode = InputMode::Insert;
            }
            KeyCode::Char(c) => {
                self.input.push(c);
                self.cursor_pos += 1;
            }
            KeyCode::Backspace => {
                if !self.input.is_empty() {
                    self.input.pop();
                    self.cursor_pos = self.cursor_pos.saturating_sub(1);
                } else {
                    self.input_mode = InputMode::Insert;
                }
            }
            _ => {}
        }
    }

    fn handle_model_select_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.model_select_index = self.model_select_index.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if !self.models.is_empty() {
                    self.model_select_index =
                        (self.model_select_index + 1).min(self.models.len() - 1);
                }
            }
            KeyCode::Enter => {
                if let Some(model) = self.models.get(self.model_select_index) {
                    self.selected_model = Some(model.alias.clone());
                    self.status_message = format!("Selected model: {}", model.alias);
                }
                self.input_mode = InputMode::Insert;
            }
            KeyCode::Esc => {
                self.input_mode = InputMode::Insert;
            }
            _ => {}
        }
    }

    pub(crate) fn execute_command(&mut self) {
        let cmd = std::mem::take(&mut self.input);
        let parts: Vec<&str> = cmd.split_whitespace().collect();

        match parts.first().copied() {
            Some("q") | Some("quit") => {
                self.should_quit = true;
            }
            Some("clear") | Some("c") => {
                self.messages.clear();
                self.add_system_message("Chat cleared.");
            }
            Some("models") | Some("m") => {
                self.refresh_models();
                let list = self
                    .models
                    .iter()
                    .map(|m| format!("{}{}", m.alias, if m.is_default { " *" } else { "" }))
                    .collect::<Vec<_>>()
                    .join(", ");
                self.add_system_message(&format!("Models: {}", list));
            }
            Some("model") => {
                if let Some(alias) = parts.get(1) {
                    self.selected_model = Some(alias.to_string());
                    self.status_message = format!("Selected: {}", alias);
                } else {
                    let current = self.selected_model.as_deref().unwrap_or("default");
                    self.add_system_message(&format!("Current model: {}", current));
                }
            }
            Some("load") => {
                if parts.len() >= 2 {
                    let spec = parts[1..].join(" ");
                    match self.client.load_model(&spec) {
                        Ok((alias, _)) => {
                            self.add_system_message(&format!("Loaded model: {}", alias));
                            self.refresh_models();
                        }
                        Err(e) => {
                            self.add_system_message(&format!("Failed to load: {}", e));
                        }
                    }
                } else {
                    self.add_system_message("Usage: :load <alias:path> or :load <path>");
                }
            }
            Some("unload") => {
                if let Some(alias) = parts.get(1) {
                    match self.client.unload_model(alias) {
                        Ok(()) => {
                            self.add_system_message(&format!("Unloaded: {}", alias));
                            self.refresh_models();
                        }
                        Err(e) => {
                            self.add_system_message(&format!("Failed: {}", e));
                        }
                    }
                }
            }
            Some("temp") | Some("temperature") => {
                if let Some(val) = parts.get(1).and_then(|s| s.parse::<f32>().ok()) {
                    self.temperature = val.clamp(0.0, 2.0);
                    self.status_message = format!("Temperature: {:.1}", self.temperature);
                } else {
                    self.add_system_message(&format!("Temperature: {:.1}", self.temperature));
                }
            }
            Some("tokens") | Some("max") => {
                if let Some(val) = parts.get(1).and_then(|s| s.parse::<u32>().ok()) {
                    self.max_tokens = val;
                    self.status_message = format!("Max tokens: {}", self.max_tokens);
                } else {
                    self.add_system_message(&format!("Max tokens: {}", self.max_tokens));
                }
            }
            Some("status") | Some("s") => {
                self.refresh_status();
                if let Some(ref status) = self.daemon_status {
                    self.add_system_message(&format!(
                        "Daemon v{} | Uptime: {}s | Models: {} | Requests: {}",
                        status.version,
                        status.uptime_secs,
                        status.models_loaded,
                        status.stats.requests_total
                    ));
                }
            }
            Some("help") | Some("h") | Some("?") => {
                self.show_help = true;
            }
            Some(cmd) => {
                self.add_system_message(&format!("Unknown command: {}", cmd));
            }
            None => {}
        }

        self.cursor_pos = 0;
    }
}
