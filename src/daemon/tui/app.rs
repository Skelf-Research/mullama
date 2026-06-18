use std::io::{self, Stdout};
use std::time::Duration;

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, widgets::ScrollbarState, Terminal};

use super::{InputMode, Message, Role};
use crate::daemon::client::DaemonClient;
use crate::daemon::protocol::{DaemonStatus, ModelStatus};

/// TUI Application
pub struct TuiApp {
    pub(crate) client: DaemonClient,

    // State
    pub(crate) messages: Vec<Message>,
    pub(crate) input: String,
    pub(crate) cursor_pos: usize,
    pub(crate) input_mode: InputMode,

    // Scrolling
    pub(crate) messages_scroll: usize,
    pub(crate) messages_scroll_state: ScrollbarState,

    // Models
    pub(crate) models: Vec<ModelStatus>,
    pub(crate) selected_model: Option<String>,
    pub(crate) model_select_index: usize,

    // Status
    pub(crate) daemon_status: Option<DaemonStatus>,
    pub(crate) status_message: String,
    pub(crate) last_error: Option<String>,

    // Generation params
    pub(crate) max_tokens: u32,
    pub(crate) temperature: f32,

    // Flags
    pub(crate) generating: bool,
    pub(crate) should_quit: bool,
    pub(crate) show_help: bool,

    // History
    pub(crate) input_history: Vec<String>,
    pub(crate) history_index: Option<usize>,
}

impl TuiApp {
    pub fn new(client: DaemonClient) -> Self {
        Self {
            client,
            messages: vec![],
            input: String::new(),
            cursor_pos: 0,
            input_mode: InputMode::Insert,
            messages_scroll: 0,
            messages_scroll_state: ScrollbarState::default(),
            models: vec![],
            selected_model: None,
            model_select_index: 0,
            daemon_status: None,
            status_message: String::new(),
            last_error: None,
            max_tokens: 512,
            temperature: 0.7,
            generating: false,
            should_quit: false,
            show_help: false,
            input_history: vec![],
            history_index: None,
        }
    }

    /// Run the TUI
    pub fn run(&mut self) -> io::Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Initial data fetch
        self.refresh_status();
        self.refresh_models();

        // Welcome message
        self.add_system_message("Welcome to Mullama! Type a message to chat, or press ? for help.");

        let result = self.event_loop(&mut terminal);

        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        result
    }

    pub(crate) fn event_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    ) -> io::Result<()> {
        loop {
            terminal.draw(|f| self.draw(f))?;

            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    self.handle_key(key);
                }
            }

            if self.should_quit {
                break;
            }
        }
        Ok(())
    }

    pub(crate) fn add_system_message(&mut self, content: &str) {
        self.messages.push(Message {
            role: Role::System,
            content: content.to_string(),
            model: None,
            tokens: None,
            duration_ms: None,
        });
    }

    pub(crate) fn refresh_status(&mut self) {
        if let Ok(status) = self.client.status() {
            self.daemon_status = Some(status);
        }
    }

    pub(crate) fn refresh_models(&mut self) {
        if let Ok(models) = self.client.list_models() {
            self.models = models;
            // Set selected model to default if not set
            if self.selected_model.is_none() {
                self.selected_model = self
                    .models
                    .iter()
                    .find(|m| m.is_default)
                    .map(|m| m.alias.clone());
            }
        }
    }

    pub(crate) fn submit_input(&mut self) {
        let input = std::mem::take(&mut self.input);
        self.cursor_pos = 0;
        self.history_index = None;

        // Add to history
        if !input.is_empty() {
            self.input_history.push(input.clone());
        }

        // Add user message
        self.messages.push(Message {
            role: Role::User,
            content: input.clone(),
            model: None,
            tokens: None,
            duration_ms: None,
        });

        // Scroll to bottom
        self.messages_scroll = self.messages.len();

        // Generate response
        self.generating = true;
        self.status_message = "Generating...".to_string();

        let model = self.selected_model.clone();
        match self
            .client
            .chat(&input, model.as_deref(), self.max_tokens, self.temperature)
        {
            Ok(result) => {
                self.messages.push(Message {
                    role: Role::Assistant,
                    content: result.text,
                    model: Some(result.model),
                    tokens: Some(result.completion_tokens),
                    duration_ms: Some(result.duration_ms),
                });

                let tps = if result.duration_ms > 0 {
                    (result.completion_tokens as f64) / (result.duration_ms as f64 / 1000.0)
                } else {
                    0.0
                };
                self.status_message = format!(
                    "{} tokens in {}ms ({:.1} tok/s)",
                    result.completion_tokens, result.duration_ms, tps
                );
            }
            Err(e) => {
                self.last_error = Some(e.to_string());
                self.status_message = "Generation failed".to_string();
                self.add_system_message(&format!("Error: {}", e));
            }
        }

        self.generating = false;
        self.messages_scroll = self.messages.len();
    }
}
