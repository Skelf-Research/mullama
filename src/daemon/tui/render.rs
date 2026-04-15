use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        Block, Borders, Clear, List, ListItem, Paragraph, Scrollbar, ScrollbarOrientation, Wrap,
    },
    Frame,
};

use super::app::TuiApp;
use super::{centered_rect, InputMode, Role};

impl TuiApp {
    pub(crate) fn draw(&mut self, f: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Header
                Constraint::Min(10),   // Chat
                Constraint::Length(3), // Input
                Constraint::Length(1), // Status
            ])
            .split(f.area());

        self.draw_header(f, chunks[0]);
        self.draw_chat(f, chunks[1]);
        self.draw_input(f, chunks[2]);
        self.draw_status(f, chunks[3]);

        if self.show_help {
            self.draw_help_popup(f);
        }

        if self.input_mode == InputMode::ModelSelect {
            self.draw_model_popup(f);
        }
    }

    fn draw_header(&self, f: &mut Frame, area: Rect) {
        let model_name = self
            .selected_model
            .as_deref()
            .or_else(|| {
                self.models
                    .iter()
                    .find(|m| m.is_default)
                    .map(|m| m.alias.as_str())
            })
            .unwrap_or("none");

        let models_count = self.models.len();

        let header = Paragraph::new(Line::from(vec![
            Span::styled(
                " Mullama ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("│ Model: "),
            Span::styled(
                model_name,
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" │ "),
            Span::styled(
                format!("{} models", models_count),
                Style::default().fg(Color::DarkGray),
            ),
            Span::raw(" │ "),
            Span::styled(
                format!("temp={:.1}", self.temperature),
                Style::default().fg(Color::DarkGray),
            ),
            Span::raw(" │ "),
            Span::styled(
                format!("max={}", self.max_tokens),
                Style::default().fg(Color::DarkGray),
            ),
        ]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );

        f.render_widget(header, area);
    }

    fn draw_chat(&mut self, f: &mut Frame, area: Rect) {
        let inner_height = area.height.saturating_sub(2) as usize;

        let items: Vec<ListItem> = self
            .messages
            .iter()
            .flat_map(|msg| {
                let (prefix, style, content_style) = match msg.role {
                    Role::User => (
                        "You",
                        Style::default()
                            .fg(Color::Green)
                            .add_modifier(Modifier::BOLD),
                        Style::default().fg(Color::White),
                    ),
                    Role::Assistant => (
                        msg.model.as_deref().unwrap_or("AI"),
                        Style::default()
                            .fg(Color::Blue)
                            .add_modifier(Modifier::BOLD),
                        Style::default().fg(Color::White),
                    ),
                    Role::System => (
                        "System",
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::ITALIC),
                        Style::default().fg(Color::Yellow),
                    ),
                };

                let mut lines = vec![ListItem::new(Line::from(vec![
                    Span::styled(format!("┌─ {} ", prefix), style),
                    if let (Some(tokens), Some(ms)) = (msg.tokens, msg.duration_ms) {
                        Span::styled(
                            format!("[{} tokens, {}ms]", tokens, ms),
                            Style::default().fg(Color::DarkGray),
                        )
                    } else {
                        Span::raw("")
                    },
                ]))];

                for line in msg.content.lines() {
                    lines.push(ListItem::new(Line::from(vec![
                        Span::styled("│ ", Style::default().fg(Color::DarkGray)),
                        Span::styled(line, content_style),
                    ])));
                }

                lines.push(ListItem::new(Line::from(Span::styled(
                    "└─",
                    Style::default().fg(Color::DarkGray),
                ))));
                lines.push(ListItem::new(Line::from("")));

                lines
            })
            .collect();

        let total_items = items.len();
        self.messages_scroll = self
            .messages_scroll
            .min(total_items.saturating_sub(inner_height));

        self.messages_scroll_state = self
            .messages_scroll_state
            .content_length(total_items)
            .position(self.messages_scroll);

        // Skip items for scrolling effect
        let visible_items: Vec<_> = items.into_iter().skip(self.messages_scroll).collect();

        let list = List::new(visible_items).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(" Chat "),
        );

        f.render_widget(list, area);

        // Scrollbar
        f.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None),
            area.inner(ratatui::layout::Margin {
                vertical: 1,
                horizontal: 0,
            }),
            &mut self.messages_scroll_state,
        );
    }

    fn draw_input(&self, f: &mut Frame, area: Rect) {
        let (title, border_color) = match self.input_mode {
            InputMode::Insert => (" Message (Enter to send) ", Color::Green),
            InputMode::Normal => (" NORMAL (i to insert) ", Color::Yellow),
            InputMode::Command => (" Command ", Color::Magenta),
            InputMode::ModelSelect => (" Select Model ", Color::Cyan),
        };

        let display_input = if self.input_mode == InputMode::Command {
            format!(":{}", self.input)
        } else {
            self.input.clone()
        };

        let input = Paragraph::new(display_input.as_str())
            .style(Style::default().fg(Color::White))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(border_color))
                    .title(title),
            );

        f.render_widget(input, area);

        // Cursor
        if self.input_mode == InputMode::Insert || self.input_mode == InputMode::Command {
            let cursor_offset = if self.input_mode == InputMode::Command {
                1
            } else {
                0
            };
            f.set_cursor_position((
                area.x + self.cursor_pos as u16 + 1 + cursor_offset,
                area.y + 1,
            ));
        }
    }

    fn draw_status(&self, f: &mut Frame, area: Rect) {
        let status = Paragraph::new(Line::from(vec![
            Span::raw(" "),
            if self.generating {
                Span::styled("● Generating", Style::default().fg(Color::Yellow))
            } else {
                Span::styled("●", Style::default().fg(Color::Green))
            },
            Span::raw(" │ "),
            Span::styled(&self.status_message, Style::default().fg(Color::DarkGray)),
            Span::raw(" │ "),
            Span::styled(
                "?: help  Ctrl+M: models  Ctrl+L: clear  Ctrl+Q: quit",
                Style::default().fg(Color::DarkGray),
            ),
        ]))
        .style(Style::default().bg(Color::Rgb(30, 30, 30)));

        f.render_widget(status, area);
    }

    fn draw_help_popup(&self, f: &mut Frame) {
        let area = centered_rect(60, 70, f.area());
        f.render_widget(Clear, area);

        let help_text = vec![
            Line::from(Span::styled(
                "Mullama TUI Help",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "Keyboard Shortcuts:",
                Style::default().add_modifier(Modifier::UNDERLINED),
            )),
            Line::from("  Ctrl+Q       Quit"),
            Line::from("  Ctrl+C       Cancel generation / Quit"),
            Line::from("  Ctrl+L       Clear chat"),
            Line::from("  Ctrl+M       Select model"),
            Line::from("  ?            Toggle help"),
            Line::from("  Esc          Normal mode"),
            Line::from("  i/a          Insert mode"),
            Line::from("  :            Command mode"),
            Line::from(""),
            Line::from(Span::styled(
                "Commands:",
                Style::default().add_modifier(Modifier::UNDERLINED),
            )),
            Line::from("  :model <n>   Select model"),
            Line::from("  :models      List models"),
            Line::from("  :load <p>    Load model (alias:path)"),
            Line::from("  :unload <n>  Unload model"),
            Line::from("  :temp <v>    Set temperature"),
            Line::from("  :tokens <n>  Set max tokens"),
            Line::from("  :status      Show daemon status"),
            Line::from("  :clear       Clear chat"),
            Line::from("  :quit        Quit"),
            Line::from(""),
            Line::from(Span::styled(
                "Press any key to close",
                Style::default().fg(Color::DarkGray),
            )),
        ];

        let help = Paragraph::new(help_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Cyan))
                    .title(" Help "),
            )
            .wrap(Wrap { trim: false });

        f.render_widget(help, area);
    }

    fn draw_model_popup(&self, f: &mut Frame) {
        let area = centered_rect(50, 50, f.area());
        f.render_widget(Clear, area);

        let items: Vec<ListItem> = self
            .models
            .iter()
            .enumerate()
            .map(|(i, model)| {
                let style = if i == self.model_select_index {
                    Style::default().bg(Color::Blue).fg(Color::White)
                } else {
                    Style::default()
                };

                let marker = if model.is_default { " *" } else { "" };
                ListItem::new(Line::from(vec![
                    Span::styled(format!(" {}{} ", model.alias, marker), style),
                    Span::styled(
                        format!("({:.0}B params)", model.info.parameters as f64 / 1e9),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]))
            })
            .collect();

        let list = List::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title(" Select Model (Enter to confirm, Esc to cancel) "),
        );

        f.render_widget(list, area);
    }
}
