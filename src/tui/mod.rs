mod globe;
mod theme;
mod worldmap;

use std::io::{self, Stdout};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Tabs, Wrap};
use ratatui::{Frame, Terminal};

use crate::gateway;
use crate::models;
use crate::orchestrator::{self, PeerRecordView};
use globe::Globe;

type ComputeTerminal = Terminal<CrosstermBackend<Stdout>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Overview,
    Peers,
    Chat,
    Logs,
}

impl Tab {
    fn all() -> [Tab; 4] {
        [Tab::Overview, Tab::Peers, Tab::Chat, Tab::Logs]
    }

    fn index(self) -> usize {
        match self {
            Tab::Overview => 0,
            Tab::Peers => 1,
            Tab::Chat => 2,
            Tab::Logs => 3,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Tab::Overview => "Overview",
            Tab::Peers => "Peers",
            Tab::Chat => "Test Chat",
            Tab::Logs => "Logs",
        }
    }
}

struct ChatLine {
    author: &'static str,
    text: String,
}

struct App {
    tab: Tab,
    globe: Globe,
    gateway_addr: String,
    orchestrator_url: String,
    peers: Vec<PeerRecordView>,
    logs: Vec<String>,
    chat_input: String,
    chat_lines: Vec<ChatLine>,
    last_refresh: Instant,
}

impl App {
    fn new(gateway_addr: String, orchestrator_url: String) -> Self {
        let mut globe = Globe::new();
        globe.set_mock_nodes();
        Self {
            tab: Tab::Overview,
            globe,
            gateway_addr,
            orchestrator_url,
            peers: Vec::new(),
            logs: vec!["TUI ready".to_string()],
            chat_input: String::new(),
            chat_lines: Vec::new(),
            last_refresh: Instant::now() - Duration::from_secs(60),
        }
    }

    fn run(mut self, terminal: &mut ComputeTerminal) -> Result<()> {
        loop {
            self.refresh_if_needed();
            self.globe.tick();
            terminal.draw(|frame| self.render(frame))?;

            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press
                        && self.handle_key(key.code, key.modifiers)?
                    {
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    fn refresh_if_needed(&mut self) {
        if self.last_refresh.elapsed() < Duration::from_secs(3) {
            return;
        }
        self.last_refresh = Instant::now();
        match orchestrator::fetch_peers(&self.orchestrator_url) {
            Ok(peers) => {
                self.peers = peers;
            }
            Err(err) => {
                self.log(format!("orchestrator refresh failed: {err:#}"));
            }
        }
    }

    fn handle_key(&mut self, code: KeyCode, modifiers: KeyModifiers) -> Result<bool> {
        if modifiers.contains(KeyModifiers::CONTROL) && matches!(code, KeyCode::Char('c')) {
            return Ok(true);
        }
        match code {
            KeyCode::Esc => return Ok(true),
            KeyCode::Tab => self.next_tab(),
            KeyCode::BackTab => self.prev_tab(),
            KeyCode::Char('1') => self.tab = Tab::Overview,
            KeyCode::Char('2') => self.tab = Tab::Peers,
            KeyCode::Char('3') => self.tab = Tab::Chat,
            KeyCode::Char('4') => self.tab = Tab::Logs,
            _ if self.tab == Tab::Chat => self.handle_chat_key(code)?,
            _ => {}
        }
        Ok(false)
    }

    fn handle_chat_key(&mut self, code: KeyCode) -> Result<()> {
        match code {
            KeyCode::Enter => {
                let prompt = self.chat_input.trim().to_string();
                if prompt.is_empty() {
                    return Ok(());
                }
                self.chat_input.clear();
                self.chat_lines.push(ChatLine {
                    author: "you",
                    text: prompt.clone(),
                });
                self.log(format!("sending prompt to gateway {}", self.gateway_addr));
                match gateway::complete_prompt(&self.gateway_addr, &prompt, 96) {
                    Ok(completion) => {
                        self.chat_lines.push(ChatLine {
                            author: "model",
                            text: completion.text.trim().to_string(),
                        });
                        self.log(format!(
                            "completion tokens={} ttft={}ms total={}ms",
                            completion.completion_tokens,
                            completion.timings.ttft_ms,
                            completion.timings.total_ms
                        ));
                    }
                    Err(err) => {
                        self.chat_lines.push(ChatLine {
                            author: "error",
                            text: format!("{err:#}"),
                        });
                    }
                }
            }
            KeyCode::Backspace => {
                self.chat_input.pop();
            }
            KeyCode::Char(ch) => {
                self.chat_input.push(ch);
            }
            _ => {}
        }
        Ok(())
    }

    fn next_tab(&mut self) {
        let tabs = Tab::all();
        self.tab = tabs[(self.tab.index() + 1) % tabs.len()];
    }

    fn prev_tab(&mut self) {
        let tabs = Tab::all();
        self.tab = tabs[(self.tab.index() + tabs.len() - 1) % tabs.len()];
    }

    fn log(&mut self, message: String) {
        self.logs.push(message);
        if self.logs.len() > 200 {
            self.logs.remove(0);
        }
    }

    fn render(&mut self, frame: &mut Frame<'_>) {
        let area = frame.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),
                Constraint::Length(3),
                Constraint::Min(4),
                Constraint::Length(1),
            ])
            .split(area);

        self.render_header(frame, chunks[0]);
        self.render_tabs(frame, chunks[1]);
        match self.tab {
            Tab::Overview => self.render_overview(frame, chunks[2]),
            Tab::Peers => self.render_peers(frame, chunks[2]),
            Tab::Chat => self.render_chat(frame, chunks[2]),
            Tab::Logs => self.render_logs(frame, chunks[2]),
        }
        self.render_footer(frame, chunks[3]);
    }

    fn render_header(&self, frame: &mut Frame<'_>, area: Rect) {
        let title = Line::from(vec![
            Span::styled("Compute", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(" Sharding  "),
            Span::styled(
                models::MODEL_ID,
                Style::default().fg(theme::palette().muted),
            ),
        ]);
        frame.render_widget(Paragraph::new(title), area);
    }

    fn render_tabs(&self, frame: &mut Frame<'_>, area: Rect) {
        let titles: Vec<Line<'_>> = Tab::all()
            .iter()
            .map(|tab| Line::from(tab.label()))
            .collect();
        let tabs = Tabs::new(titles)
            .select(self.tab.index())
            .block(Block::default().borders(Borders::BOTTOM))
            .highlight_style(Style::default().add_modifier(Modifier::BOLD));
        frame.render_widget(tabs, area);
    }

    fn render_overview(&mut self, frame: &mut Frame<'_>, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(48), Constraint::Percentage(52)])
            .split(area);

        let globe_block = Block::default()
            .title("Compute network")
            .borders(Borders::ALL);
        let globe_inner = globe_block.inner(chunks[0]);
        frame.render_widget(globe_block, chunks[0]);
        self.globe
            .render(globe_inner, frame.buffer_mut(), theme::palette());

        let head_count = self
            .peers
            .iter()
            .filter(|peer| peer.advert.role.to_string() == "head")
            .count();
        let tail_count = self
            .peers
            .iter()
            .filter(|peer| peer.advert.role.to_string() == "tail")
            .count();
        let status = vec![
            Line::from(format!("Gateway: {}", self.gateway_addr)),
            Line::from(format!("Orchestrator: {}", self.orchestrator_url)),
            Line::from(""),
            Line::from(format!("Model: {}", models::MODEL_LABEL)),
            Line::from("Validated split: head layers 0-20, tail layers 21-41"),
            Line::from(""),
            Line::from(format!("Known peers: {}", self.peers.len())),
            Line::from(format!("Heads: {head_count}  Tails: {tail_count}")),
            Line::from(""),
            Line::from("Tab to switch views. Esc exits."),
        ];
        let paragraph = Paragraph::new(status)
            .block(Block::default().title("Status").borders(Borders::ALL))
            .wrap(Wrap { trim: true });
        frame.render_widget(paragraph, chunks[1]);
    }

    fn render_peers(&self, frame: &mut Frame<'_>, area: Rect) {
        let items = if self.peers.is_empty() {
            vec![ListItem::new("No peers discovered yet")]
        } else {
            self.peers
                .iter()
                .map(|peer| {
                    let latency = peer
                        .latency_ms
                        .map(|value| format!("{value}ms"))
                        .unwrap_or_else(|| "-".to_string());
                    ListItem::new(format!(
                        "{:<5} {:>6}  {:<24}  stage={}",
                        peer.advert.role,
                        latency,
                        short_id(&peer.advert.id),
                        peer.advert.stage_addr.as_deref().unwrap_or("-")
                    ))
                })
                .collect()
        };
        frame.render_widget(
            List::new(items).block(Block::default().title("Peers").borders(Borders::ALL)),
            area,
        );
    }

    fn render_chat(&self, frame: &mut Frame<'_>, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(3), Constraint::Length(3)])
            .split(area);
        let lines: Vec<Line<'_>> = self
            .chat_lines
            .iter()
            .flat_map(|line| {
                vec![
                    Line::from(Span::styled(
                        format!("{}:", line.author),
                        Style::default().add_modifier(Modifier::BOLD),
                    )),
                    Line::from(line.text.clone()),
                    Line::from(""),
                ]
            })
            .collect();
        frame.render_widget(
            Paragraph::new(lines)
                .block(Block::default().title("Chat").borders(Borders::ALL))
                .wrap(Wrap { trim: false }),
            chunks[0],
        );
        frame.render_widget(Clear, chunks[1]);
        frame.render_widget(
            Paragraph::new(self.chat_input.as_str())
                .block(Block::default().title("Prompt").borders(Borders::ALL)),
            chunks[1],
        );
    }

    fn render_logs(&self, frame: &mut Frame<'_>, area: Rect) {
        let logs: Vec<ListItem<'_>> = self
            .logs
            .iter()
            .rev()
            .take(area.height.saturating_sub(2) as usize)
            .map(|line| ListItem::new(line.clone()))
            .collect();
        frame.render_widget(
            List::new(logs).block(Block::default().title("Logs").borders(Borders::ALL)),
            area,
        );
    }

    fn render_footer(&self, frame: &mut Frame<'_>, area: Rect) {
        frame.render_widget(
            Paragraph::new("1 Overview  2 Peers  3 Chat  4 Logs  Tab switch  Esc quit"),
            area,
        );
    }
}

pub fn run_tui(gateway_addr: String, orchestrator_url: String) -> Result<()> {
    enable_raw_mode().context("enabling raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("entering alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("creating terminal")?;

    let result = App::new(gateway_addr, orchestrator_url).run(&mut terminal);

    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();
    terminal.show_cursor().ok();
    result
}

fn short_id(id: &str) -> String {
    id.chars().take(8).collect()
}
