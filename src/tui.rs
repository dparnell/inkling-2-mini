//! Terminal dashboard for `serve --tui`: live sparklines for throughput, latency, GPU load and
//! memory, a table of recent requests, and the server log.

use crate::metrics::{self, ReqRow, ReqState, Sample, Snapshot, metrics};
use anyhow::Result;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Cell, Paragraph, Row, Sparkline, Table, TableState};
use ratatui::{DefaultTerminal, Frame};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

/// A running dashboard; dropping it without `stop` leaves the terminal in raw mode.
pub struct Tui {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Tui {
    /// Take over the terminal and start drawing.
    pub fn start() -> Result<Tui> {
        let terminal = ratatui::try_init()?;
        let stop = Arc::new(AtomicBool::new(false));
        let s = stop.clone();
        let handle = std::thread::Builder::new()
            .name("inkling-tui".into())
            .spawn(move || run(terminal, &s))?;
        Ok(Tui {
            stop,
            handle: Some(handle),
        })
    }

    /// Stop drawing and give the terminal back.
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for Tui {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[derive(Default)]
struct View {
    /// rows scrolled back from the newest request
    scroll: usize,
}

fn run(mut terminal: DefaultTerminal, stop: &AtomicBool) {
    let mut view = View::default();
    while !stop.load(Ordering::SeqCst) {
        let size = terminal.size().unwrap_or_default();
        let snap = metrics().snapshot(size.width as usize, 200);
        if terminal.draw(|f| draw(f, &snap, &view)).is_err() {
            break;
        }
        if !event::poll(Duration::from_millis(250)).unwrap_or(false) {
            continue;
        }
        let Ok(Event::Key(k)) = event::read() else {
            continue;
        };
        if k.kind != KeyEventKind::Press {
            continue;
        }
        let ctrl_c = k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL);
        match k.code {
            KeyCode::Char('q') | KeyCode::Esc => quit(&snap),
            _ if ctrl_c => quit(&snap),
            KeyCode::Up => view.scroll += 1,
            KeyCode::Down => view.scroll = view.scroll.saturating_sub(1),
            KeyCode::PageUp => view.scroll += 10,
            KeyCode::PageDown => view.scroll = view.scroll.saturating_sub(10),
            KeyCode::Home | KeyCode::End => view.scroll = 0,
            _ => {}
        }
        view.scroll = view.scroll.min(snap.requests.len().saturating_sub(1));
    }
    ratatui::restore();
}

fn quit(snap: &Snapshot) {
    if snap.info.status == "serving" {
        metrics().set_info(|i| i.status = "stopping".into());
        metrics::request_shutdown();
    } else {
        // still loading: nothing to shut down gracefully
        ratatui::restore();
        std::process::exit(130);
    }
}

// ---------------------------------------------------------------------------------------------
// formatting

fn bytes(n: u64) -> String {
    let n = n as f64;
    if n >= 1e9 {
        format!("{:.2} GB", n / 1e9)
    } else if n >= 1e6 {
        format!("{:.0} MB", n / 1e6)
    } else {
        format!("{:.0} kB", n / 1e3)
    }
}

fn count(n: u64) -> String {
    let f = n as f64;
    if f >= 1e6 {
        format!("{:.1}M", f / 1e6)
    } else if f >= 1e4 {
        format!("{:.1}k", f / 1e3)
    } else {
        n.to_string()
    }
}

fn clock(secs: f64) -> String {
    let s = secs as u64;
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
    } else {
        format!("{:02}:{:02}", s / 60, s % 60)
    }
}

fn ms(v: f64) -> String {
    if v >= 10_000.0 {
        format!("{:.1} s", v / 1e3)
    } else {
        format!("{v:.0} ms")
    }
}

// ---------------------------------------------------------------------------------------------
// drawing

fn draw(f: &mut Frame, snap: &Snapshot, view: &View) {
    let area = f.area();
    let log_h = if area.height >= 44 {
        10
    } else if area.height >= 32 {
        6
    } else {
        4
    };
    let [header, row1, row2, requests, logs] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(6),
        Constraint::Length(6),
        Constraint::Min(6),
        Constraint::Length(log_h),
    ])
    .areas(area);
    draw_header(f, header, snap);

    let cols = |r: Rect| -> [Rect; 3] {
        Layout::horizontal([
            Constraint::Fill(1),
            Constraint::Fill(1),
            Constraint::Fill(1),
        ])
        .areas(r)
    };
    let [a, b, c] = cols(row1);
    let [d, e, x, g] = Layout::horizontal([Constraint::Fill(1), Constraint::Fill(1), Constraint::Fill(1), Constraint::Fill(1)]).areas(row2);
    let s = &snap.samples;
    draw_generated(f, a, s);
    draw_prompt(f, b, s);
    draw_ttft(f, c, s);
    draw_gpu(f, d, s, snap.info.gpu_source.as_deref());
    draw_vram(f, e, s);
    draw_experts(f, x, s);
    draw_ram(f, g, s);
    draw_requests(f, requests, snap, view);
    draw_logs(f, logs, snap);
}

fn draw_header(f: &mut Frame, area: Rect, snap: &Snapshot) {
    let i = &snap.info;
    let status_style = match i.status.as_str() {
        "serving" => Style::new().fg(Color::Green).bold(),
        "stopping" => Style::new().fg(Color::Red).bold(),
        _ => Style::new().fg(Color::Yellow).bold(),
    };
    let sep = Span::raw(" │ ").dark_gray();
    let mut spans = vec![
        Span::styled(
            " inkling ",
            Style::new().fg(Color::Black).bg(Color::Cyan).bold(),
        ),
        Span::raw(" "),
        Span::styled(i.status.clone(), status_style),
    ];
    for part in [&i.model, &i.gpu, &i.addr] {
        if !part.is_empty() {
            spans.push(sep.clone());
            spans.push(Span::raw(part.clone()));
        }
    }
    spans.push(sep.clone());
    spans.push(Span::raw(format!(
        "up {}",
        clock(snap.uptime.as_secs_f64())
    )));
    let [left, right] =
        Layout::horizontal([Constraint::Fill(1), Constraint::Length(19)]).areas(area);
    f.render_widget(Paragraph::new(Line::from(spans)), left);
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::raw("↑↓").bold(),
            Span::raw(" scroll ").dark_gray(),
            Span::raw("q").bold(),
            Span::raw(" quit ").dark_gray(),
        ]))
        .right_aligned(),
        right,
    );
}

/// The newest `width` points of `series`, right-aligned (older slots left empty).
fn tail(series: Vec<Option<u64>>, width: usize) -> Vec<Option<u64>> {
    let n = series.len();
    let mut out = vec![None; width.saturating_sub(n)];
    out.extend(series.into_iter().skip(n.saturating_sub(width)));
    out
}

fn spark(
    f: &mut Frame,
    area: Rect,
    title: Line,
    footer: Line,
    data: Vec<Option<u64>>,
    color: Color,
    max: Option<u64>,
) {
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().dark_gray())
        .title(title)
        .title_bottom(footer);
    let inner = block.inner(area);
    let data = tail(data, inner.width as usize);
    let mut w = Sparkline::default()
        .block(block)
        .data(data)
        .style(Style::new().fg(color))
        .absent_value_symbol(" ");
    if let Some(m) = max {
        w = w.max(m.max(1));
    }
    f.render_widget(w, area);
}

fn title(name: &str, value: String, color: Color) -> Line<'static> {
    Line::from(vec![
        Span::raw(format!(" {name} ")).bold(),
        Span::styled(value, Style::new().fg(color).add_modifier(Modifier::BOLD)),
        Span::raw(" "),
    ])
}

fn note(text: String) -> Line<'static> {
    Line::from(Span::raw(format!(" {text} ")).dark_gray())
}

fn peak(s: &[Sample], f: impl Fn(&Sample) -> f64) -> f64 {
    s.iter().map(f).fold(0.0, f64::max)
}

/// Mean over the samples where the value was non-zero (i.e. while busy).
fn active_mean(s: &[Sample], f: impl Fn(&Sample) -> f64) -> Option<f64> {
    let v: Vec<f64> = s.iter().map(f).filter(|&x| x > 0.0).collect();
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}

fn draw_generated(f: &mut Frame, area: Rect, s: &[Sample]) {
    let cur = s.last().map(|x| x.gen_tps).unwrap_or(0.0);
    let audio = s.last().map(|x| x.audio_fps).unwrap_or(0.0);
    let value = if audio > 0.0 {
        format!("{cur:.0}/s · {audio:.0} speech frames/s")
    } else {
        format!("{cur:.0} tok/s")
    };
    let footer = match active_mean(s, |x| x.gen_tps) {
        Some(m) => format!("peak {:.0} · avg while busy {m:.0}", peak(s, |x| x.gen_tps)),
        None => "idle".into(),
    };
    let data = s.iter().map(|x| Some(x.gen_tps.round() as u64)).collect();
    spark(
        f,
        area,
        title("Generated", value, Color::Green),
        note(footer),
        data,
        Color::Green,
        None,
    );
}

fn draw_prompt(f: &mut Frame, area: Rect, s: &[Sample]) {
    let cur = s.last().map(|x| x.prompt_tps).unwrap_or(0.0);
    let footer = match active_mean(s, |x| x.prompt_tps) {
        Some(_) => format!(
            "peak {:.0} · cached tokens excluded",
            peak(s, |x| x.prompt_tps)
        ),
        None => "idle · cached tokens excluded".into(),
    };
    let data = s
        .iter()
        .map(|x| Some(x.prompt_tps.round() as u64))
        .collect();
    spark(
        f,
        area,
        title("Prompt", format!("{cur:.0} tok/s"), Color::Cyan),
        note(footer),
        data,
        Color::Cyan,
        None,
    );
}

fn draw_ttft(f: &mut Frame, area: Rect, s: &[Sample]) {
    let last = s.iter().rev().find_map(|x| x.ttft_ms);
    let vals: Vec<f64> = s.iter().filter_map(|x| x.ttft_ms).collect();
    let value = last.map(ms).unwrap_or_else(|| "–".into());
    let footer = if vals.is_empty() {
        "no requests yet".into()
    } else {
        let mut v = vals.clone();
        v.sort_by(|a, b| a.total_cmp(b));
        format!(
            "median {} · max {}",
            ms(v[v.len() / 2]),
            ms(*v.last().unwrap())
        )
    };
    let data = s
        .iter()
        .map(|x| x.ttft_ms.map(|t| t.round() as u64))
        .collect();
    spark(
        f,
        area,
        title("Time to first token", value, Color::Yellow),
        note(footer),
        data,
        Color::Yellow,
        None,
    );
}

fn draw_gpu(f: &mut Frame, area: Rect, s: &[Sample], source: Option<&str>) {
    let last = s.last();
    let busy = last.map(|x| x.busy * 100.0).unwrap_or(0.0);
    match last.and_then(|x| x.gpu_util) {
        Some(u) => {
            let data = s
                .iter()
                .map(|x| x.gpu_util.map(|u| u.round() as u64))
                .collect();
            spark(
                f,
                area,
                title("GPU", format!("{u:.0}%"), Color::Magenta),
                note(format!(
                    "engine busy {busy:.0}% · {}",
                    source.unwrap_or("device")
                )),
                data,
                Color::Magenta,
                Some(100),
            );
        }
        None => {
            let data = s
                .iter()
                .map(|x| Some((x.busy * 100.0).round() as u64))
                .collect();
            spark(
                f,
                area,
                title("GPU (engine busy)", format!("{busy:.0}%"), Color::Magenta),
                note("no device counters; share of time running requests".into()),
                data,
                Color::Magenta,
                Some(100),
            );
        }
    }
}

fn draw_experts(f: &mut Frame, area: Rect, s: &[Sample]) {
    let last = s.last().cloned().unwrap_or_default();
    let hit = s.iter().rev().find_map(|x| x.expert_hit);
    let value = match hit {
        Some(h) => format!("{:.1}% hit", h * 100.0),
        None => "idle".into(),
    };
    let data = s.iter().map(|x| Some((x.upload_bps / 1e6) as u64)).collect();
    spark(
        f,
        area,
        title("MoE cache", value, Color::Yellow),
        note(format!("streaming {:.0} MB/s", last.upload_bps / 1e6)),
        data,
        Color::Yellow,
        None,
    );
}

fn draw_vram(f: &mut Frame, area: Rect, s: &[Sample]) {
    let last = s.last().cloned().unwrap_or_default();
    match last.vram {
        Some((used, total)) => {
            let data = s.iter().map(|x| x.vram.map(|(u, _)| u >> 20)).collect();
            spark(
                f,
                area,
                title(
                    "VRAM",
                    format!("{} / {}", bytes(used), bytes(total)),
                    Color::Blue,
                ),
                note(format!("inkling buffers {}", bytes(last.gpu_alloc))),
                data,
                Color::Blue,
                Some(total >> 20),
            );
        }
        None => {
            let data = s.iter().map(|x| Some(x.gpu_alloc >> 20)).collect();
            spark(
                f,
                area,
                title("GPU buffers", bytes(last.gpu_alloc), Color::Blue),
                note("allocated by inkling (no device counters)".into()),
                data,
                Color::Blue,
                None,
            );
        }
    }
}

fn draw_ram(f: &mut Frame, area: Rect, s: &[Sample]) {
    let last = s.last().map(|x| x.rss).unwrap_or(0);
    let pk = s.iter().map(|x| x.rss).max().unwrap_or(0);
    let data = s.iter().map(|x| Some(x.rss >> 20)).collect();
    spark(
        f,
        area,
        title("Process memory", bytes(last), Color::LightRed),
        note(format!("resident · peak {}", bytes(pk))),
        data,
        Color::LightRed,
        None,
    );
}

fn state_cell(r: &ReqRow) -> (String, Style) {
    match &r.state {
        ReqState::Queued => ("queued".into(), Style::new().fg(Color::Gray)),
        ReqState::Prefill => ("prefill".into(), Style::new().fg(Color::Yellow).bold()),
        ReqState::Generating => ("generating".into(), Style::new().fg(Color::Cyan).bold()),
        ReqState::Done(f) if f == "cancelled" => ("cancelled".into(), Style::new().dark_gray()),
        ReqState::Done(f) => (f.clone(), Style::new().fg(Color::Green)),
        ReqState::Failed(_) => ("failed".into(), Style::new().fg(Color::Red).bold()),
        ReqState::Rejected(code, _) => (format!("{code}"), Style::new().fg(Color::Red)),
    }
}

/// Short modality labels for the table.
fn short(m: &str) -> String {
    m.replace("text", "txt")
        .replace("image", "img")
        .replace("audio", "aud")
}

/// Short endpoint names (rejected requests carry the raw path).
fn endpoint_name(e: &str) -> &str {
    match e {
        "chat/completions" => "chat",
        "audio/speech" => "speech",
        "audio/transcriptions" => "transcriptions",
        other => other,
    }
}

fn draw_requests(f: &mut Frame, area: Rect, snap: &Snapshot, view: &View) {
    let t = &snap.totals;
    let running = snap
        .requests
        .iter()
        .filter(|r| matches!(r.state, ReqState::Prefill | ReqState::Generating))
        .count();
    let queued = snap
        .requests
        .iter()
        .filter(|r| r.state == ReqState::Queued)
        .count();
    let dim = |s: String| Span::raw(s).dark_gray();
    let bold = |s: String| Span::raw(s).bold();
    let errs = t.failed + t.rejected;
    let mut summary = vec![
        Span::raw(" "),
        bold(count(t.requests)),
        dim(" requests · ".into()),
        bold(running.to_string()),
        dim(" running · ".into()),
        bold(queued.to_string()),
        dim(" queued · ".into()),
        bold(count(t.completed)),
        dim(" done · ".into()),
        Span::styled(
            count(errs),
            if errs > 0 {
                Style::new().fg(Color::Red).bold()
            } else {
                Style::new().bold()
            },
        ),
        dim(" errors".into()),
    ];
    if t.cancelled > 0 {
        summary.extend([
            dim(" · ".into()),
            bold(count(t.cancelled)),
            dim(" cancelled".into()),
        ]);
    }
    summary.extend([
        dim("  │  prompt ".into()),
        bold(count(t.prompt_tokens)),
        dim(format!(" tok ({} cached) · out ", count(t.cached_tokens))),
        bold(count(t.generated)),
    ]);
    if t.audio_frames > 0 {
        summary.push(dim(format!(" ({} speech frames)", count(t.audio_frames))));
    }

    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().dark_gray())
        .title(Line::from(" Requests ").bold());
    let inner = block.inner(area);
    f.render_widget(block, area);
    let [sum_area, table_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Fill(1)]).areas(inner);
    f.render_widget(Paragraph::new(Line::from(summary)), sum_area);

    let header = Row::new(
        [
            "time",
            "endpoint",
            "in → out",
            "state",
            "prompt",
            "out",
            "prefill",
            "ttft",
            "tok/s",
            "total",
            "detail",
        ]
        .map(|h| Cell::from(h).style(Style::new().dark_gray().bold())),
    );
    let rows: Vec<Row> = snap
        .requests
        .iter()
        .rev()
        .map(|r| {
            let (state, st) = state_cell(r);
            let opt = |v: Option<f64>, f: fn(f64) -> String| v.map(f).unwrap_or_default();
            let engine = !r.output.is_empty();
            let io = if engine {
                format!(
                    "{}→{}{}",
                    short(&r.input),
                    short(r.output),
                    if r.stream { " ⇢" } else { "" }
                )
            } else {
                String::new()
            };
            let prompt = match (engine, r.cached) {
                (false, _) => String::new(),
                (true, 0) => count(r.prompt as u64),
                (true, c) => format!("{} ({})", count(r.prompt as u64), count(c as u64)),
            };
            let out = match (engine, r.audio_frames) {
                (false, _) => String::new(),
                (true, 0) => r.generated.to_string(),
                (true, a) => format!("{} ({a}♪)", r.generated),
            };
            let detail = match &r.state {
                ReqState::Failed(e) | ReqState::Rejected(_, e) => e.clone(),
                _ => String::new(),
            };
            Row::new(vec![
                Cell::from(clock(r.at)).style(Style::new().dark_gray()),
                Cell::from(endpoint_name(&r.endpoint).to_string()),
                Cell::from(io),
                Cell::from(state).style(st),
                Cell::from(prompt),
                Cell::from(out),
                Cell::from(opt(r.prefill_ms, ms)),
                Cell::from(opt(r.ttft_ms, ms)),
                Cell::from(opt(r.decode_tps, |v| format!("{v:.0}"))),
                Cell::from(opt(r.total_ms, ms)),
                Cell::from(detail).style(Style::new().fg(Color::Red)),
            ])
        })
        .collect();
    if rows.is_empty() {
        f.render_widget(
            Paragraph::new(Line::from(" waiting for requests…").dark_gray()),
            table_area,
        );
        return;
    }
    let widths = [
        Constraint::Length(7),
        Constraint::Length(14),
        Constraint::Length(15),
        Constraint::Length(10),
        Constraint::Length(11),
        Constraint::Length(10),
        Constraint::Length(7),
        Constraint::Length(7),
        Constraint::Length(5),
        Constraint::Length(7),
        Constraint::Min(8),
    ];
    let table = Table::new(rows, widths).header(header).column_spacing(1);
    let mut state = TableState::default().with_offset(view.scroll);
    f.render_stateful_widget(table, table_area, &mut state);
}

fn draw_logs(f: &mut Frame, area: Rect, snap: &Snapshot) {
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().dark_gray())
        .title(Line::from(" Log ").bold());
    let inner = block.inner(area);
    let n = inner.height as usize;
    let lines: Vec<Line> = snap
        .logs
        .iter()
        .skip(snap.logs.len().saturating_sub(n))
        .map(|l| {
            let style = if l.contains(" ERROR ") {
                Style::new().fg(Color::Red)
            } else if l.contains(" WARN ") {
                Style::new().fg(Color::Yellow)
            } else {
                Style::new().gray()
            };
            Line::styled(l.clone(), style)
        })
        .collect();
    f.render_widget(Paragraph::new(lines).block(block), area);
}
