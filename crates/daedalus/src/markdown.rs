//! A small CommonMark → ratatui renderer.
//!
//! `render` turns assistant markdown into `Line`s styled from a
//! [`MarkdownStyle`] (which the TUI builds from the active theme). The subset
//! is deliberate: headings, emphasis, inline/fenced code, bullet/ordered lists,
//! blockquotes, horizontal rules, and links rendered as `label (url)`. Syntax
//! highlighting and GFM tables/task-lists are out of scope.

use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

/// Ratatui styles for each markdown element, sourced from theme tokens.
#[derive(Debug, Clone, Copy)]
pub struct MarkdownStyle {
    pub text: Style,
    pub heading: Style,
    pub code: Style,
    pub code_block: Style,
    pub link: Style,
    pub quote: Style,
    pub bullet: Style,
}

/// Render `text` as markdown into ratatui lines. `width` sizes horizontal
/// rules.
pub fn render(text: &str, width: u16, style: &MarkdownStyle) -> Vec<Line<'static>> {
    let mut r = Renderer {
        width: width.max(1),
        st: *style,
        out: Vec::new(),
        cur: Vec::new(),
        inline: Vec::new(),
        lists: Vec::new(),
        quote: 0,
        code: false,
        code_buf: String::new(),
        links: Vec::new(),
    };
    for event in Parser::new_ext(text, Options::empty()) {
        r.event(event);
    }
    r.finish()
}

struct Renderer {
    width: u16,
    st: MarkdownStyle,
    out: Vec<Line<'static>>,
    /// The line currently being built.
    cur: Vec<Span<'static>>,
    /// Nested inline styles (emphasis/strong/link/heading); base is `st.text`.
    inline: Vec<Style>,
    /// One entry per open list: the next ordered number, or `None` for bullets.
    lists: Vec<Option<u64>>,
    quote: usize,
    code: bool,
    code_buf: String,
    /// Open links as `(destination, label-so-far)`.
    links: Vec<(String, String)>,
}

impl Renderer {
    fn event(&mut self, event: Event<'_>) {
        match event {
            Event::Start(tag) => self.start(tag),
            Event::End(tag) => self.end(tag),
            Event::Text(t) => {
                if self.code {
                    self.code_buf.push_str(&t);
                } else {
                    let style = self.current_style();
                    self.push_text(&t, style);
                }
            }
            Event::Code(c) => {
                let style = self.st.code;
                self.push_text(&c, style);
            }
            Event::SoftBreak => {
                let style = self.current_style();
                self.push_text(" ", style);
            }
            Event::HardBreak => self.flush(),
            Event::Rule => {
                self.flush();
                self.blank();
                let rule = "─".repeat(self.width as usize);
                self.out.push(Line::from(Span::styled(rule, self.st.quote)));
            }
            Event::InlineMath(c) | Event::DisplayMath(c) => {
                let style = self.st.code;
                self.push_text(&c, style);
            }
            Event::TaskListMarker(done) => {
                let style = self.st.text;
                self.push_text(if done { "[x] " } else { "[ ] " }, style);
            }
            // Raw HTML and footnote references have no terminal rendering.
            Event::Html(_) | Event::InlineHtml(_) | Event::FootnoteReference(_) => {}
        }
    }

    fn start(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => {
                self.flush();
                self.blank();
            }
            Tag::Heading { level, .. } => {
                self.flush();
                self.blank();
                let mut style = self.st.heading;
                if level == HeadingLevel::H1 {
                    style = style.add_modifier(Modifier::UNDERLINED);
                }
                self.inline.push(style);
            }
            Tag::BlockQuote(_) => {
                self.flush();
                self.blank();
                self.quote += 1;
            }
            Tag::CodeBlock(_) => {
                self.flush();
                self.blank();
                self.code = true;
            }
            Tag::List(start) => {
                self.flush();
                if self.lists.is_empty() {
                    self.blank();
                }
                self.lists.push(start);
            }
            Tag::Item => {
                self.flush();
                let indent = "  ".repeat(self.lists.len().saturating_sub(1));
                let marker = match self.lists.last_mut() {
                    Some(Some(n)) => {
                        let m = format!("{n}. ");
                        *n += 1;
                        m
                    }
                    _ => "• ".to_string(),
                };
                self.cur
                    .push(Span::styled(format!("{indent}{marker}"), self.st.bullet));
            }
            Tag::Emphasis => {
                let style = self.current_style().add_modifier(Modifier::ITALIC);
                self.inline.push(style);
            }
            Tag::Strong => {
                let style = self.current_style().add_modifier(Modifier::BOLD);
                self.inline.push(style);
            }
            Tag::Link { dest_url, .. } => {
                self.inline.push(self.st.link);
                self.links.push((dest_url.to_string(), String::new()));
            }
            _ => {}
        }
    }

    fn end(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph => self.flush(),
            TagEnd::Heading(_) => {
                self.flush();
                self.inline.pop();
            }
            TagEnd::BlockQuote(_) => {
                self.flush();
                self.quote = self.quote.saturating_sub(1);
            }
            TagEnd::CodeBlock => {
                let buf = std::mem::take(&mut self.code_buf);
                self.code = false;
                for line in buf.trim_end_matches('\n').split('\n') {
                    self.out.push(Line::from(Span::styled(
                        format!("  {line}"),
                        self.st.code_block,
                    )));
                }
                self.blank();
            }
            TagEnd::List(_) => {
                self.flush();
                self.lists.pop();
                if self.lists.is_empty() {
                    self.blank();
                }
            }
            TagEnd::Item => self.flush(),
            TagEnd::Emphasis | TagEnd::Strong => {
                self.inline.pop();
            }
            TagEnd::Link => {
                let link = self.links.pop();
                self.inline.pop();
                if let Some((dest, label)) = link {
                    if !dest.is_empty() && dest != label {
                        let style = self.st.link;
                        self.push_text(&format!(" ({dest})"), style);
                    }
                }
            }
            _ => {}
        }
    }

    fn current_style(&self) -> Style {
        self.inline.last().copied().unwrap_or(self.st.text)
    }

    fn push_text(&mut self, text: &str, style: Style) {
        self.cur.push(Span::styled(text.to_string(), style));
        if let Some(link) = self.links.last_mut() {
            link.1.push_str(text);
        }
    }

    /// Push the current line (with any blockquote prefix) into the output.
    fn flush(&mut self) {
        if self.cur.is_empty() {
            return;
        }
        let mut spans = Vec::new();
        for _ in 0..self.quote {
            spans.push(Span::styled("│ ", self.st.quote));
        }
        spans.extend(std::mem::take(&mut self.cur));
        self.out.push(Line::from(spans));
    }

    /// Insert a blank separator line when the previous line has content.
    fn blank(&mut self) {
        if let Some(last) = self.out.last() {
            if !last.spans.is_empty() {
                self.out.push(Line::default());
            }
        }
    }

    fn finish(mut self) -> Vec<Line<'static>> {
        self.flush();
        while matches!(self.out.last(), Some(l) if l.spans.is_empty()) {
            self.out.pop();
        }
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn style() -> MarkdownStyle {
        MarkdownStyle {
            text: Style::default(),
            heading: Style::default().add_modifier(Modifier::BOLD),
            code: Style::default().fg(ratatui::style::Color::Yellow),
            code_block: Style::default().fg(ratatui::style::Color::Yellow),
            link: Style::default().add_modifier(Modifier::UNDERLINED),
            quote: Style::default().add_modifier(Modifier::DIM),
            bullet: Style::default(),
        }
    }

    fn plain(lines: &[Line<'static>]) -> Vec<String> {
        lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect()
    }

    #[test]
    fn emphasis_and_inline_code() {
        let lines = render("a **b** *c* `d` e", 40, &style());
        assert_eq!(plain(&lines), vec!["a b c d e"]);
        let spans = &lines[0].spans;
        assert!(spans[1].style.add_modifier.contains(Modifier::BOLD));
        assert!(spans[3].style.add_modifier.contains(Modifier::ITALIC));
        assert_eq!(spans[5].style.fg, Some(ratatui::style::Color::Yellow));
    }

    #[test]
    fn heading_is_its_own_bold_line() {
        let lines = render("# Title\n\nbody", 40, &style());
        assert_eq!(plain(&lines), vec!["Title", "", "body"]);
        assert!(lines[0].spans[0]
            .style
            .add_modifier
            .contains(Modifier::BOLD));
        assert!(lines[0].spans[0]
            .style
            .add_modifier
            .contains(Modifier::UNDERLINED));
    }

    #[test]
    fn fenced_code_block_keeps_lines_and_drops_fences() {
        let lines = render("```rust\nfn main() {}\nlet x = 1;\n```\n", 40, &style());
        assert_eq!(plain(&lines), vec!["  fn main() {}", "  let x = 1;"]);
        assert_eq!(
            lines[0].spans[0].style.fg,
            Some(ratatui::style::Color::Yellow)
        );
    }

    #[test]
    fn bullet_and_ordered_lists_with_nesting() {
        let lines = render(
            "- one\n- two\n  - nested\n\n1. first\n2. second\n",
            40,
            &style(),
        );
        assert_eq!(
            plain(&lines),
            vec!["• one", "• two", "  • nested", "", "1. first", "2. second"]
        );
    }

    #[test]
    fn blockquote_prefixes_lines() {
        let lines = render("> quoted\n", 40, &style());
        assert_eq!(plain(&lines), vec!["│ quoted"]);
    }

    #[test]
    fn rule_spans_the_width() {
        let lines = render("a\n\n---\n", 10, &style());
        assert_eq!(plain(&lines)[2], "──────────");
    }

    #[test]
    fn link_renders_label_and_url() {
        let lines = render("[docs](https://example.com)", 60, &style());
        assert_eq!(plain(&lines), vec!["docs (https://example.com)"]);
        // A link whose label equals its destination is not duplicated.
        let lines = render("<https://example.com>", 60, &style());
        assert_eq!(plain(&lines), vec!["https://example.com"]);
    }

    #[test]
    fn soft_break_reflows_and_hard_break_splits() {
        let lines = render("one\ntwo\n\nthree  \nfour", 40, &style());
        assert_eq!(plain(&lines), vec!["one two", "", "three", "four"]);
    }

    #[test]
    fn unterminated_fence_renders_to_end() {
        // A streamed answer may cut off mid-fence; it should still render as a
        // code block rather than literal backticks.
        let lines = render("```\nlet x = 1;\nlet y =", 40, &style());
        assert_eq!(plain(&lines), vec!["  let x = 1;", "  let y ="]);
    }
}
