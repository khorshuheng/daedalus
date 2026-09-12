//! TUI theming (CRAB-140): a data-only `Theme` resolved from `config.toml`.
//!
//! This module deliberately has no ratatui/terminal dependency: it produces
//! plain color/style data ([`Theme`], [`StyleSpec`]) that the `crab` binary
//! maps onto ratatui styles. Colors are terminal palette indices, RGB, or the
//! terminal default, so themes stay portable.
//!
//! Config shape:
//!
//! ```toml
//! [theme]
//! name = "dark"                 # dark | light (default: COLORFGBG detection)
//! [theme.vars]
//! accent = "#00aaff"
//! [theme.colors]
//! user      = "cyan"
//! thinking  = { fg = "dark_gray", italic = true }
//! selection = { fg = "black", bg = "accent", bold = true }
//! ```

use std::collections::BTreeMap;

use serde::Deserialize;

/// A terminal color, independent of any rendering crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThemeColor {
    /// The terminal's default foreground/background (`""`).
    #[default]
    Default,
    /// An ANSI/256-color palette index (0..=255).
    Indexed(u8),
    /// A 24-bit RGB color (`#rrggbb`).
    Rgb(u8, u8, u8),
}

/// Style modifiers a token can carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Modifiers {
    pub bold: bool,
    pub dim: bool,
    pub italic: bool,
    pub underlined: bool,
    pub reversed: bool,
    pub crossed_out: bool,
}

/// The foreground/background/modifiers of one token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StyleSpec {
    pub fg: ThemeColor,
    pub bg: ThemeColor,
    pub modifiers: Modifiers,
}

/// The role-based token set (CRAB-140). Markdown tokens are added by CRAB-145.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Token {
    Text,
    User,
    Assistant,
    Thinking,
    Tool,
    ToolOk,
    ToolErr,
    Notice,
    Border,
    Title,
    Spinner,
    Input,
    Selection,
}

impl Token {
    /// Number of tokens.
    pub const COUNT: usize = 13;

    /// `(toml key, token)` for every token — one source of truth for parsing,
    /// formatting, and error messages.
    pub const ALL: [(&'static str, Token); Self::COUNT] = [
        ("text", Token::Text),
        ("user", Token::User),
        ("assistant", Token::Assistant),
        ("thinking", Token::Thinking),
        ("tool", Token::Tool),
        ("tool_ok", Token::ToolOk),
        ("tool_err", Token::ToolErr),
        ("notice", Token::Notice),
        ("border", Token::Border),
        ("title", Token::Title),
        ("spinner", Token::Spinner),
        ("input", Token::Input),
        ("selection", Token::Selection),
    ];

    fn index(self) -> usize {
        self as usize
    }
}

/// A resolved theme: a base preset plus any per-token overrides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Theme {
    pub name: String,
    tokens: [StyleSpec; Token::COUNT],
}

impl Default for Theme {
    fn default() -> Self {
        Self::dark()
    }
}

impl Theme {
    /// The style for one token.
    pub fn token(&self, token: Token) -> StyleSpec {
        self.tokens[token.index()]
    }

    fn set(&mut self, token: Token, spec: StyleSpec) {
        self.tokens[token.index()] = spec;
    }

    /// A built-in preset by name, or `None`.
    pub fn builtin(name: &str) -> Option<Theme> {
        match name {
            "dark" => Some(Theme::dark()),
            "light" => Some(Theme::light()),
            _ => None,
        }
    }

    /// The `dark` preset — reproduces crab's original hardcoded colors.
    pub fn dark() -> Theme {
        let mut t = Theme {
            name: "dark".to_string(),
            tokens: [StyleSpec::default(); Token::COUNT],
        };
        t.set(Token::User, fg(6));
        t.set(Token::Assistant, fg(7));
        t.set(Token::Thinking, italic(8));
        t.set(Token::Tool, fg(8));
        t.set(Token::ToolOk, fg(8));
        t.set(Token::ToolErr, fg(8));
        t.set(Token::Notice, fg(3));
        t.set(Token::Selection, bold());
        t
    }

    /// The `light` preset — for light terminal backgrounds.
    pub fn light() -> Theme {
        let mut t = Theme {
            name: "light".to_string(),
            tokens: [StyleSpec::default(); Token::COUNT],
        };
        t.set(Token::User, fg(4));
        t.set(Token::Assistant, fg(0));
        t.set(Token::Thinking, italic(8));
        t.set(Token::Tool, fg(8));
        t.set(Token::ToolOk, fg(8));
        t.set(Token::ToolErr, fg(8));
        t.set(Token::Notice, fg(5));
        t.set(Token::Selection, bold());
        t
    }
}

fn fg(index: u8) -> StyleSpec {
    StyleSpec {
        fg: ThemeColor::Indexed(index),
        ..Default::default()
    }
}

fn italic(index: u8) -> StyleSpec {
    StyleSpec {
        fg: ThemeColor::Indexed(index),
        modifiers: Modifiers {
            italic: true,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn bold() -> StyleSpec {
    StyleSpec {
        modifiers: Modifiers {
            bold: true,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Parse a color value: `""` (default), `#rrggbb`, `0..=255`, an ANSI name, or
/// a `[theme.vars]` reference.
pub fn parse_color(s: &str, vars: &BTreeMap<String, ThemeColor>) -> Result<ThemeColor, String> {
    let t = s.trim();
    if t.is_empty() {
        return Ok(ThemeColor::Default);
    }
    if let Some(hex) = t.strip_prefix('#') {
        if hex.len() == 6 {
            if let Ok(n) = u32::from_str_radix(hex, 16) {
                return Ok(ThemeColor::Rgb(
                    ((n >> 16) & 0xff) as u8,
                    ((n >> 8) & 0xff) as u8,
                    (n & 0xff) as u8,
                ));
            }
        }
        return Err(format!("invalid hex color '{s}' (expected #rrggbb)"));
    }
    if let Ok(n) = t.parse::<u16>() {
        if n <= 255 {
            return Ok(ThemeColor::Indexed(n as u8));
        }
        return Err(format!("color index {n} out of range (0..=255)"));
    }
    if let Some(idx) = ansi_index(t) {
        return Ok(ThemeColor::Indexed(idx));
    }
    if let Some(c) = vars.get(t) {
        return Ok(*c);
    }
    Err(format!(
        "unknown color '{s}' (use \"#rrggbb\", 0..=255, an ANSI name, or a [theme.vars] entry)"
    ))
}

fn ansi_index(name: &str) -> Option<u8> {
    Some(match name.to_ascii_lowercase().as_str() {
        "black" => 0,
        "red" => 1,
        "green" => 2,
        "yellow" => 3,
        "blue" => 4,
        "magenta" => 5,
        "cyan" => 6,
        "white" => 7,
        "bright_black" | "gray" | "grey" | "dark_gray" | "dark_grey" => 8,
        "bright_red" => 9,
        "bright_green" => 10,
        "bright_yellow" => 11,
        "bright_blue" => 12,
        "bright_magenta" => 13,
        "bright_cyan" => 14,
        "bright_white" => 15,
        _ => return None,
    })
}

/// A single token value in `[theme.colors]`: either a color string or an
/// inline table with per-field overrides.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StyleValue {
    Color(String),
    Inline(InlineStyle),
}

/// Inline token table; every field is optional and inherits the base preset.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InlineStyle {
    pub fg: Option<String>,
    pub bg: Option<String>,
    pub bold: Option<bool>,
    pub dim: Option<bool>,
    pub italic: Option<bool>,
    pub underlined: Option<bool>,
    pub reversed: Option<bool>,
    pub crossed_out: Option<bool>,
}

/// Per-token overrides; unknown token keys are rejected.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThemeColors {
    pub text: Option<StyleValue>,
    pub user: Option<StyleValue>,
    pub assistant: Option<StyleValue>,
    pub thinking: Option<StyleValue>,
    pub tool: Option<StyleValue>,
    pub tool_ok: Option<StyleValue>,
    pub tool_err: Option<StyleValue>,
    pub notice: Option<StyleValue>,
    pub border: Option<StyleValue>,
    pub title: Option<StyleValue>,
    pub spinner: Option<StyleValue>,
    pub input: Option<StyleValue>,
    pub selection: Option<StyleValue>,
}

impl ThemeColors {
    fn get(&self, token: Token) -> Option<&StyleValue> {
        match token {
            Token::Text => self.text.as_ref(),
            Token::User => self.user.as_ref(),
            Token::Assistant => self.assistant.as_ref(),
            Token::Thinking => self.thinking.as_ref(),
            Token::Tool => self.tool.as_ref(),
            Token::ToolOk => self.tool_ok.as_ref(),
            Token::ToolErr => self.tool_err.as_ref(),
            Token::Notice => self.notice.as_ref(),
            Token::Border => self.border.as_ref(),
            Token::Title => self.title.as_ref(),
            Token::Spinner => self.spinner.as_ref(),
            Token::Input => self.input.as_ref(),
            Token::Selection => self.selection.as_ref(),
        }
    }

    fn overlay(&mut self, higher: &ThemeColors) {
        macro_rules! take {
            ($f:ident) => {
                if higher.$f.is_some() {
                    self.$f = higher.$f.clone();
                }
            };
        }
        take!(text);
        take!(user);
        take!(assistant);
        take!(thinking);
        take!(tool);
        take!(tool_ok);
        take!(tool_err);
        take!(notice);
        take!(border);
        take!(title);
        take!(spinner);
        take!(input);
        take!(selection);
    }
}

/// The `[theme]` table, before resolution.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThemePartial {
    pub name: Option<String>,
    #[serde(default)]
    pub vars: BTreeMap<String, String>,
    #[serde(default)]
    pub colors: ThemeColors,
}

impl ThemePartial {
    /// Merge `higher` over `self`: a set field in `higher` wins. Unlike scalar
    /// config keys, the theme table merges per field, so a `--theme` name
    /// override does not discard token overrides from the config file.
    pub fn overlay(&mut self, higher: &ThemePartial) {
        if higher.name.is_some() {
            self.name = higher.name.clone();
        }
        for (k, v) in &higher.vars {
            self.vars.insert(k.clone(), v.clone());
        }
        self.colors.overlay(&higher.colors);
    }
}

/// Resolve a partial theme against a base preset.
pub fn resolve(partial: &ThemePartial, detected: &str) -> Result<Theme, String> {
    let name = partial.name.clone().unwrap_or_else(|| detected.to_string());
    let mut theme = Theme::builtin(&name)
        .ok_or_else(|| format!("unknown theme '{name}' (supported: dark, light)"))?;

    let mut vars: BTreeMap<String, ThemeColor> = BTreeMap::new();
    for (k, v) in &partial.vars {
        vars.insert(k.clone(), parse_color(v, &vars)?);
    }

    for (_, token) in Token::ALL {
        if let Some(value) = partial.colors.get(token) {
            let mut spec = theme.token(token);
            match value {
                StyleValue::Color(s) => spec.fg = parse_color(s, &vars)?,
                StyleValue::Inline(inl) => {
                    if let Some(fg) = &inl.fg {
                        spec.fg = parse_color(fg, &vars)?;
                    }
                    if let Some(bg) = &inl.bg {
                        spec.bg = parse_color(bg, &vars)?;
                    }
                    if let Some(v) = inl.bold {
                        spec.modifiers.bold = v;
                    }
                    if let Some(v) = inl.dim {
                        spec.modifiers.dim = v;
                    }
                    if let Some(v) = inl.italic {
                        spec.modifiers.italic = v;
                    }
                    if let Some(v) = inl.underlined {
                        spec.modifiers.underlined = v;
                    }
                    if let Some(v) = inl.reversed {
                        spec.modifiers.reversed = v;
                    }
                    if let Some(v) = inl.crossed_out {
                        spec.modifiers.crossed_out = v;
                    }
                }
            }
            theme.set(token, spec);
        }
    }
    Ok(theme)
}

/// Best-effort dark/light detection from `COLORFGBG` (no terminal I/O).
/// Missing or malformed values fall back to `dark`.
pub fn detect_scheme() -> &'static str {
    match std::env::var("COLORFGBG") {
        Ok(v) => scheme_from_colorfgbg(&v),
        Err(_) => "dark",
    }
}

/// `COLORFGBG` is `fg;bg` (sometimes with more fields); the last field is the
/// background palette index: `< 8` is dark, `>= 8` is light.
fn scheme_from_colorfgbg(value: &str) -> &'static str {
    match value
        .rsplit(';')
        .next()
        .and_then(|s| s.trim().parse::<u32>().ok())
    {
        Some(n) if n >= 8 => "light",
        _ => "dark",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_vars() -> BTreeMap<String, ThemeColor> {
        BTreeMap::new()
    }

    #[test]
    fn parse_color_forms() {
        assert_eq!(parse_color("", &no_vars()).unwrap(), ThemeColor::Default);
        assert_eq!(
            parse_color("#00aaff", &no_vars()).unwrap(),
            ThemeColor::Rgb(0, 0xaa, 0xff)
        );
        assert_eq!(
            parse_color("240", &no_vars()).unwrap(),
            ThemeColor::Indexed(240)
        );
        assert_eq!(
            parse_color("cyan", &no_vars()).unwrap(),
            ThemeColor::Indexed(6)
        );
        assert_eq!(
            parse_color("dark_gray", &no_vars()).unwrap(),
            ThemeColor::Indexed(8)
        );
        assert_eq!(
            parse_color("bright_red", &no_vars()).unwrap(),
            ThemeColor::Indexed(9)
        );
        let mut vars = BTreeMap::new();
        vars.insert("accent".into(), ThemeColor::Rgb(1, 2, 3));
        assert_eq!(
            parse_color("accent", &vars).unwrap(),
            ThemeColor::Rgb(1, 2, 3)
        );
        for bad in ["nope", "#gg0000", "256", "#12345"] {
            assert!(parse_color(bad, &no_vars()).is_err(), "{bad}");
        }
    }

    #[test]
    fn builtin_presets_match_the_original_colors() {
        let dark = Theme::dark();
        assert_eq!(dark.token(Token::User).fg, ThemeColor::Indexed(6)); // cyan
        assert_eq!(dark.token(Token::Assistant).fg, ThemeColor::Indexed(7)); // white
        assert_eq!(dark.token(Token::Tool).fg, ThemeColor::Indexed(8)); // dark gray
        assert_eq!(dark.token(Token::Notice).fg, ThemeColor::Indexed(3)); // yellow
        assert!(dark.token(Token::Thinking).modifiers.italic);
        assert!(Theme::builtin("light").is_some());
        assert!(Theme::builtin("nope").is_none());
    }

    #[test]
    fn colors_table_parses_strings_and_inline_tables() {
        let partial: ThemePartial = toml::from_str(
            r##"
            name = "dark"
            [vars]
            accent = "#00aaff"
            [colors]
            user = "accent"
            thinking = { fg = "dark_gray", italic = true }
            "##,
        )
        .unwrap();
        let theme = resolve(&partial, "light").unwrap();
        // The explicit name wins over detection.
        assert_eq!(theme.name, "dark");
        assert_eq!(theme.token(Token::User).fg, ThemeColor::Rgb(0, 0xaa, 0xff));
        assert_eq!(theme.token(Token::Thinking).fg, ThemeColor::Indexed(8));
        assert!(theme.token(Token::Thinking).modifiers.italic);
        // Untouched tokens come from the base preset.
        assert_eq!(theme.token(Token::Notice).fg, ThemeColor::Indexed(3));
    }

    #[test]
    fn inline_overrides_inherit_unset_fields() {
        let partial: ThemePartial =
            toml::from_str("[colors]\nselection = { bold = true }\n").unwrap();
        let theme = resolve(&partial, "dark").unwrap();
        // `Selection` in the dark preset is already bold; setting bold keeps it.
        assert!(theme.token(Token::Selection).modifiers.bold);
    }

    #[test]
    fn unknown_token_or_color_is_an_error() {
        let bad_token = toml::from_str::<ThemePartial>("[colors]\nnope = \"red\"\n");
        assert!(bad_token.is_err());
        let partial: ThemePartial = toml::from_str("[colors]\nuser = \"nope\"\n").unwrap();
        let err = resolve(&partial, "dark").unwrap_err();
        assert!(err.contains("unknown color"), "{err}");
        let bad_name: ThemePartial = toml::from_str("name = \"neon\"\n").unwrap();
        let err = resolve(&bad_name, "dark").unwrap_err();
        assert!(err.contains("unknown theme 'neon'"), "{err}");
    }

    #[test]
    fn overlay_merges_name_vars_and_tokens() {
        let mut file: ThemePartial = toml::from_str(
            "name = \"light\"\n[vars]\na = \"red\"\n[colors]\nuser = \"blue\"\ntool = \"green\"\n",
        )
        .unwrap();
        // A flag sets only the name; the file's token overrides must survive.
        let flags: ThemePartial = toml::from_str("name = \"dark\"\n").unwrap();
        file.overlay(&flags);
        let theme = resolve(&file, "light").unwrap();
        assert_eq!(theme.name, "dark");
        assert_eq!(theme.token(Token::User).fg, ThemeColor::Indexed(4)); // file override kept
        assert_eq!(theme.token(Token::Tool).fg, ThemeColor::Indexed(2));
    }

    #[test]
    fn colorfgbg_detection() {
        assert_eq!(scheme_from_colorfgbg("15;0"), "dark");
        assert_eq!(scheme_from_colorfgbg("0;15"), "light");
        assert_eq!(scheme_from_colorfgbg("0;8"), "light");
        assert_eq!(scheme_from_colorfgbg("0;7"), "dark");
        assert_eq!(scheme_from_colorfgbg("garbage"), "dark");
        assert_eq!(scheme_from_colorfgbg(""), "dark");
    }
}
