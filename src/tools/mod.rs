//! The tool layer (CRAB-102): a fixed, closed set of four tools.
//!
//! A shared `Tool` trait with `name()`, `schema()` (JSON Schema for arguments)
//! and `run(&self, workspace, args) -> Result<ToolOutput, ToolError>`. The
//! `ToolSet` + `resolver` route a `{name, args}` pair to the correct executor.

pub mod bash;
pub mod edit;
pub mod read;
pub mod resolver;
pub mod write;

use std::fmt;
use std::path::Path;

use serde_json::Value;

use crate::workspace::Workspace;

/// The result of running a tool: human-readable text fed back to the model.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: String,
}

/// A typed error raised by a tool or the resolver.
#[derive(Debug, Clone)]
pub enum ToolError {
    /// Missing / malformed argument.
    Argument(String),
    /// The referenced file does not exist.
    NotFound(String),
    /// The operation is invalid (e.g. edit matched zero or many times).
    Invalid(String),
    /// I/O failure.
    Io(String),
    /// A path escapes the workspace.
    Escape(String),
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolError::Argument(m) => write!(f, "bad arguments: {m}"),
            ToolError::NotFound(m) => write!(f, "not found: {m}"),
            ToolError::Invalid(m) => write!(f, "invalid: {m}"),
            ToolError::Io(m) => write!(f, "io error: {m}"),
            ToolError::Escape(m) => write!(f, "path escaped: {m}"),
        }
    }
}

impl std::error::Error for ToolError {}

/// A built-in tool. Implementations must be cheap to construct and stateless
/// apart from their output cap.
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    /// JSON Schema describing the accepted arguments.
    fn schema(&self) -> Value;
    fn run(&self, workspace: &Workspace, args: &Value) -> Result<ToolOutput, ToolError>;
}

/// Truncate `s` to at most `max` bytes on a UTF-8 boundary, appending an
/// ellipsis note when truncated, so a single result cannot blow up the context.
pub(crate) fn truncate(s: String, max: usize) -> String {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…[truncated: {} bytes total]", &s[..end], s.len())
}

/// Require a string argument.
pub(crate) fn arg_string(args: &Value, key: &str) -> Result<String, ToolError> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| ToolError::Argument(format!("'{key}' must be a string")))
}

/// Optional integer argument.
pub(crate) fn arg_usize(args: &Value, key: &str) -> Result<Option<usize>, ToolError> {
    match args.get(key) {
        None => Ok(None),
        Some(Value::Number(n)) => n
            .as_u64()
            .map(|v| Some(v as usize))
            .ok_or_else(|| ToolError::Argument(format!("'{key}' must be a non-negative integer"))),
        Some(_) => Err(ToolError::Argument(format!(
            "'{key}' must be a non-negative integer"
        ))),
    }
}

/// Resolve a tool-supplied path relative to the workspace, mapping an escape to
/// a `ToolError::Escape`.
pub(crate) fn resolve(workspace: &Workspace, rel: &Path) -> Result<std::path::PathBuf, ToolError> {
    workspace.resolve(rel).map_err(ToolError::Escape)
}
