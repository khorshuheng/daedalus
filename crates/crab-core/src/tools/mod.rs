//! The tool layer (CRAB-102): the built-in tools, plus external MCP
//! tools (CRAB-133) registered at runtime.
//!
//! A shared `Tool` trait with `name()`, `schema()` (JSON Schema for arguments)
//! and `run(&self, workspace, args) -> Result<ToolOutput, ToolError>`. The
//! `ToolSet` + `resolver` route a `{name, args}` pair to the correct executor.

pub mod bash;
pub mod edit;
pub mod mutation;
pub mod read;
pub mod resolver;
pub mod search;
pub mod write;

use std::path::Path;

use futures::future::BoxFuture;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::workspace::Workspace;

/// The result of running a tool: human-readable text fed back to the model.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: String,
}

/// A typed error raised by a tool or the resolver.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ToolError {
    /// Missing / malformed argument.
    #[error("bad arguments: {0}")]
    Argument(String),
    /// The referenced file does not exist.
    #[error("not found: {0}")]
    NotFound(String),
    /// The operation is invalid (e.g. edit matched zero or many times).
    #[error("invalid: {0}")]
    Invalid(String),
    /// I/O failure.
    #[error("io error: {0}")]
    Io(String),
    /// A command failed (non-zero exit or killed by signal).
    #[error("{0}")]
    Command(String),
    /// A command exceeded its timeout.
    #[error("command timed out: {0}")]
    Timeout(String),
    /// The tool was interrupted by a cancellation request.
    #[error("cancelled")]
    Cancelled,
}
/// A tool. Implementations must be cheap to construct and stateless apart
/// from their output cap. `run` is async (CRAB-130): the built-ins wrap their
/// blocking bodies in `spawn_blocking`; an MCP tool (CRAB-133) implements it
/// natively async behind the same trait. `name` returns `&str` (not
/// `&'static str`) because MCP tool names are runtime data.
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    /// JSON Schema describing the accepted arguments.
    fn schema(&self) -> Value;
    fn run<'a>(
        &'a self,
        workspace: &'a Workspace,
        args: &'a Value,
        cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<ToolOutput, ToolError>>;
}

/// Validate `args` against the small JSON-Schema subset this crate's tools
/// declare (CRAB-107 #9): an object with `properties` (string/integer/array
/// of objects, optional `minimum`), `required`, and `oneOf`. The tools are a
/// closed set we author, so validating against exactly the subset we emit
/// (rather than pulling a full JSON-Schema engine) is sufficient and keeps
/// bad model args from reaching executors. Returns a clear `ToolError`
/// describing the first violation.
pub(crate) fn validate_args(schema: &Value, args: &Value) -> Result<(), ToolError> {
    if schema.get("type").and_then(Value::as_str) != Some("object") {
        return Err(ToolError::Argument("schema must describe an object".into()));
    }
    let obj = args
        .as_object()
        .ok_or_else(|| ToolError::Argument("arguments must be a JSON object".into()))?;

    // oneOf: at least one branch must validate (used by edit: edits XOR oldText/newText).
    if let Some(one_of) = schema.get("oneOf").and_then(Value::as_array) {
        let mut any = false;
        for branch in one_of {
            if let Some(required) = branch.get("required").and_then(Value::as_array) {
                if required
                    .iter()
                    .all(|k| k.as_str().map(|k| obj.contains_key(k)).unwrap_or(false))
                {
                    any = true;
                    break;
                }
            }
        }
        if !any {
            let branch = one_of.first().cloned().unwrap_or(Value::Null);
            let required = branch
                .get("required")
                .and_then(Value::as_array)
                .map(|r| {
                    r.iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            return Err(ToolError::Argument(format!(
                "one of these argument sets is required: {required}"
            )));
        }
    }

    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        for key in required {
            if let Some(key) = key.as_str() {
                if !obj.contains_key(key) {
                    return Err(ToolError::Argument(format!("missing required '{key}'")));
                }
            }
        }
    }

    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        for (key, prop) in properties {
            if let Some(value) = obj.get(key) {
                check_property(key, prop, value)?;
            }
        }
    }
    Ok(())
}

/// Validate a single property value against its subschema.
fn check_property(key: &str, prop: &Value, value: &Value) -> Result<(), ToolError> {
    let type_name = prop.get("type").and_then(Value::as_str).unwrap_or("");
    let ok = match type_name {
        "string" => value.is_string(),
        "integer" => value.is_i64() || value.is_u64(),
        "array" => {
            if !value.is_array() {
                false
            } else if let Some(items) = prop.get("items") {
                let items_type = items.get("type").and_then(Value::as_str).unwrap_or("");
                value.as_array().is_some_and(|arr| {
                    arr.iter().all(|it| match items_type {
                        "object" => it.is_object(),
                        "string" => it.is_string(),
                        "integer" => it.is_i64() || it.is_u64(),
                        _ => true,
                    })
                })
            } else {
                true
            }
        }
        "object" => value.is_object(),
        _ => true,
    };
    if !ok {
        return Err(ToolError::Argument(format!(
            "'{key}' must be a {type_name}"
        )));
    }
    if let Some(min) = prop.get("minimum").and_then(Value::as_i64) {
        if let Some(n) = value.as_i64() {
            if n < min {
                return Err(ToolError::Argument(format!("'{key}' must be >= {min}")));
            }
        }
    }
    Ok(())
}

/// Truncate `s` to at most `max` bytes keeping the *tail* (last bytes),
/// prepending an ellipsis note when truncated. Suitable for command output,
/// where errors and final results appear at the end.
pub(crate) fn truncate_tail(s: &str, max: usize) -> (String, bool) {
    if s.len() <= max {
        return (s.to_string(), false);
    }
    let mut start = s.len() - max;
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    let kept = &s[start..];
    (
        format!(
            "…[truncated: {} bytes total, showing last {} bytes]\n{}",
            s.len(),
            kept.len(),
            kept
        ),
        true,
    )
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

/// Resolve a tool-supplied path: relative paths join the workspace root, while
/// absolute paths and `..` are honored (the workspace guard was removed).
pub(crate) fn resolve(workspace: &Workspace, rel: &Path) -> Result<std::path::PathBuf, ToolError> {
    workspace.resolve(rel).map_err(ToolError::Io)
}

#[cfg(test)]
mod validate_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn rejects_non_object_args() {
        let schema = json!({"type": "object", "properties": {}});
        let err = validate_args(&schema, &json!([1, 2])).unwrap_err();
        assert!(err.to_string().contains("must be a JSON object"));
    }

    #[test]
    fn rejects_missing_required() {
        let schema = json!({
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        });
        let err = validate_args(&schema, &json!({})).unwrap_err();
        assert!(err.to_string().contains("missing required 'path'"));
    }

    #[test]
    fn rejects_wrong_type() {
        let schema = json!({
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        });
        let err = validate_args(&schema, &json!({"path": 42})).unwrap_err();
        assert!(err.to_string().contains("'path' must be a string"));
    }

    #[test]
    fn enforces_minimum_on_integers() {
        let schema = json!({
            "type": "object",
            "properties": {"timeout": {"type": "integer", "minimum": 1}},
            "required": []
        });
        let err = validate_args(&schema, &json!({"timeout": 0})).unwrap_err();
        assert!(err.to_string().contains("'timeout' must be >= 1"));
        assert!(validate_args(&schema, &json!({"timeout": 5})).is_ok());
    }

    #[test]
    fn accepts_valid_args() {
        let schema = json!({
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "minimum": 1}
            },
            "required": ["command"]
        });
        assert!(validate_args(&schema, &json!({"command": "ls", "timeout": 2})).is_ok());
    }

    #[test]
    fn one_of_requires_at_least_one_branch() {
        let schema = json!({
            "type": "object",
            "properties": {
                "edits": {"type": "array"},
                "oldText": {"type": "string"},
                "newText": {"type": "string"}
            },
            "required": ["path"],
            "oneOf": [
                {"required": ["edits"]},
                {"required": ["oldText", "newText"]}
            ]
        });
        // Neither branch satisfied.
        let err = validate_args(&schema, &json!({"path": "f"})).unwrap_err();
        assert!(err
            .to_string()
            .contains("one of these argument sets is required"));
        // edits branch ok.
        assert!(validate_args(&schema, &json!({"path": "f", "edits": []})).is_ok());
        // oldText+newText branch ok.
        assert!(validate_args(
            &schema,
            &json!({"path": "f", "oldText": "a", "newText": "b"})
        )
        .is_ok());
    }
}
