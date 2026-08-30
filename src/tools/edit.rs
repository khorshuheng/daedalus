//! The `edit` tool: apply precise, validated text replacements to a file.
//!
//! Each `oldText` must match exactly once in the current file content; zero or
//! multiple matches are an error. Multiple disjoint edits can be supplied in a
//! single call via the `edits` array (or a single top-level `oldText`/`newText`).

use std::path::Path;

use serde_json::{json, Value};

use super::{arg_string, resolve, Tool, ToolError, ToolOutput};
use crate::workspace::Workspace;

pub struct EditTool {
    pub max_output: usize,
}

impl Tool for EditTool {
    fn name(&self) -> &'static str {
        "edit"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File to edit." },
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": { "type": "string" },
                            "newText": { "type": "string" }
                        },
                        "required": ["oldText", "newText"]
                    },
                    "description": "Disjoint replacements to apply in order."
                },
                "oldText": { "type": "string", "description": "Alternative to 'edits' for a single replacement." },
                "newText": { "type": "string" }
            },
            "required": ["path"],
            "oneOf": [
                { "required": ["edits"] },
                { "required": ["oldText", "newText"] }
            ]
        })
    }

    fn run(&self, workspace: &Workspace, args: &Value) -> Result<ToolOutput, ToolError> {
        let path = arg_string(args, "path")?;
        let edits: Vec<(String, String)> = if let Some(edits) = args.get("edits").and_then(|v| v.as_array())
        {
            if edits.is_empty() {
                return Err(ToolError::Argument("'edits' must not be empty".into()));
            }
            edits
                .iter()
                .map(|e| {
                    let old = arg_string(e, "oldText")?;
                    let new = arg_string(e, "newText")?;
                    Ok((old, new))
                })
                .collect::<Result<_, ToolError>>()?
        } else {
            vec![(arg_string(args, "oldText")?, arg_string(args, "newText")?)]
        };

        let resolved = resolve(workspace, Path::new(&path))?;
        let text = match std::fs::read_to_string(&resolved) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(ToolError::NotFound(path));
            }
            Err(e) => return Err(ToolError::Io(e.to_string())),
        };

        let mut out = text;
        for (old, new) in &edits {
            if old.is_empty() {
                return Err(ToolError::Invalid(
                    "'oldText' must not be empty".into(),
                ));
            }
            let count = out.matches(old.as_str()).count();
            match count {
                0 => {
                    return Err(ToolError::Invalid(format!(
                        "'oldText' not found in '{}': {old:?}",
                        path
                    )))
                }
                1 => {
                    out = out.replacen(old.as_str(), new.as_str(), 1);
                }
                n => {
                    return Err(ToolError::Invalid(format!(
                        "'oldText' matched {n} times in '{}' (expected exactly 1): {old:?}",
                        path
                    )))
                }
            }
        }

        std::fs::write(&resolved, &out).map_err(|e| ToolError::Io(e.to_string()))?;

        Ok(ToolOutput {
            content: format!(
                "applied {} edit(s) to {}",
                edits.len(),
                path
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::Workspace;

    fn setup(name: &str, contents: &str) -> (Workspace, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("crab-edit-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.txt"), contents).unwrap();
        (Workspace::new(dir.clone()).unwrap(), dir)
    }

    fn read(dir: &std::path::Path) -> String {
        std::fs::read_to_string(dir.join("a.txt")).unwrap()
    }

    #[test]
    fn unique_match_replaces() {
        let (ws, dir) = setup("unique", "hello world");
        let tool = EditTool { max_output: 1000 };
        tool.run(&ws, &json!({"path": "a.txt", "oldText": "world", "newText": "there"}))
            .unwrap();
        assert_eq!(read(&dir), "hello there");
    }

    #[test]
    fn zero_match_is_error() {
        let (ws, _dir) = setup("zero", "hello");
        let tool = EditTool { max_output: 1000 };
        let err = tool
            .run(&ws, &json!({"path": "a.txt", "oldText": "zzz", "newText": "x"}))
            .unwrap_err();
        assert!(matches!(err, ToolError::Invalid(_)));
    }

    #[test]
    fn multiple_match_is_error() {
        let (ws, _dir) = setup("multi", "a a a");
        let tool = EditTool { max_output: 1000 };
        let err = tool
            .run(&ws, &json!({"path": "a.txt", "oldText": "a", "newText": "b"}))
            .unwrap_err();
        assert!(matches!(err, ToolError::Invalid(_)));
    }

    #[test]
    fn multiple_disjoint_edits() {
        let (ws, dir) = setup("disjoint", "one two three");
        let tool = EditTool { max_output: 1000 };
        tool.run(
            &ws,
            &json!({"path": "a.txt", "edits": [
                {"oldText": "one", "newText": "1"},
                {"oldText": "three", "newText": "3"}
            ]}),
        )
        .unwrap();
        assert_eq!(read(&dir), "1 two 3");
    }
}
