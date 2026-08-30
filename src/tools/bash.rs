//! The `bash` tool: run a shell command in the workspace directory.

use std::process::Command;

use serde_json::{json, Value};

use super::{arg_string, truncate, Tool, ToolError, ToolOutput};
use crate::workspace::Workspace;

pub struct BashTool {
    pub max_output: usize,
}

impl Tool for BashTool {
    fn name(&self) -> &'static str {
        "bash"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "Shell command to run in the workspace." }
            },
            "required": ["command"]
        })
    }

    fn run(&self, workspace: &Workspace, args: &Value) -> Result<ToolOutput, ToolError> {
        let command = arg_string(args, "command")?;

        let output = Command::new("sh")
            .arg("-c")
            .arg(&command)
            .current_dir(workspace.root())
            .output()
            .map_err(|e| ToolError::Io(e.to_string()))?;

        let mut s = String::new();
        if !output.stdout.is_empty() {
            s.push_str("stdout:\n");
            s.push_str(&String::from_utf8_lossy(&output.stdout));
            s.push('\n');
        }
        if !output.stderr.is_empty() {
            s.push_str("stderr:\n");
            s.push_str(&String::from_utf8_lossy(&output.stderr));
            s.push('\n');
        }
        let code = output
            .status
            .code()
            .map_or_else(|| "signal".to_string(), |c| c.to_string());
        s.push_str(&format!("exit code: {code}"));

        Ok(ToolOutput {
            content: truncate(s, self.max_output),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::Workspace;

    fn setup(name: &str) -> (Workspace, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("crab-bash-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        (Workspace::new(dir.clone()).unwrap(), dir)
    }

    #[test]
    fn captures_stdout_and_exit_code() {
        let (ws, _dir) = setup("out");
        let tool = BashTool { max_output: 1000 };
        let out = tool.run(&ws, &json!({"command": "echo hello"})).unwrap();
        assert!(out.content.contains("hello"));
        assert!(out.content.contains("exit code: 0"));
    }

    #[test]
    fn captures_stderr_and_nonzero_exit() {
        let (ws, _dir) = setup("err");
        let tool = BashTool { max_output: 1000 };
        let out = tool.run(&ws, &json!({"command": "echo boo 1>&2; exit 3"})).unwrap();
        assert!(out.content.contains("stderr:"));
        assert!(out.content.contains("boo"));
        assert!(out.content.contains("exit code: 3"));
    }

    #[test]
    fn runs_in_workspace_dir() {
        let (ws, dir) = setup("pwd");
        std::fs::write(dir.join("marker.txt"), "x").unwrap();
        let tool = BashTool { max_output: 1000 };
        let out = tool.run(&ws, &json!({"command": "ls"})).unwrap();
        assert!(out.content.contains("marker.txt"));
    }

    #[test]
    fn caps_output() {
        let (ws, _dir) = setup("cap");
        let tool = BashTool { max_output: 32 };
        let out = tool
            .run(&ws, &json!({"command": "printf '%.0s1' {1..10000}"}))
            .unwrap();
        assert!(out.content.contains("[truncated"));
    }
}
