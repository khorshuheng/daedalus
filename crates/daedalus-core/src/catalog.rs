//! Static model context windows.
//!
//! A model's context window is a property of the model, not of the endpoint.
//! Neither the provider API nor a provider's model listing reports one for
//! every model — DeepSeek, OpenAI, and Anthropic name their models without a
//! window — so the footer cannot always derive a usage percentage from the
//! wire alone. This module holds the windows daedalus knows by hand.
//!
//! A `(provider, model)` pair that is absent has an unknown window. Callers
//! must treat that as "percentage unavailable" rather than substituting a
//! budget, so a made-up denominator never masquerades as a real percentage.

/// The context window (input tokens) for a known `(provider, model)`, or
/// `None` when daedalus has no figure for it.
///
/// `provider` is the registry name (e.g. `"deepseek"`); `model` is the model
/// id as sent on the wire.
pub fn context_window(provider: &str, model: &str) -> Option<usize> {
    match provider {
        "deepseek" => deepseek_window(model),
        _ => None,
    }
}

/// DeepSeek's published window is 64K input tokens for both the chat and the
/// reasoning model; the endpoint does not report it, so it is pinned here.
fn deepseek_window(model: &str) -> Option<usize> {
    match model {
        "deepseek-chat" | "deepseek-reasoner" => Some(64_000),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepseek_models_report_their_window() {
        assert_eq!(context_window("deepseek", "deepseek-chat"), Some(64_000));
        assert_eq!(
            context_window("deepseek", "deepseek-reasoner"),
            Some(64_000)
        );
    }

    #[test]
    fn unknown_models_and_providers_have_no_window() {
        assert_eq!(context_window("deepseek", "deepseek-typo"), None);
        assert_eq!(context_window("openai", "gpt-4o"), None);
        assert_eq!(context_window("anthropic", "claude-sonnet"), None);
    }
}
