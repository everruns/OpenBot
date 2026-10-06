//! Which provider and model this Bot runs, read from `shared/model-providers.json`.
//!
//! The same lookup `shared/model_providers.py` and `shared/model-providers.ts` make, read straight
//! as JSON as the file's own `_readme` invites any other language to: `BOT_PROVIDER` and
//! `BOT_MODEL` win, the `bots` row supplies what they leave unset, and a model the file names
//! belongs to the file's provider only.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// This Bot's row in the spec file.
pub const BOT_ID: &str = "agent-everruns";

#[derive(Debug, Deserialize)]
struct Spec {
    providers: HashMap<String, ProviderRow>,
    bots: HashMap<String, BotRow>,
}

#[derive(Debug, Deserialize)]
struct ProviderRow {
    default_model: String,
}

#[derive(Debug, Deserialize)]
struct BotRow {
    provider: String,
    model: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BotSettings {
    pub provider: String,
    pub model: String,
}

/// Where the file is: `BOT_SPEC_PATH`, else `/shared` in the image, else `shared/` in the tree.
pub fn spec_path(lookup: &impl Fn(&str) -> Option<String>) -> PathBuf {
    if let Some(path) = lookup("BOT_SPEC_PATH").filter(|value| !value.trim().is_empty()) {
        return PathBuf::from(path.trim());
    }
    let image = Path::new("/shared/model-providers.json");
    if image.is_file() {
        return image.to_path_buf();
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../shared/model-providers.json")
}

/// Environment over spec file over provider default, blank meaning unset.
pub fn bot_settings(
    spec_json: &str,
    lookup: &impl Fn(&str) -> Option<String>,
) -> Result<BotSettings, String> {
    let spec: Spec = serde_json::from_str(spec_json)
        .map_err(|error| format!("shared/model-providers.json is not readable: {error}"))?;
    let entry = spec.bots.get(BOT_ID).ok_or_else(|| {
        format!("shared/model-providers.json: bots has no {BOT_ID} entry. Add one before this Bot reads the spec file.")
    })?;
    let set = |name: &str| {
        lookup(name)
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    };

    let provider = set("BOT_PROVIDER")
        .map(|value| value.to_lowercase())
        .unwrap_or_else(|| entry.provider.clone());
    let model = match set("BOT_MODEL") {
        Some(model) => model,
        None if provider == entry.provider => entry.model.clone(),
        None => spec
            .providers
            .get(&provider)
            .or_else(|| spec.providers.get("openai"))
            .map(|row| row.default_model.clone())
            .ok_or("shared/model-providers.json: providers has no openai row")?,
    };
    Ok(BotSettings { provider, model })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPEC: &str = r#"{
      "providers": {
        "openai": { "default_model": "gpt-5.5" },
        "anthropic": { "default_model": "claude-sonnet-4-5" }
      },
      "bots": { "agent-everruns": { "provider": "openai", "model": "gpt-4o-mini" } }
    }"#;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |name| map.get(name).cloned()
    }

    #[test]
    fn the_file_decides_when_the_environment_names_nothing() {
        let settings =
            bot_settings(SPEC, &env(&[("BOT_PROVIDER", ""), ("BOT_MODEL", " ")])).unwrap();
        assert_eq!(settings.provider, "openai");
        assert_eq!(settings.model, "gpt-4o-mini");
    }

    #[test]
    fn another_provider_gets_its_own_default_not_the_files_model() {
        let settings = bot_settings(SPEC, &env(&[("BOT_PROVIDER", "Anthropic")])).unwrap();
        assert_eq!(settings.provider, "anthropic");
        assert_eq!(settings.model, "claude-sonnet-4-5");
    }

    #[test]
    fn the_environment_model_wins() {
        let settings = bot_settings(SPEC, &env(&[("BOT_MODEL", "llama3.1:8b")])).unwrap();
        assert_eq!(settings.model, "llama3.1:8b");
    }

    #[test]
    fn the_repository_file_has_this_bots_row() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../shared/model-providers.json");
        let spec = std::fs::read_to_string(path).unwrap();
        bot_settings(&spec, &env(&[])).unwrap();
    }
}
