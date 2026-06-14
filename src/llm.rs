use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fmt;
use std::fs;
#[cfg(feature = "llm-http")]
use std::fs::OpenOptions;
use std::hash::{Hash, Hasher};
#[cfg(feature = "llm-http")]
use std::io::Write;
use std::path::{Path, PathBuf};
#[cfg(feature = "llm-http")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "llm-http")]
use std::time::Duration;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmMessage {
    pub role: String,
    pub content: String,
}

impl LlmMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LlmCompletionRequest {
    pub model: String,
    pub messages: Vec<LlmMessage>,
    pub temperature: f32,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmProviderMetadata {
    pub provider: String,
    pub model: String,
    pub base_url: Option<String>,
    pub prompt_version: Option<String>,
}

pub trait LlmProvider: Clone + Send + Sync + 'static {
    fn complete(&self, request: &LlmCompletionRequest) -> Result<String, LlmError>;
    fn metadata(&self) -> LlmProviderMetadata;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlmError {
    message: String,
}

impl LlmError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for LlmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for LlmError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlmProviderConfig {
    pub provider: String,
    pub model: String,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub fixture_path: Option<PathBuf>,
    pub cache_path: Option<PathBuf>,
    pub timeout_secs: u64,
}

impl LlmProviderConfig {
    pub fn from_env() -> Result<Self, LlmError> {
        let provider = env::var("AGENT_MEMORY_LLM_PROVIDER")
            .unwrap_or_else(|_| "openai-compatible".to_string())
            .to_lowercase();
        let model = env::var("AGENT_MEMORY_LLM_MODEL")
            .map_err(|_| LlmError::new("AGENT_MEMORY_LLM_MODEL is required for llm answerer"))?;
        let base_url = env::var("AGENT_MEMORY_LLM_BASE_URL").ok();
        let api_key = env::var("AGENT_MEMORY_LLM_API_KEY").ok();
        let fixture_path = env::var("AGENT_MEMORY_LLM_FIXTURE").ok().map(PathBuf::from);
        let cache_path = env::var("AGENT_MEMORY_LLM_CACHE").ok().map(PathBuf::from);
        let timeout_secs = env::var("AGENT_MEMORY_LLM_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(60);

        Ok(Self {
            provider,
            model,
            base_url,
            api_key,
            fixture_path,
            cache_path,
            timeout_secs,
        })
    }

    pub fn extractor_from_env() -> Result<Self, LlmError> {
        let provider = env::var("AGENT_MEMORY_EXTRACTOR_LLM_PROVIDER")
            .or_else(|_| env::var("AGENT_MEMORY_LLM_PROVIDER"))
            .unwrap_or_else(|_| "openai-compatible".to_string())
            .to_lowercase();
        let model = env::var("AGENT_MEMORY_EXTRACTOR_LLM_MODEL")
            .or_else(|_| env::var("AGENT_MEMORY_LLM_MODEL"))
            .map_err(|_| {
                LlmError::new(
                    "AGENT_MEMORY_EXTRACTOR_LLM_MODEL (or AGENT_MEMORY_LLM_MODEL) is required",
                )
            })?;
        let base_url = env::var("AGENT_MEMORY_EXTRACTOR_LLM_BASE_URL")
            .or_else(|_| env::var("AGENT_MEMORY_LLM_BASE_URL"))
            .ok();
        let api_key = env::var("AGENT_MEMORY_EXTRACTOR_LLM_API_KEY")
            .or_else(|_| env::var("AGENT_MEMORY_LLM_API_KEY"))
            .ok();
        let fixture_path = env::var("AGENT_MEMORY_EXTRACTOR_LLM_FIXTURE")
            .or_else(|_| env::var("AGENT_MEMORY_LLM_FIXTURE"))
            .ok()
            .map(PathBuf::from);
        let cache_path = env::var("AGENT_MEMORY_EXTRACTOR_LLM_CACHE")
            .or_else(|_| env::var("AGENT_MEMORY_LLM_CACHE"))
            .ok()
            .map(PathBuf::from);
        let timeout_secs = env::var("AGENT_MEMORY_EXTRACTOR_LLM_TIMEOUT_SECS")
            .or_else(|_| env::var("AGENT_MEMORY_LLM_TIMEOUT_SECS"))
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(60);

        Ok(Self {
            provider,
            model,
            base_url,
            api_key,
            fixture_path,
            cache_path,
            timeout_secs,
        })
    }
}

#[derive(Clone, Debug)]
pub enum ConfiguredLlmProvider {
    Fixture(FixtureLlmProvider),
    #[cfg(feature = "llm-http")]
    OpenAiCompatible(OpenAiCompatibleProvider),
}

impl ConfiguredLlmProvider {
    pub fn from_env() -> Result<Self, LlmError> {
        Self::from_config(LlmProviderConfig::from_env()?)
    }

    pub fn from_config(config: LlmProviderConfig) -> Result<Self, LlmError> {
        match config.provider.as_str() {
            "fixture" | "offline-fixture" => {
                let path = config.fixture_path.ok_or_else(|| {
                    LlmError::new("AGENT_MEMORY_LLM_FIXTURE is required for fixture provider")
                })?;
                Ok(Self::Fixture(FixtureLlmProvider::open(
                    path,
                    config.model,
                    "fixture".to_string(),
                )?))
            }
            "openai-compatible" | "openai" => {
                #[cfg(feature = "llm-http")]
                {
                    return Ok(Self::OpenAiCompatible(OpenAiCompatibleProvider::new(
                        config,
                    )?));
                }
                #[cfg(not(feature = "llm-http"))]
                {
                    Err(LlmError::new(
                        "openai-compatible provider requires building with --features llm-http",
                    ))
                }
            }
            other => Err(LlmError::new(format!("unsupported LLM provider: {other}"))),
        }
    }
}

impl LlmProvider for ConfiguredLlmProvider {
    fn complete(&self, request: &LlmCompletionRequest) -> Result<String, LlmError> {
        match self {
            Self::Fixture(provider) => provider.complete(request),
            #[cfg(feature = "llm-http")]
            Self::OpenAiCompatible(provider) => provider.complete(request),
        }
    }

    fn metadata(&self) -> LlmProviderMetadata {
        match self {
            Self::Fixture(provider) => provider.metadata(),
            #[cfg(feature = "llm-http")]
            Self::OpenAiCompatible(provider) => provider.metadata(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct FixtureLlmProvider {
    responses: BTreeMap<String, String>,
    model: String,
    provider_name: String,
    path: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct FixtureRecord {
    request_hash: String,
    response: String,
}

impl FixtureLlmProvider {
    pub fn open(
        path: impl Into<PathBuf>,
        model: String,
        provider_name: String,
    ) -> Result<Self, LlmError> {
        let path = path.into();
        let responses = read_fixture_records(&path)?;
        Ok(Self {
            responses,
            model,
            provider_name,
            path,
        })
    }
}

impl LlmProvider for FixtureLlmProvider {
    fn complete(&self, request: &LlmCompletionRequest) -> Result<String, LlmError> {
        let hash = request_hash(request)?;
        self.responses.get(&hash).cloned().ok_or_else(|| {
            LlmError::new(format!(
                "fixture response missing for request_hash={hash} in {}",
                self.path.display()
            ))
        })
    }

    fn metadata(&self) -> LlmProviderMetadata {
        LlmProviderMetadata {
            provider: self.provider_name.clone(),
            model: self.model.clone(),
            base_url: Some(self.path.display().to_string()),
            prompt_version: None,
        }
    }
}

#[cfg(feature = "llm-http")]
#[derive(Clone, Debug)]
pub struct OpenAiCompatibleProvider {
    model: String,
    base_url: String,
    api_key: String,
    timeout: Duration,
    cache_path: Option<PathBuf>,
    cache: Arc<Mutex<BTreeMap<String, String>>>,
}

#[cfg(feature = "llm-http")]
impl OpenAiCompatibleProvider {
    pub fn new(config: LlmProviderConfig) -> Result<Self, LlmError> {
        let base_url = config
            .base_url
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string())
            .trim_end_matches('/')
            .to_string();
        let api_key = config
            .api_key
            .ok_or_else(|| LlmError::new("AGENT_MEMORY_LLM_API_KEY is required"))?;
        Ok(Self {
            model: config.model,
            base_url,
            api_key,
            timeout: Duration::from_secs(config.timeout_secs),
            cache: Arc::new(Mutex::new(read_optional_cache(
                config.cache_path.as_deref(),
            )?)),
            cache_path: config.cache_path,
        })
    }
}

#[cfg(feature = "llm-http")]
impl LlmProvider for OpenAiCompatibleProvider {
    fn complete(&self, request: &LlmCompletionRequest) -> Result<String, LlmError> {
        let hash = request_hash(request)?;
        if let Some(response) = self
            .cache
            .lock()
            .map_err(|_| LlmError::new("LLM cache lock poisoned"))?
            .get(&hash)
            .cloned()
        {
            return Ok(response);
        }

        let agent = ureq::AgentBuilder::new().timeout(self.timeout).build();
        let url = format!("{}/chat/completions", self.base_url);
        let response = agent
            .post(&url)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_string(&serde_json::to_string(request).map_err(json_error)?)
            .map_err(http_error)?;
        let value: serde_json::Value = response
            .into_json()
            .map_err(|error| LlmError::new(format!("LLM response JSON error: {error}")))?;
        let content = extract_chat_message_content(&value)?;

        if let Some(path) = &self.cache_path {
            append_cache_record(path, &hash, &content)?;
            self.cache
                .lock()
                .map_err(|_| LlmError::new("LLM cache lock poisoned"))?
                .insert(hash, content.clone());
        }

        Ok(content)
    }

    fn metadata(&self) -> LlmProviderMetadata {
        LlmProviderMetadata {
            provider: "openai-compatible".to_string(),
            model: self.model.clone(),
            base_url: Some(self.base_url.clone()),
            prompt_version: None,
        }
    }
}

#[cfg(feature = "llm-http")]
fn extract_chat_message_content(value: &serde_json::Value) -> Result<String, LlmError> {
    value
        .pointer("/choices/0/message/content")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|content| !content.is_empty())
        .or_else(|| {
            value
                .pointer("/choices/0/message/reasoning_content")
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|content| !content.is_empty())
        })
        .map(ToOwned::to_owned)
        .ok_or_else(|| {
            LlmError::new("LLM response missing choices[0].message.content or reasoning_content")
        })
}

#[cfg(feature = "llm-http")]
fn read_optional_cache(path: Option<&Path>) -> Result<BTreeMap<String, String>, LlmError> {
    match path {
        Some(path) if path.exists() => read_fixture_records(path),
        _ => Ok(BTreeMap::new()),
    }
}

#[cfg(feature = "llm-http")]
fn append_cache_record(path: &Path, request_hash: &str, response: &str) -> Result<(), LlmError> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|error| LlmError::new(format!("failed to create LLM cache dir: {error}")))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| LlmError::new(format!("failed to open LLM cache: {error}")))?;
    let record = FixtureRecord {
        request_hash: request_hash.to_string(),
        response: response.to_string(),
    };
    let line = serde_json::to_string(&record).map_err(json_error)?;
    writeln!(file, "{line}")
        .map_err(|error| LlmError::new(format!("failed to write LLM cache: {error}")))
}

pub fn request_hash(request: &LlmCompletionRequest) -> Result<String, LlmError> {
    let serialized = serde_json::to_string(request).map_err(json_error)?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    serialized.hash(&mut hasher);
    Ok(format!("{:016x}", hasher.finish()))
}

fn read_fixture_records(path: &Path) -> Result<BTreeMap<String, String>, LlmError> {
    let content = fs::read_to_string(path)
        .map_err(|error| LlmError::new(format!("failed to read fixture: {error}")))?;
    let mut records = BTreeMap::new();
    for (index, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: FixtureRecord = serde_json::from_str(line).map_err(|error| {
            LlmError::new(format!("invalid fixture JSONL line {}: {error}", index + 1))
        })?;
        records.insert(record.request_hash, record.response);
    }
    Ok(records)
}

fn json_error(error: serde_json::Error) -> LlmError {
    LlmError::new(format!("LLM JSON error: {error}"))
}

#[cfg(feature = "llm-http")]
fn http_error(error: ureq::Error) -> LlmError {
    match error {
        ureq::Error::Status(code, response) => {
            let body = response.into_string().unwrap_or_default();
            LlmError::new(format!("LLM HTTP status {code}: {body}"))
        }
        ureq::Error::Transport(error) => LlmError::new(format!("LLM transport error: {error}")),
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "llm-http")]
    use super::extract_chat_message_content;

    #[cfg(feature = "llm-http")]
    #[test]
    fn chat_message_content_falls_back_to_reasoning_content() {
        let value = serde_json::json!({
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning_content": "  {\"answer\":\"Paris\"}  "
                }
            }]
        });

        let content = extract_chat_message_content(&value).unwrap();

        assert_eq!(content, "{\"answer\":\"Paris\"}");
    }
}
