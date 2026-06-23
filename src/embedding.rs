use std::collections::BTreeSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::LazyLock;

use rust_stemmers::{Algorithm, Stemmer};

static STEMMER: LazyLock<Stemmer> = LazyLock::new(|| Stemmer::create(Algorithm::English));

/// Trait for pluggable embedding providers (Ollama, OpenAI, local models, etc).
pub trait EmbeddingProvider: Send + Sync {
    fn embed(&self, text: &str) -> Vec<f32>;
    fn dimensions(&self) -> usize;
}

/// Ollama-hosted embedding model accessed via HTTP.
#[cfg(feature = "embed-ollama")]
pub struct OllamaEmbeddingProvider {
    base_url: String,
    model: String,
    dimensions: usize,
}

#[cfg(feature = "embed-ollama")]
impl OllamaEmbeddingProvider {
    pub fn from_env() -> Self {
        let base_url =
            std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
        let model =
            std::env::var("OLLAMA_EMBED_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string());
        Self {
            base_url,
            model,
            dimensions: 768,
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = dimensions;
        self
    }
}

#[cfg(feature = "embed-ollama")]
impl EmbeddingProvider for OllamaEmbeddingProvider {
    fn embed(&self, text: &str) -> Vec<f32> {
        let body = serde_json::json!({
            "model": self.model,
            "prompt": text,
        });
        match ureq::post(&format!("{}/api/embeddings", self.base_url))
            .set("Content-Type", "application/json")
            .send_json(&body)
        {
            Ok(response) => {
                let json: serde_json::Value = response.into_json().unwrap_or_default();
                json["embedding"]
                    .as_array()
                    .map(|values| {
                        values
                            .iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect()
                    })
                    .unwrap_or_default()
            }
            Err(_) => {
                // Fall back to hash embedding on error
                HashEmbedding::new(self.dimensions).embed(text)
            }
        }
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

/// Tokenizes text into lowercase stemmed tokens, stripping possessives.
///
/// Splits on non-alphanumeric boundaries (preserving apostrophes for
/// contractions/possessives), lowercases, strips trailing `'s` and `'`
/// suffixes, then applies Porter stemming.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_alphanumeric() && ch != '\'')
        .filter(|token| !token.is_empty())
        .map(|token| {
            let lowered = token.to_lowercase();
            // Strip possessive suffixes: "'s" then bare "'"
            // ("Caroline's" → "caroline", "dogs'" → "dogs")
            let stripped = lowered
                .strip_suffix("'s")
                .or_else(|| lowered.strip_suffix('\''))
                .map(|s| s.to_string())
                .unwrap_or(lowered);
            // Apply Porter stemming ("camping" → "camp", "researching" → "research")
            STEMMER.stem(&stripped).into_owned()
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct HashEmbedding {
    dimensions: usize,
}

impl HashEmbedding {
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions: dimensions.max(8),
        }
    }

    pub fn embed(&self, text: &str) -> Vec<f32> {
        let mut vector = vec![0.0; self.dimensions];
        for token in tokenize(text) {
            let mut hasher = DefaultHasher::new();
            token.hash(&mut hasher);
            let hash = hasher.finish();
            let bucket = hash as usize % self.dimensions;
            let sign = if hash & 1 == 0 { 1.0 } else { -1.0 };
            vector[bucket] += sign;
        }
        normalize(vector)
    }
}

pub fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

pub fn token_overlap_score(left: &str, right: &str) -> f32 {
    let left_tokens: BTreeSet<_> = tokenize(left).into_iter().collect();
    let right_tokens: BTreeSet<_> = tokenize(right).into_iter().collect();
    if left_tokens.is_empty() || right_tokens.is_empty() {
        return 0.0;
    }
    let intersection = left_tokens.intersection(&right_tokens).count() as f32;
    let union = left_tokens.union(&right_tokens).count() as f32;
    intersection / union
}

fn normalize(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}
