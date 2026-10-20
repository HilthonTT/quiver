use std::time::Duration;

use serde::Deserialize;
use serde_json::json;

/// Settings for turning text into vectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingConfig {
    /// Full URL of the `POST /v1/embeddings` endpoint.
    pub url: String,
    /// Model name, sent as `model` if set.
    pub model: Option<String>,
    /// Bearer token, if the endpoint needs one.
    pub api_key: Option<String>,
}

/// Embeds text with an OpenAI-compatible endpoint.
#[derive(Debug, Clone)]
pub struct Embedder {
    http: reqwest::Client,
    config: EmbeddingConfig,
}

#[derive(Deserialize)]
struct Response {
    data: Vec<Item>,
}

#[derive(Deserialize)]
struct Item {
    embedding: Vec<f32>,
}

impl Embedder {
    /// Creates an embedder for `config`.
    pub fn new(config: EmbeddingConfig) -> Result<Self, String> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| e.to_string())?;
        Ok(Self { http, config })
    }

    /// Returns the embedding of `text`.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let mut body = json!({ "input": text });
        if let Some(model) = &self.config.model {
            body["model"] = json!(model);
        }
        let mut req = self.http.post(&self.config.url).json(&body);
        if let Some(key) = &self.config.api_key {
            req = req.bearer_auth(key);
        }
        let res = req
            .send()
            .await
            .map_err(|e| format!("embedding request failed: {e}"))?;
        let status = res.status();
        if !status.is_success() {
            let text = res.text().await.unwrap_or_default();
            return Err(format!("embedding endpoint returned {status}: {text}"));
        }
        let parsed: Response = res
            .json()
            .await
            .map_err(|e| format!("unexpected embedding response: {e}"))?;
        parsed
            .data
            .into_iter()
            .next()
            .map(|i| i.embedding)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| "embedding response contained no vector".into())
    }
}
