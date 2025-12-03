// Embeddings client using Ollama
// Generates vector embeddings for text chunks and queries

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

const OLLAMA_EMBED_URL: &str = "http://localhost:11434/api/embeddings";
const DEFAULT_MODEL: &str = "nomic-embed-text";

/// Ollama embeddings request
#[derive(Debug, Serialize)]
struct EmbedRequest {
    model: String,
    prompt: String,
}

/// Ollama embeddings response
#[derive(Debug, Deserialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
}

/// Embeddings client for generating vector representations
pub struct EmbeddingsClient {
    model: String,
    http: reqwest::Client,
}

impl EmbeddingsClient {
    pub fn new() -> Self {
        Self {
            model: DEFAULT_MODEL.to_string(),
            http: reqwest::Client::new(),
        }
    }

    pub fn with_model(model: &str) -> Self {
        Self {
            model: model.to_string(),
            http: reqwest::Client::new(),
        }
    }

    /// Generate embedding for a single text
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let request = EmbedRequest {
            model: self.model.clone(),
            prompt: text.to_string(),
        };

        let response = self.http
            .post(OLLAMA_EMBED_URL)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("Ollama embed error {}: {}", status, body));
        }

        let embed_response: EmbedResponse = response.json().await?;
        Ok(embed_response.embedding)
    }

    /// Generate embeddings for multiple texts (batched)
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());
        
        // Ollama doesn't support batch embeddings natively, so we do them sequentially
        // In production, could parallelize with tokio::spawn
        for text in texts {
            let embedding = self.embed(text).await?;
            embeddings.push(embedding);
        }
        
        Ok(embeddings)
    }

    /// Check if the embedding model is available
    pub async fn health_check(&self) -> Result<bool> {
        let test_embedding = self.embed("test").await;
        Ok(test_embedding.is_ok())
    }

    /// Get the dimensionality of embeddings (by generating a test embedding)
    pub async fn embedding_dim(&self) -> Result<usize> {
        let embedding = self.embed("test").await?;
        Ok(embedding.len())
    }
}

impl Default for EmbeddingsClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

/// Compute dot product between two vectors
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) + 1.0).abs() < 0.001);
    }
}
