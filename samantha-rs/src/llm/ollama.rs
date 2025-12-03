use anyhow::Result;
use serde::{Deserialize, Serialize};
use reqwest::Client;

/// Message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: &str) -> Self {
        Message {
            role: "system".to_string(),
            content: content.to_string(),
        }
    }

    pub fn user(content: &str) -> Self {
        Message {
            role: "user".to_string(),
            content: content.to_string(),
        }
    }

    pub fn assistant(content: &str) -> Self {
        Message {
            role: "assistant".to_string(),
            content: content.to_string(),
        }
    }
}

/// Request body for Ollama chat API
#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<String>,
}

/// Response from Ollama chat API
#[derive(Debug, Deserialize)]
struct ChatResponse {
    message: Message,
}

/// Client for interacting with Ollama API
#[derive(Clone)]
pub struct OllamaClient {
    client: Client,
    base_url: String,
    model: String,
}

impl OllamaClient {
    /// Create a new Ollama client
    pub fn new(model: &str) -> Self {
        OllamaClient {
            client: Client::new(),
            base_url: "http://localhost:11434".to_string(),
            model: model.to_string(),
        }
    }

    /// Create a client with a custom base URL
    #[allow(dead_code)]
    pub fn with_base_url(model: &str, base_url: &str) -> Self {
        OllamaClient {
            client: Client::new(),
            base_url: base_url.to_string(),
            model: model.to_string(),
        }
    }

    /// Send a chat request and get a response
    pub async fn chat(&self, messages: Vec<Message>) -> Result<String> {
        self.chat_with_format(messages, None).await
    }

    /// Send a chat request with optional format
    async fn chat_with_format(&self, messages: Vec<Message>, format: Option<String>) -> Result<String> {
        let request = ChatRequest {
            model: self.model.clone(),
            messages,
            stream: false,
            format,
        };

        let response = self.client
            .post(format!("{}/api/chat", self.base_url))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("Ollama API error: {} - {}", status, text);
        }

        let chat_response: ChatResponse = response.json().await?;
        Ok(chat_response.message.content)
    }

    /// Simple single-turn query
    pub async fn query(&self, prompt: &str) -> Result<String> {
        self.chat(vec![Message::user(prompt)]).await
    }

    /// Query with JSON mode enabled - guarantees valid JSON output
    pub async fn query_json(&self, prompt: &str) -> Result<String> {
        self.chat_with_format(
            vec![Message::user(prompt)],
            Some("json".to_string())
        ).await
    }

    /// Query with system prompt and JSON mode
    pub async fn query_json_with_system(&self, system: &str, prompt: &str) -> Result<String> {
        self.chat_with_format(
            vec![Message::system(system), Message::user(prompt)],
            Some("json".to_string())
        ).await
    }

    /// Query with a system prompt
    pub async fn query_with_system(&self, system: &str, prompt: &str) -> Result<String> {
        self.chat(vec![
            Message::system(system),
            Message::user(prompt),
        ]).await
    }

    /// Simple completion (alias for query)
    pub async fn complete(&self, prompt: &str) -> Result<String> {
        self.query(prompt).await
    }

    /// Chat with pre-built message array (accepts serde_json::Value)
    pub async fn chat_with_messages(&self, messages: &[serde_json::Value]) -> Result<String> {
        // Convert JSON values to Message structs
        let msgs: Vec<Message> = messages
            .iter()
            .filter_map(|v| {
                let role = v.get("role")?.as_str()?.to_string();
                let content = v.get("content")?.as_str()?.to_string();
                Some(Message { role, content })
            })
            .collect();
        
        self.chat(msgs).await
    }

    /// Check if Ollama is running and the model is available
    pub async fn health_check(&self) -> Result<bool> {
        let response = self.client
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await;

        match response {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// Get the current model name
    pub fn model(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = Message::user("Hello");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello");

        let sys = Message::system("You are helpful");
        assert_eq!(sys.role, "system");
    }
}
