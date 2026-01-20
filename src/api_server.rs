//! # OpenAI-Compatible API Server
//!
//! This module provides an HTTP server that exposes the Torus LLM through an
//! OpenAI-compatible API, allowing it to be used as a drop-in replacement for
//! OpenAI's API in applications like OpenCode.
//!
//! ## Endpoints
//!
//! - `POST /v1/chat/completions` - Chat completions (main endpoint)
//! - `POST /v1/completions` - Text completions
//! - `GET /v1/models` - List available models
//! - `GET /health` - Health check
//!
//! ## Usage with OpenCode
//!
//! Configure OpenCode to use this server:
//!
//! ```toml
//! # opencode.toml
//! [provider.torus]
//! type = "openai"
//! api_key = "not-needed"
//! base_url = "http://localhost:8080/v1"
//!
//! [model.torus-small]
//! provider = "torus"
//! model = "torus-small"
//! ```

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::llm::{SamplingStrategy, TextGenerator, TorusLLM, TorusLLMConfig};
use crate::tokenizer::{format_chat_prompt, ChatMessage, SimpleTokenizer};
use crate::TorusResult;

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Host to bind to
    pub host: String,
    /// Port to listen on
    pub port: u16,
    /// Model configuration
    pub model_config: TorusLLMConfig,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Default temperature
    pub default_temperature: f64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            model_config: TorusLLMConfig::tiny(),
            max_tokens: 2048,
            default_temperature: 0.7,
        }
    }
}

// ============================================================================
// OpenAI API Types
// ============================================================================

/// Chat completion request (OpenAI-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessageInput>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
}

/// Chat message input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessageInput {
    pub role: String,
    pub content: String,
}

impl From<ChatMessageInput> for ChatMessage {
    fn from(msg: ChatMessageInput) -> Self {
        ChatMessage {
            role: msg.role,
            content: msg.content,
        }
    }
}

/// Chat completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

/// Chat choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessageOutput,
    pub finish_reason: String,
}

/// Chat message output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessageOutput {
    pub role: String,
    pub content: String,
}

/// Token usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Text completion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

/// Text completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<TextChoice>,
    pub usage: Usage,
}

/// Text choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: String,
}

/// Model info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

/// Models list response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// Error response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub code: Option<String>,
}

// ============================================================================
// API Handler
// ============================================================================

/// The API handler that processes requests
pub struct ApiHandler {
    model: Arc<TorusLLM>,
    tokenizer: Arc<SimpleTokenizer>,
    config: ServerConfig,
}

impl ApiHandler {
    /// Create a new API handler
    pub fn new(model: TorusLLM, tokenizer: SimpleTokenizer, config: ServerConfig) -> Self {
        Self {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            config,
        }
    }

    /// Handle chat completion request
    pub fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> TorusResult<ChatCompletionResponse> {
        // Convert messages
        let messages: Vec<ChatMessage> = request.messages.into_iter().map(Into::into).collect();

        // Format as prompt
        let prompt = format_chat_prompt(&messages);

        // Tokenize
        let prompt_tokens = self.tokenizer.encode(&prompt);
        let prompt_len = prompt_tokens.len();

        // Set up sampling
        let temperature = request
            .temperature
            .unwrap_or(self.config.default_temperature);
        let top_p = request.top_p.unwrap_or(0.95);
        let max_tokens = request.max_tokens.unwrap_or(self.config.max_tokens);

        let strategy = SamplingStrategy::Combined {
            top_k: 50,
            top_p,
            temperature,
        };

        // Generate
        let generator = TextGenerator::new(
            // Clone the model - in production you'd use Arc properly
            (*self.model).clone_model()?,
            strategy,
        );

        let generated_ids = generator.generate(
            &prompt_tokens,
            max_tokens,
            Some(self.tokenizer.special_tokens().eos_id),
        )?;

        // Decode response
        let response_text = self.tokenizer.decode_skip_special(&generated_ids);
        let completion_len = generated_ids.len();

        // Build response
        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", generate_id()),
            object: "chat.completion".to_string(),
            created: current_timestamp(),
            model: request.model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessageOutput {
                    role: "assistant".to_string(),
                    content: response_text,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: prompt_len,
                completion_tokens: completion_len,
                total_tokens: prompt_len + completion_len,
            },
        };

        Ok(response)
    }

    /// Handle text completion request
    pub fn completion(&self, request: CompletionRequest) -> TorusResult<CompletionResponse> {
        // Tokenize
        let prompt_tokens = self.tokenizer.encode(&request.prompt);
        let prompt_len = prompt_tokens.len();

        // Set up sampling
        let temperature = request
            .temperature
            .unwrap_or(self.config.default_temperature);
        let top_p = request.top_p.unwrap_or(0.95);
        let max_tokens = request.max_tokens.unwrap_or(self.config.max_tokens);

        let strategy = SamplingStrategy::Combined {
            top_k: 50,
            top_p,
            temperature,
        };

        // Generate
        let generator = TextGenerator::new((*self.model).clone_model()?, strategy);

        let generated_ids = generator.generate(
            &prompt_tokens,
            max_tokens,
            Some(self.tokenizer.special_tokens().eos_id),
        )?;

        // Decode response
        let response_text = self.tokenizer.decode_skip_special(&generated_ids);
        let completion_len = generated_ids.len();

        // Build response
        let response = CompletionResponse {
            id: format!("cmpl-{}", generate_id()),
            object: "text_completion".to_string(),
            created: current_timestamp(),
            model: request.model,
            choices: vec![TextChoice {
                index: 0,
                text: response_text,
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: prompt_len,
                completion_tokens: completion_len,
                total_tokens: prompt_len + completion_len,
            },
        };

        Ok(response)
    }

    /// List available models
    pub fn list_models(&self) -> ModelsResponse {
        ModelsResponse {
            object: "list".to_string(),
            data: vec![
                ModelInfo {
                    id: "torus-tiny".to_string(),
                    object: "model".to_string(),
                    created: 1700000000,
                    owned_by: "torus-attention".to_string(),
                },
                ModelInfo {
                    id: "torus-small".to_string(),
                    object: "model".to_string(),
                    created: 1700000000,
                    owned_by: "torus-attention".to_string(),
                },
                ModelInfo {
                    id: "torus-medium".to_string(),
                    object: "model".to_string(),
                    created: 1700000000,
                    owned_by: "torus-attention".to_string(),
                },
            ],
        }
    }
}

// Helper trait to clone the model (needed since TorusLLM doesn't impl Clone)
trait CloneModel {
    fn clone_model(&self) -> TorusResult<TorusLLM>;
}

impl CloneModel for TorusLLM {
    fn clone_model(&self) -> TorusResult<TorusLLM> {
        // For now, create a new random model with the same config
        // In production, you'd properly share weights
        let (model, _) = TorusLLM::new_random(self.config().clone(), self.device())?;
        Ok(model)
    }
}

/// Generate a random ID
fn generate_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    format!("{:x}", nanos)
}

/// Get current Unix timestamp
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

// ============================================================================
// Server Runner (using Tokio + minimal HTTP)
// ============================================================================

/// Simple HTTP server for the API
///
/// Note: This is a minimal implementation. For production, use axum, actix-web, or warp.
pub mod server {
    use super::*;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::TcpListener;

    /// Run the API server
    pub async fn run_server(handler: Arc<ApiHandler>, config: &ServerConfig) -> TorusResult<()> {
        let addr = format!("{}:{}", config.host, config.port);
        let listener = TcpListener::bind(&addr)
            .await
            .map_err(|e| crate::error::TorusError::Io(e.to_string()))?;

        println!("🌀 Torus LLM API Server running at http://{}", addr);
        println!("   Endpoints:");
        println!("   - POST /v1/chat/completions");
        println!("   - POST /v1/completions");
        println!("   - GET  /v1/models");
        println!("   - GET  /health");

        loop {
            let (stream, _) = listener
                .accept()
                .await
                .map_err(|e| crate::error::TorusError::Io(e.to_string()))?;

            let handler = Arc::clone(&handler);

            tokio::spawn(async move {
                if let Err(e) = handle_connection(stream, handler).await {
                    eprintln!("Connection error: {:?}", e);
                }
            });
        }
    }

    async fn handle_connection(
        mut stream: tokio::net::TcpStream,
        handler: Arc<ApiHandler>,
    ) -> TorusResult<()> {
        let (reader, mut writer) = stream.split();
        let mut reader = BufReader::new(reader);

        // Read request line
        let mut request_line = String::new();
        reader
            .read_line(&mut request_line)
            .await
            .map_err(|e| crate::error::TorusError::Io(e.to_string()))?;

        // Parse method and path
        let parts: Vec<&str> = request_line.split_whitespace().collect();
        if parts.len() < 2 {
            return send_error(&mut writer, 400, "Bad Request").await;
        }

        let method = parts[0];
        let path = parts[1];

        // Read headers
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            reader
                .read_line(&mut line)
                .await
                .map_err(|e| crate::error::TorusError::Io(e.to_string()))?;

            if line == "\r\n" || line == "\n" {
                break;
            }

            let lower = line.to_lowercase();
            if lower.starts_with("content-length:") {
                if let Some(len_str) = lower.strip_prefix("content-length:") {
                    content_length = len_str.trim().parse().unwrap_or(0);
                }
            }
        }

        // Read body if present
        let mut body = vec![0u8; content_length];
        if content_length > 0 {
            tokio::io::AsyncReadExt::read_exact(&mut reader, &mut body)
                .await
                .map_err(|e| crate::error::TorusError::Io(e.to_string()))?;
        }
        let body_str = String::from_utf8_lossy(&body);

        // Route request
        match (method, path) {
            ("GET", "/health") => send_json(&mut writer, 200, r#"{"status":"ok"}"#).await,
            ("GET", "/v1/models") => {
                let response = handler.list_models();
                let json = serde_json::to_string(&response).unwrap();
                send_json(&mut writer, 200, &json).await
            }
            ("POST", "/v1/chat/completions") => {
                match serde_json::from_str::<ChatCompletionRequest>(&body_str) {
                    Ok(request) => match handler.chat_completion(request) {
                        Ok(response) => {
                            let json = serde_json::to_string(&response).unwrap();
                            send_json(&mut writer, 200, &json).await
                        }
                        Err(e) => {
                            let error = ErrorResponse {
                                error: ErrorDetail {
                                    message: e.to_string(),
                                    r#type: "internal_error".to_string(),
                                    code: None,
                                },
                            };
                            let json = serde_json::to_string(&error).unwrap();
                            send_json(&mut writer, 500, &json).await
                        }
                    },
                    Err(e) => {
                        let error = ErrorResponse {
                            error: ErrorDetail {
                                message: format!("Invalid request: {}", e),
                                r#type: "invalid_request_error".to_string(),
                                code: None,
                            },
                        };
                        let json = serde_json::to_string(&error).unwrap();
                        send_json(&mut writer, 400, &json).await
                    }
                }
            }
            ("POST", "/v1/completions") => {
                match serde_json::from_str::<CompletionRequest>(&body_str) {
                    Ok(request) => match handler.completion(request) {
                        Ok(response) => {
                            let json = serde_json::to_string(&response).unwrap();
                            send_json(&mut writer, 200, &json).await
                        }
                        Err(e) => {
                            let error = ErrorResponse {
                                error: ErrorDetail {
                                    message: e.to_string(),
                                    r#type: "internal_error".to_string(),
                                    code: None,
                                },
                            };
                            let json = serde_json::to_string(&error).unwrap();
                            send_json(&mut writer, 500, &json).await
                        }
                    },
                    Err(e) => {
                        let error = ErrorResponse {
                            error: ErrorDetail {
                                message: format!("Invalid request: {}", e),
                                r#type: "invalid_request_error".to_string(),
                                code: None,
                            },
                        };
                        let json = serde_json::to_string(&error).unwrap();
                        send_json(&mut writer, 400, &json).await
                    }
                }
            }
            _ => send_error(&mut writer, 404, "Not Found").await,
        }
    }

    async fn send_json(
        writer: &mut tokio::net::tcp::WriteHalf<'_>,
        status: u16,
        body: &str,
    ) -> TorusResult<()> {
        let status_text = match status {
            200 => "OK",
            400 => "Bad Request",
            404 => "Not Found",
            500 => "Internal Server Error",
            _ => "Unknown",
        };

        let response = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
            status, status_text, body.len(), body
        );

        writer
            .write_all(response.as_bytes())
            .await
            .map_err(|e| crate::error::TorusError::Io(e.to_string()))?;
        writer
            .flush()
            .await
            .map_err(|e| crate::error::TorusError::Io(e.to_string()))?;

        Ok(())
    }

    async fn send_error(
        writer: &mut tokio::net::tcp::WriteHalf<'_>,
        status: u16,
        message: &str,
    ) -> TorusResult<()> {
        let error = ErrorResponse {
            error: ErrorDetail {
                message: message.to_string(),
                r#type: "error".to_string(),
                code: None,
            },
        };
        let json = serde_json::to_string(&error).unwrap();
        send_json(writer, status, &json).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_completion_request_parse() {
        let json = r#"{
            "model": "torus-small",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "temperature": 0.7
        }"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.model, "torus-small");
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.temperature, Some(0.7));
    }

    #[test]
    fn test_models_response() {
        let config = ServerConfig::default();
        let tokenizer = SimpleTokenizer::new_basic(config.model_config.vocab_size);
        let device = candle_core::Device::Cpu;
        let (model, _) = TorusLLM::new_random(config.model_config.clone(), &device).unwrap();
        let handler = ApiHandler::new(model, tokenizer, config);

        let response = handler.list_models();
        assert_eq!(response.object, "list");
        assert!(!response.data.is_empty());
    }
}
