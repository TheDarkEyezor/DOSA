// Web search integration for DOSA
// Provides internet search capabilities similar to Martin's search feature

use anyhow::{Result, anyhow};
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Web search provider
pub struct WebSearch {
    client: Client,
    // Optional API keys for different services
    serp_api_key: Option<String>,
    openweather_api_key: Option<String>,
}

/// Search result from any provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub snippet: String,
    pub url: Option<String>,
    pub source: String,
}

/// Weather information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherInfo {
    pub location: String,
    pub temperature: f32,
    pub description: String,
    pub humidity: Option<u8>,
    pub wind_speed: Option<f32>,
}

/// DuckDuckGo Instant Answer response
#[derive(Debug, Deserialize)]
struct DuckDuckGoResponse {
    #[serde(rename = "Abstract")]
    abstract_text: String,
    #[serde(rename = "AbstractSource")]
    abstract_source: String,
    #[serde(rename = "AbstractURL")]
    abstract_url: String,
    #[serde(rename = "Heading")]
    heading: String,
    #[serde(rename = "RelatedTopics")]
    related_topics: Vec<DuckDuckGoTopic>,
    #[serde(rename = "Answer")]
    answer: String,
    #[serde(rename = "AnswerType")]
    answer_type: String,
}

#[derive(Debug, Deserialize)]
struct DuckDuckGoTopic {
    #[serde(rename = "Text")]
    text: Option<String>,
    #[serde(rename = "FirstURL")]
    first_url: Option<String>,
}

/// OpenWeatherMap response
#[derive(Debug, Deserialize)]
struct OpenWeatherResponse {
    name: String,
    main: OpenWeatherMain,
    weather: Vec<OpenWeatherCondition>,
    wind: Option<OpenWeatherWind>,
}

#[derive(Debug, Deserialize)]
struct OpenWeatherMain {
    temp: f32,
    humidity: u8,
}

#[derive(Debug, Deserialize)]
struct OpenWeatherCondition {
    description: String,
}

#[derive(Debug, Deserialize)]
struct OpenWeatherWind {
    speed: f32,
}

impl WebSearch {
    /// Create a new WebSearch instance
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            serp_api_key: std::env::var("SERP_API_KEY").ok(),
            openweather_api_key: std::env::var("OPENWEATHER_API_KEY").ok(),
        }
    }
    
    /// Create with specific API keys
    pub fn with_keys(serp_api_key: Option<String>, openweather_api_key: Option<String>) -> Self {
        Self {
            client: Client::new(),
            serp_api_key,
            openweather_api_key,
        }
    }
    
    /// Search using DuckDuckGo Instant Answers (free, no API key needed)
    /// Good for: definitions, facts, quick answers
    pub async fn search_duckduckgo(&self, query: &str) -> Result<Vec<SearchResult>> {
        let url = format!(
            "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1",
            urlencoding::encode(query)
        );
        
        let response: DuckDuckGoResponse = self.client
            .get(&url)
            .header("User-Agent", "DOSA/1.0")
            .send()
            .await?
            .json()
            .await?;
        
        let mut results = Vec::new();
        
        // Add the main abstract if available
        if !response.abstract_text.is_empty() {
            results.push(SearchResult {
                title: response.heading.clone(),
                snippet: response.abstract_text,
                url: Some(response.abstract_url),
                source: response.abstract_source,
            });
        }
        
        // Add direct answer if available
        if !response.answer.is_empty() {
            results.push(SearchResult {
                title: format!("{} ({})", response.heading, response.answer_type),
                snippet: response.answer,
                url: None,
                source: "DuckDuckGo".to_string(),
            });
        }
        
        // Add related topics
        for topic in response.related_topics.iter().take(3) {
            if let Some(text) = &topic.text {
                results.push(SearchResult {
                    title: text.chars().take(50).collect::<String>() + "...",
                    snippet: text.clone(),
                    url: topic.first_url.clone(),
                    source: "DuckDuckGo".to_string(),
                });
            }
        }
        
        Ok(results)
    }
    
    /// Get weather information
    pub async fn get_weather(&self, location: &str) -> Result<WeatherInfo> {
        if let Some(api_key) = &self.openweather_api_key {
            self.get_weather_openweather(location, api_key).await
        } else {
            // Fallback to wttr.in (free, no API key)
            self.get_weather_wttr(location).await
        }
    }
    
    /// Get weather from OpenWeatherMap
    async fn get_weather_openweather(&self, location: &str, api_key: &str) -> Result<WeatherInfo> {
        let url = format!(
            "https://api.openweathermap.org/data/2.5/weather?q={}&appid={}&units=metric",
            urlencoding::encode(location),
            api_key
        );
        
        let response: OpenWeatherResponse = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;
        
        Ok(WeatherInfo {
            location: response.name,
            temperature: response.main.temp,
            description: response.weather.first()
                .map(|w| w.description.clone())
                .unwrap_or_default(),
            humidity: Some(response.main.humidity),
            wind_speed: response.wind.map(|w| w.speed),
        })
    }
    
    /// Get weather from wttr.in (free, no API key needed)
    async fn get_weather_wttr(&self, location: &str) -> Result<WeatherInfo> {
        let url = format!("https://wttr.in/{}?format=j1", urlencoding::encode(location));
        
        let response: serde_json::Value = self.client
            .get(&url)
            .header("User-Agent", "DOSA/1.0")
            .send()
            .await?
            .json()
            .await?;
        
        let current = response.get("current_condition")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .ok_or_else(|| anyhow!("Invalid weather response"))?;
        
        let area = response.get("nearest_area")
            .and_then(|a| a.as_array())
            .and_then(|arr| arr.first())
            .and_then(|a| a.get("areaName"))
            .and_then(|n| n.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.get("value"))
            .and_then(|v| v.as_str())
            .unwrap_or(location);
        
        Ok(WeatherInfo {
            location: area.to_string(),
            temperature: current.get("temp_C")
                .and_then(|t| t.as_str())
                .and_then(|t| t.parse().ok())
                .unwrap_or(0.0),
            description: current.get("weatherDesc")
                .and_then(|d| d.as_array())
                .and_then(|arr| arr.first())
                .and_then(|v| v.get("value"))
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown")
                .to_string(),
            humidity: current.get("humidity")
                .and_then(|h| h.as_str())
                .and_then(|h| h.parse().ok()),
            wind_speed: current.get("windspeedKmph")
                .and_then(|w| w.as_str())
                .and_then(|w| w.parse().ok()),
        })
    }
    
    /// Quick search - tries to find a direct answer
    pub async fn quick_answer(&self, query: &str) -> Result<String> {
        // Check if it's a weather query
        let lower = query.to_lowercase();
        if lower.contains("weather") {
            // Extract location from query
            let location = extract_location_from_query(&lower)
                .unwrap_or_else(|| "London".to_string());
            
            let weather = self.get_weather(&location).await?;
            return Ok(format!(
                "🌤️ Weather in {}: {}°C, {}{}{}",
                weather.location,
                weather.temperature,
                weather.description,
                weather.humidity.map(|h| format!(", {}% humidity", h)).unwrap_or_default(),
                weather.wind_speed.map(|w| format!(", wind {} km/h", w)).unwrap_or_default(),
            ));
        }
        
        // Try DuckDuckGo for general queries
        let results = self.search_duckduckgo(query).await?;
        
        if let Some(first) = results.first() {
            if !first.snippet.is_empty() {
                return Ok(format!(
                    "🔍 {}\n\n{}\n\nSource: {}",
                    first.title,
                    first.snippet,
                    first.source
                ));
            }
        }
        
        Err(anyhow!("No results found for: {}", query))
    }
    
    /// Check if a query looks like a web search request
    pub fn is_web_query(input: &str) -> bool {
        let lower = input.to_lowercase();
        
        // Weather queries
        if lower.contains("weather") || lower.contains("temperature") || lower.contains("forecast") {
            return true;
        }
        
        // Stock/finance queries
        let stock_patterns = [
            "stock price",
            "stock",
            "share price",
            "market cap",
            "trading at",
            "worth",
            "earnings",
        ];
        if stock_patterns.iter().any(|p| lower.contains(p)) {
            // Make sure it's not a personal question like "who do I know at stock..."
            if !lower.starts_with("who do i know") {
                return true;
            }
        }
        
        // CEO/company leadership queries
        let company_patterns = [
            "ceo of",
            "who runs",
            "who founded",
            "who started",
            "who leads",
            "founder of",
            "president of",
        ];
        if company_patterns.iter().any(|p| lower.contains(p)) {
            return true;
        }
        
        // Search queries
        let search_patterns = [
            "search for",
            "look up",
            "google",
            "what is",
            "who is",
            "when was",
            "where is",
            "how to",
            "how do",
            "how can",
            "define",
            "news about",
            "latest on",
            "tell me about",
            "what does",
            "explain",
            "meaning of",
            "definition of",
        ];
        
        if search_patterns.iter().any(|p| lower.contains(p)) {
            // Filter out personal queries that should go to knowledge graph
            let personal_patterns = [
                "who do i know",
                "my contacts",
                "my friends",
                "my calendar",
                "my schedule",
                "my emails",
                "my tasks",
            ];
            if personal_patterns.iter().any(|p| lower.contains(p)) {
                return false;
            }
            return true;
        }
        
        // Factual questions
        let fact_patterns = [
            "population of",
            "capital of",
            "how tall",
            "how far",
            "how many",
            "how much does",
            "price of",
            "cost of",
            "distance to",
        ];
        if fact_patterns.iter().any(|p| lower.contains(p)) {
            return true;
        }
        
        // Product/review queries
        let product_patterns = [
            "best ",
            "top ",
            "review of",
            " vs ",
            " versus ",
            "compare ",
        ];
        if product_patterns.iter().any(|p| lower.contains(p)) {
            return true;
        }
        
        // News queries
        if lower.starts_with("news") || lower.contains("headlines") || lower.contains("current events") {
            return true;
        }
        
        // Recipe queries
        if lower.contains("recipe") || lower.starts_with("how to make") || lower.starts_with("how to cook") {
            return true;
        }
        
        false
    }
}

impl Default for WebSearch {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract location from a weather query
fn extract_location_from_query(query: &str) -> Option<String> {
    // Common patterns: "weather in London", "London weather", "what's the weather like in Paris"
    let patterns = [
        "weather in ",
        "weather for ",
        "weather at ",
        "temperature in ",
        "forecast for ",
        "like in ",
    ];
    
    for pattern in patterns {
        if let Some(idx) = query.find(pattern) {
            let start = idx + pattern.len();
            let location: String = query[start..]
                .chars()
                .take_while(|c| c.is_alphabetic() || *c == ' ')
                .collect();
            let location = location.trim();
            if !location.is_empty() {
                return Some(location.to_string());
            }
        }
    }
    
    // Try "X weather" pattern
    if query.ends_with(" weather") {
        let location = query.trim_end_matches(" weather").trim();
        if !location.is_empty() && !location.contains("the") {
            return Some(location.to_string());
        }
    }
    
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_is_web_query() {
        assert!(WebSearch::is_web_query("what's the weather like"));
        assert!(WebSearch::is_web_query("search for rust programming"));
        assert!(WebSearch::is_web_query("who is Elon Musk"));
        assert!(WebSearch::is_web_query("how to make pasta"));
        
        assert!(!WebSearch::is_web_query("schedule meeting with John"));
        assert!(!WebSearch::is_web_query("email Sarah about the project"));
    }
    
    #[test]
    fn test_extract_location() {
        assert_eq!(
            extract_location_from_query("weather in london"),
            Some("london".to_string())
        );
        assert_eq!(
            extract_location_from_query("what's the temperature in new york"),
            Some("new york".to_string())
        );
        assert_eq!(
            extract_location_from_query("paris weather"),
            Some("paris".to_string())
        );
    }
    
    #[tokio::test]
    async fn test_weather_wttr() {
        let search = WebSearch::new();
        let result = search.get_weather_wttr("London").await;
        // This might fail without network, so just check it doesn't panic
        println!("Weather result: {:?}", result);
    }
    
    #[tokio::test]
    async fn test_duckduckgo_search() {
        let search = WebSearch::new();
        let results = search.search_duckduckgo("Rust programming language").await;
        println!("DDG results: {:?}", results);
    }
}
