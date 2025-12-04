// Semantic text chunking for RAG
// Splits documents into overlapping chunks suitable for embedding

use std::path::Path;

/// Configuration for text chunking
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Target size for each chunk in characters
    pub chunk_size: usize,
    /// Overlap between consecutive chunks
    pub chunk_overlap: usize,
    /// Separators to use for splitting (in order of priority)
    pub separators: Vec<String>,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 200,
            separators: vec![
                "\n\n".to_string(),  // Paragraphs
                "\n".to_string(),    // Lines
                ". ".to_string(),    // Sentences
                ", ".to_string(),    // Clauses
                " ".to_string(),     // Words
            ],
        }
    }
}

/// A chunk of text with metadata
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// The chunk content
    pub content: String,
    /// Start character index in original document
    pub start_idx: usize,
    /// End character index in original document
    pub end_idx: usize,
    /// Chunk index in sequence
    pub chunk_idx: usize,
    /// Source document path or identifier
    pub source: String,
}

impl TextChunk {
    /// Create a display summary
    pub fn summary(&self, max_len: usize) -> String {
        if self.content.len() <= max_len {
            self.content.clone()
        } else {
            format!("{}...", &self.content[..max_len])
        }
    }
}

/// Recursive character text splitter (similar to LangChain's approach)
pub struct TextChunker {
    config: ChunkConfig,
}

impl TextChunker {
    pub fn new() -> Self {
        Self {
            config: ChunkConfig::default(),
        }
    }

    pub fn with_config(config: ChunkConfig) -> Self {
        Self { config }
    }

    /// Split text into chunks
    pub fn chunk_text(&self, text: &str, source: &str) -> Vec<TextChunk> {
        let mut chunks = Vec::new();
        let splits = self.split_recursive(text, &self.config.separators);
        
        let mut current_chunk = String::new();
        let mut current_start = 0;
        let mut chunk_idx = 0;

        for split in splits {
            // If adding this split would exceed chunk size
            if !current_chunk.is_empty() 
                && current_chunk.len() + split.len() > self.config.chunk_size 
            {
                // Save current chunk
                let end_idx = current_start + current_chunk.len();
                chunks.push(TextChunk {
                    content: current_chunk.trim().to_string(),
                    start_idx: current_start,
                    end_idx,
                    chunk_idx,
                    source: source.to_string(),
                });
                chunk_idx += 1;

                // Start new chunk with overlap
                let overlap_start = if current_chunk.len() > self.config.chunk_overlap {
                    current_chunk.len() - self.config.chunk_overlap
                } else {
                    0
                };
                current_chunk = current_chunk[overlap_start..].to_string();
                current_start = end_idx - (current_chunk.len());
            }

            current_chunk.push_str(&split);
        }

        // Don't forget the last chunk
        if !current_chunk.trim().is_empty() {
            chunks.push(TextChunk {
                content: current_chunk.trim().to_string(),
                start_idx: current_start,
                end_idx: current_start + current_chunk.len(),
                chunk_idx,
                source: source.to_string(),
            });
        }

        chunks
    }

    /// Recursively split text using separators
    fn split_recursive(&self, text: &str, separators: &[String]) -> Vec<String> {
        if separators.is_empty() {
            return vec![text.to_string()];
        }

        let separator = &separators[0];
        let remaining_separators = &separators[1..];

        let mut result = Vec::new();
        let parts: Vec<&str> = text.split(separator.as_str()).collect();

        for (i, part) in parts.iter().enumerate() {
            if part.is_empty() {
                continue;
            }

            // If this part is small enough, keep it
            if part.len() <= self.config.chunk_size {
                // Add the separator back (except for last part)
                if i < parts.len() - 1 {
                    result.push(format!("{}{}", part, separator));
                } else {
                    result.push(part.to_string());
                }
            } else {
                // Recursively split with remaining separators
                let sub_parts = self.split_recursive(part, remaining_separators);
                result.extend(sub_parts);
            }
        }

        result
    }

    /// Load and chunk a file
    pub fn chunk_file(&self, path: &Path) -> anyhow::Result<Vec<TextChunk>> {
        let content = std::fs::read_to_string(path)?;
        let source = path.to_string_lossy().to_string();
        Ok(self.chunk_text(&content, &source))
    }

    /// Chunk multiple files from a directory
    pub fn chunk_directory(&self, dir: &Path, extensions: &[&str]) -> anyhow::Result<Vec<TextChunk>> {
        let mut all_chunks = Vec::new();

        if !dir.is_dir() {
            return Err(anyhow::anyhow!("Not a directory: {:?}", dir));
        }

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                let ext = path.extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");

                if extensions.is_empty() || extensions.contains(&ext) {
                    match self.chunk_file(&path) {
                        Ok(chunks) => all_chunks.extend(chunks),
                        Err(e) => {
                            eprintln!("Warning: Could not process {:?}: {}", path, e);
                        }
                    }
                }
            }
        }

        Ok(all_chunks)
    }
}

impl Default for TextChunker {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple BM25-style keyword scorer for hybrid search
pub struct KeywordScorer {
    /// IDF values for terms (would be computed from corpus in production)
    k1: f32,
    b: f32,
}

impl KeywordScorer {
    pub fn new() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }

    /// Score a document against a query using simple term frequency
    pub fn score(&self, query: &str, document: &str) -> f32 {
        let query_lower = query.to_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();
        
        let doc_lower = document.to_lowercase();
        let doc_len = document.split_whitespace().count() as f32;
        let avg_doc_len = 200.0; // Approximate average

        let mut score = 0.0;
        for term in &query_terms {
            let tf = doc_lower.matches(term).count() as f32;
            if tf > 0.0 {
                // Simplified BM25 (without IDF since we don't have corpus stats)
                let numerator = tf * (self.k1 + 1.0);
                let denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_doc_len));
                score += numerator / denominator;
            }
        }

        score
    }
}

impl Default for KeywordScorer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_chunking() {
        let chunker = TextChunker::with_config(ChunkConfig {
            chunk_size: 100,
            chunk_overlap: 20,
            ..Default::default()
        });

        let text = "This is the first paragraph.\n\nThis is the second paragraph.\n\nThis is the third paragraph.";
        let chunks = chunker.chunk_text(text, "test.txt");

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(!chunk.content.is_empty());
            assert_eq!(chunk.source, "test.txt");
        }
    }

    #[test]
    fn test_keyword_scorer() {
        let scorer = KeywordScorer::new();
        
        let doc = "The quick brown fox jumps over the lazy dog";
        let query = "quick fox";
        
        let score = scorer.score(query, doc);
        assert!(score > 0.0);
        
        let irrelevant_score = scorer.score("cat bird", doc);
        assert!(irrelevant_score < score);
    }
}
