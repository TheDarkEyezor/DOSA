// RAG (Retrieval-Augmented Generation) Pipeline
// Ingests documents, creates embeddings, and retrieves relevant context for queries

use anyhow::Result;
use rusqlite::params;
use std::path::Path;
use std::sync::Arc;

use super::chunking::{TextChunker, ChunkConfig, KeywordScorer};
use super::embeddings::{EmbeddingsClient, cosine_similarity};
use crate::llm::OllamaClient;
use crate::storage::Database;

/// A document stored in the RAG system
#[derive(Debug, Clone)]
pub struct Document {
    pub id: i64,
    pub source: String,
    pub title: String,
    pub content: String,
    pub doc_type: String,
    pub created_at: String,
}

/// A chunk with its embedding stored in the database
#[derive(Debug, Clone)]
pub struct StoredChunk {
    pub id: i64,
    pub document_id: i64,
    pub content: String,
    pub chunk_idx: usize,
    pub embedding: Vec<f32>,
}

/// Search result with relevance score
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: StoredChunk,
    pub document: Document,
    pub semantic_score: f32,
    pub keyword_score: f32,
    pub combined_score: f32,
}

impl SearchResult {
    /// Format for display
    pub fn display(&self) -> String {
        format!(
            "📄 {} (score: {:.2})\n   {}",
            self.document.title,
            self.combined_score,
            self.chunk.content.chars().take(200).collect::<String>()
        )
    }
}

/// RAG configuration
#[derive(Debug, Clone)]
pub struct RagConfig {
    /// Weight for semantic search (0.0 - 1.0)
    pub semantic_weight: f32,
    /// Weight for keyword search (0.0 - 1.0)
    pub keyword_weight: f32,
    /// Number of results to retrieve
    pub top_k: usize,
    /// Minimum similarity threshold
    pub min_similarity: f32,
    /// Chunk configuration
    pub chunk_config: ChunkConfig,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            semantic_weight: 0.7,
            keyword_weight: 0.3,
            top_k: 5,
            min_similarity: 0.3,
            chunk_config: ChunkConfig::default(),
        }
    }
}

/// RAG Pipeline for document retrieval and question answering
pub struct RagPipeline {
    db: Arc<Database>,
    embeddings: EmbeddingsClient,
    chunker: TextChunker,
    keyword_scorer: KeywordScorer,
    config: RagConfig,
}

impl RagPipeline {
    pub fn new(db: Arc<Database>) -> Self {
        let config = RagConfig::default();
        Self {
            db,
            embeddings: EmbeddingsClient::new(),
            chunker: TextChunker::with_config(config.chunk_config.clone()),
            keyword_scorer: KeywordScorer::new(),
            config,
        }
    }

    pub fn with_config(db: Arc<Database>, config: RagConfig) -> Self {
        Self {
            db,
            embeddings: EmbeddingsClient::new(),
            chunker: TextChunker::with_config(config.chunk_config.clone()),
            keyword_scorer: KeywordScorer::new(),
            config,
        }
    }

    /// Initialize RAG tables in the database
    pub fn init_tables(db: &Database) -> Result<()> {
        let conn = db.connection();
        
        conn.execute(
            "CREATE TABLE IF NOT EXISTS rag_documents (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                doc_type TEXT DEFAULT 'text',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source)
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS rag_chunks (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                chunk_idx INTEGER NOT NULL,
                start_idx INTEGER,
                end_idx INTEGER,
                embedding BLOB,
                FOREIGN KEY (document_id) REFERENCES rag_documents(id) ON DELETE CASCADE
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_document ON rag_chunks(document_id)",
            [],
        )?;

        Ok(())
    }

    /// Ingest a document into the RAG system
    pub async fn ingest_document(&self, source: &str, title: &str, content: &str, doc_type: &str) -> Result<i64> {
        let conn = self.db.connection();

        // Insert or update document
        conn.execute(
            "INSERT OR REPLACE INTO rag_documents (source, title, content, doc_type) VALUES (?1, ?2, ?3, ?4)",
            params![source, title, content, doc_type],
        )?;

        let doc_id = conn.last_insert_rowid();

        // Delete old chunks for this document
        conn.execute(
            "DELETE FROM rag_chunks WHERE document_id = ?1",
            params![doc_id],
        )?;

        // Chunk the document
        let chunks = self.chunker.chunk_text(content, source);

        // Generate embeddings and store chunks
        for chunk in chunks {
            let embedding = self.embeddings.embed(&chunk.content).await?;
            let embedding_bytes = embedding_to_bytes(&embedding);

            conn.execute(
                "INSERT INTO rag_chunks (document_id, content, chunk_idx, start_idx, end_idx, embedding)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    doc_id,
                    chunk.content,
                    chunk.chunk_idx as i64,
                    chunk.start_idx as i64,
                    chunk.end_idx as i64,
                    embedding_bytes,
                ],
            )?;
        }

        Ok(doc_id)
    }

    /// Ingest a file into the RAG system
    pub async fn ingest_file(&self, path: &Path) -> Result<i64> {
        let content = std::fs::read_to_string(path)?;
        let source = path.to_string_lossy().to_string();
        let title = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("Unknown")
            .to_string();
        
        let doc_type = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("txt")
            .to_string();

        self.ingest_document(&source, &title, &content, &doc_type).await
    }

    /// Search for relevant chunks using hybrid search
    pub async fn search(&self, query: &str) -> Result<Vec<SearchResult>> {
        // Generate query embedding
        let query_embedding = self.embeddings.embed(query).await?;

        let conn = self.db.connection();

        // Get all chunks with embeddings
        let mut stmt = conn.prepare(
            "SELECT c.id, c.document_id, c.content, c.chunk_idx, c.embedding,
                    d.id, d.source, d.title, d.content, d.doc_type, d.created_at
             FROM rag_chunks c
             JOIN rag_documents d ON c.document_id = d.id
             WHERE c.embedding IS NOT NULL"
        )?;

        let mut results: Vec<SearchResult> = stmt.query_map([], |row| {
            let embedding_bytes: Vec<u8> = row.get(4)?;
            let embedding = bytes_to_embedding(&embedding_bytes);
            
            let chunk = StoredChunk {
                id: row.get(0)?,
                document_id: row.get(1)?,
                content: row.get(2)?,
                chunk_idx: row.get::<_, i64>(3)? as usize,
                embedding,
            };

            let document = Document {
                id: row.get(5)?,
                source: row.get(6)?,
                title: row.get(7)?,
                content: row.get(8)?,
                doc_type: row.get(9)?,
                created_at: row.get(10)?,
            };

            Ok((chunk, document))
        })?
        .filter_map(|r| r.ok())
        .map(|(chunk, document)| {
            let semantic_score = cosine_similarity(&query_embedding, &chunk.embedding);
            let keyword_score = self.keyword_scorer.score(query, &chunk.content);
            
            // Normalize keyword score (rough approximation)
            let normalized_keyword = (keyword_score / 5.0).min(1.0);
            
            let combined_score = self.config.semantic_weight * semantic_score
                + self.config.keyword_weight * normalized_keyword;

            SearchResult {
                chunk,
                document,
                semantic_score,
                keyword_score,
                combined_score,
            }
        })
        .filter(|r| r.combined_score >= self.config.min_similarity)
        .collect();

        // Sort by combined score descending
        results.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());

        // Take top k
        results.truncate(self.config.top_k);

        Ok(results)
    }

    /// Query with RAG: search for context and generate answer
    pub async fn query(&self, question: &str, llm: &OllamaClient) -> Result<RagAnswer> {
        // Search for relevant context
        let results = self.search(question).await?;

        if results.is_empty() {
            return Ok(RagAnswer {
                answer: "I couldn't find any relevant documents to answer this question.".to_string(),
                sources: vec![],
                context_used: String::new(),
            });
        }

        // Build context from search results
        let mut context_parts = Vec::new();
        let mut sources = Vec::new();

        for (i, result) in results.iter().enumerate() {
            context_parts.push(format!(
                "[Source {}: {}]\n{}",
                i + 1,
                result.document.title,
                result.chunk.content
            ));
            
            if !sources.contains(&result.document.title) {
                sources.push(result.document.title.clone());
            }
        }

        let context = context_parts.join("\n\n");

        // Generate answer using LLM
        let prompt = format!(
            r#"Answer the following question based on the provided context. 
If the context doesn't contain enough information, say so.
Cite your sources by referring to [Source N].

CONTEXT:
{}

QUESTION: {}

ANSWER:"#,
            context, question
        );

        let answer = llm.query(&prompt).await?;

        Ok(RagAnswer {
            answer: answer.trim().to_string(),
            sources,
            context_used: context,
        })
    }

    /// List all ingested documents
    pub fn list_documents(&self) -> Result<Vec<Document>> {
        let conn = self.db.connection();
        
        let mut stmt = conn.prepare(
            "SELECT id, source, title, content, doc_type, created_at FROM rag_documents ORDER BY created_at DESC"
        )?;

        let docs = stmt.query_map([], |row| {
            Ok(Document {
                id: row.get(0)?,
                source: row.get(1)?,
                title: row.get(2)?,
                content: row.get(3)?,
                doc_type: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

        Ok(docs)
    }

    /// Delete a document and its chunks
    pub fn delete_document(&self, doc_id: i64) -> Result<()> {
        let conn = self.db.connection();
        conn.execute("DELETE FROM rag_documents WHERE id = ?1", params![doc_id])?;
        Ok(())
    }

    /// Get document count and chunk count
    pub fn stats(&self) -> Result<RagStats> {
        let conn = self.db.connection();
        
        let doc_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM rag_documents",
            [],
            |row| row.get(0),
        )?;

        let chunk_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM rag_chunks",
            [],
            |row| row.get(0),
        )?;

        Ok(RagStats {
            document_count: doc_count as usize,
            chunk_count: chunk_count as usize,
        })
    }
}

/// Answer from RAG query
#[derive(Debug, Clone)]
pub struct RagAnswer {
    pub answer: String,
    pub sources: Vec<String>,
    pub context_used: String,
}

impl RagAnswer {
    pub fn display(&self) -> String {
        let mut output = self.answer.clone();
        
        if !self.sources.is_empty() {
            output.push_str("\n\n📚 Sources:\n");
            for source in &self.sources {
                output.push_str(&format!("   • {}\n", source));
            }
        }
        
        output
    }
}

/// RAG system statistics
#[derive(Debug, Clone)]
pub struct RagStats {
    pub document_count: usize,
    pub chunk_count: usize,
}

impl RagStats {
    pub fn display(&self) -> String {
        format!(
            "📊 RAG Stats: {} documents, {} chunks indexed",
            self.document_count, self.chunk_count
        )
    }
}

/// Convert embedding vector to bytes for storage
fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter()
        .flat_map(|f| f.to_le_bytes())
        .collect()
}

/// Convert bytes back to embedding vector
fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks(4)
        .map(|chunk| {
            let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
            f32::from_le_bytes(arr)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_serialization() {
        let original = vec![1.0f32, -2.5, 3.14159, 0.0];
        let bytes = embedding_to_bytes(&original);
        let restored = bytes_to_embedding(&bytes);
        
        assert_eq!(original.len(), restored.len());
        for (a, b) in original.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 0.0001);
        }
    }
}
