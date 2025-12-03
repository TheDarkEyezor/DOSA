pub mod alerts;
pub mod query;
pub mod extraction;
pub mod graphrag;
pub mod kag;
pub mod document;
pub mod context;
pub mod multihop;
pub mod embeddings;
pub mod chunking;
pub mod rag;

pub use alerts::AlertSystem;
pub use query::{NaturalLanguageQuery, IntentRouter};
pub use extraction::EntityExtractor;
pub use graphrag::GraphRAG;
pub use kag::LogicalSolver;
pub use document::{DocumentIngester, IngestionResult};
pub use context::{ConversationContext, EventReference, EmailReference, PersonReference, PendingAction};
pub use multihop::{MultiHopEngine, MultiHopResult, MultiHopData};
pub use embeddings::{EmbeddingsClient, cosine_similarity};
pub use chunking::{TextChunker, ChunkConfig, TextChunk, KeywordScorer};
pub use rag::{RagPipeline, RagConfig, RagAnswer, RagStats, SearchResult};

