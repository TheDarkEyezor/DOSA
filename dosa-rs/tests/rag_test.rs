// Integration test for the RAG system
use std::sync::Arc;
use std::path::Path;

// Import the modules we need to test
#[tokio::test]
async fn test_text_chunking() {
    use dosa::intelligence::chunking::{TextChunker, ChunkConfig};
    
    let chunker = TextChunker::with_config(ChunkConfig {
        chunk_size: 100,
        chunk_overlap: 20,
        ..Default::default()
    });
    
    let text = r#"This is the first paragraph with some content.

This is the second paragraph with different content.

This is the third paragraph with even more content to test chunking properly."#;
    
    let chunks = chunker.chunk_text(text, "test.txt");
    
    assert!(!chunks.is_empty(), "Should produce at least one chunk");
    
    for chunk in &chunks {
        assert!(!chunk.content.is_empty(), "Chunk content should not be empty");
        assert_eq!(chunk.source, "test.txt", "Source should match");
    }
    
    println!("✅ Text chunking test passed: {} chunks created", chunks.len());
}

#[test]
fn test_cosine_similarity() {
    use dosa::intelligence::embeddings::cosine_similarity;
    
    // Identical vectors should have similarity 1.0
    let a = vec![1.0f32, 0.0, 0.0];
    let b = vec![1.0f32, 0.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!((sim - 1.0).abs() < 0.001, "Identical vectors should have similarity ~1.0");
    
    // Orthogonal vectors should have similarity 0.0
    let c = vec![0.0f32, 1.0, 0.0];
    let sim_orth = cosine_similarity(&a, &c);
    assert!(sim_orth.abs() < 0.001, "Orthogonal vectors should have similarity ~0.0");
    
    // Opposite vectors should have similarity -1.0
    let d = vec![-1.0f32, 0.0, 0.0];
    let sim_opp = cosine_similarity(&a, &d);
    assert!((sim_opp + 1.0).abs() < 0.001, "Opposite vectors should have similarity ~-1.0");
    
    println!("✅ Cosine similarity test passed");
}

#[test]
fn test_keyword_scorer() {
    use dosa::intelligence::chunking::KeywordScorer;
    
    let scorer = KeywordScorer::new();
    
    let doc = "The quick brown fox jumps over the lazy dog";
    
    // Query with matching terms should have positive score
    let score1 = scorer.score("quick fox", doc);
    assert!(score1 > 0.0, "Matching query should have positive score");
    
    // Query with no matching terms should have zero score
    let score2 = scorer.score("elephant zebra", doc);
    assert!(score2 == 0.0, "Non-matching query should have zero score");
    
    // Query with more matches should have higher score
    let score3 = scorer.score("quick brown fox jumps", doc);
    assert!(score3 > score1, "Query with more matches should have higher score");
    
    println!("✅ Keyword scorer test passed");
}

#[test]
fn test_chunk_config_default() {
    use dosa::intelligence::chunking::ChunkConfig;
    
    let config = ChunkConfig::default();
    
    assert_eq!(config.chunk_size, 1000, "Default chunk size should be 1000");
    assert_eq!(config.chunk_overlap, 200, "Default overlap should be 200");
    assert!(!config.separators.is_empty(), "Should have default separators");
    
    println!("✅ ChunkConfig defaults test passed");
}

#[test]
fn test_embedding_serialization() {
    // Test that we can serialize and deserialize embeddings
    let original = vec![1.0f32, -2.5, 3.14159, 0.0, -0.00001, 1000.0];
    
    // Simulate embedding_to_bytes
    let bytes: Vec<u8> = original.iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    
    // Simulate bytes_to_embedding
    let restored: Vec<f32> = bytes.chunks(4)
        .map(|chunk| {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            f32::from_le_bytes(arr)
        })
        .collect();
    
    assert_eq!(original.len(), restored.len());
    for (a, b) in original.iter().zip(restored.iter()) {
        assert!((a - b).abs() < 0.0001, "Values should match after serialization");
    }
    
    println!("✅ Embedding serialization test passed");
}
