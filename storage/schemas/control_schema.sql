CREATE TABLE IF NOT EXISTS knowledge_domains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    keywords TEXT NOT NULL,
    total_documents INTEGER NOT NULL,
    vector_store_path TEXT NOT NULL,
    db_path TEXT NOT NULL,
    embeddings_dimension INTEGER NOT NULL,
    faiss_index INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
