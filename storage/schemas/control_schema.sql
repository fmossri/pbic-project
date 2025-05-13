CREATE TABLE IF NOT EXISTS knowledge_domains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    description TEXT NOT NULL,
    keywords TEXT NOT NULL,
    total_documents INTEGER NOT NULL,
    vector_store_path TEXT NOT NULL UNIQUE,
    db_path TEXT NOT NULL UNIQUE,
    embeddings_dimension INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS knowledge_domain_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain_id INTEGER NOT NULL,
    embeddings_model TEXT NOT NULL,
    normalize_embeddings BOOLEAN NOT NULL,
    combine_embeddings BOOLEAN NOT NULL,
    embedding_weight REAL NOT NULL,
    faiss_index_type TEXT NOT NULL,
    chunking_strategy TEXT NOT NULL,
    chunk_size INTEGER NOT NULL,
    chunk_overlap INTEGER NOT NULL,
    cluster_distance_threshold REAL NOT NULL,
    chunk_max_words INTEGER NOT NULL,
    FOREIGN KEY (domain_id) REFERENCES knowledge_domains(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

DROP TRIGGER IF EXISTS update_knowledge_domain_updated_at;

CREATE TRIGGER update_knowledge_domain_updated_at
    AFTER UPDATE ON knowledge_domains
    FOR EACH ROW
    BEGIN
        UPDATE knowledge_domains SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;