CREATE TABLE IF NOT EXISTS knowledge_domains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    keywords TEXT NOT NULL,
    total_documents INTEGER NOT NULL,
    faiss_index_path TEXT NOT NULL,
    domain_db_path TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
