PRAGMA foreign_keys = ON;

/*CREATE TABLE IF NOT EXISTS domains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL
    description TEXT NOT NULL
    keywords TEXT NOT NULL
)*/

CREATE TABLE IF NOT EXISTS document_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    hash TEXT NOT NULL UNIQUE,
    total_pages INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    content TEXT NOT NULL UNIQUE,
    metadata TEXT NOT NULL,  -- JSON string containing page_list, index_list, keywords, filename
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES document_files(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

DROP TRIGGER IF EXISTS update_document_file_updated_at;

CREATE TRIGGER update_document_file_updated_at
    AFTER UPDATE ON document_files
    FOR EACH ROW
    BEGIN
        UPDATE document_files SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;
