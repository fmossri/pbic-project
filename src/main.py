import os

# Initialize components
db_manager = SQLiteManager(
    db_path=os.path.join("storage", "domains", "test_domain", "content.db"),
    schema_path=os.path.join("storage", "schemas", "schema.sql")
)
faiss_manager = FaissManager(
    index_path=os.path.join("storage", "domains", "test_domain", "vector_store", "index.faiss")
) 