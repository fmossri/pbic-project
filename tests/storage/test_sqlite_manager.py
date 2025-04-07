import os
import pytest
import sqlite3
import tempfile
import numpy as np

from components.storage.sqlite_manager import SQLiteManager
from components.models.document_file import DocumentFile
from components.models.chunk import Chunk
from components.models.embedding import Embedding

class TestSQLiteManager:
    """Test suite for SQLiteManager class."""

    REAL_SCHEMA_PATH = os.path.join("databases", "schemas", "schema.sql")
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_file:
            temp_db_path = temp_file.name
            
        yield temp_db_path
        
        # Cleanup after tests
        if os.path.exists(temp_db_path):
            os.unlink(temp_db_path)
    
    @pytest.fixture
    def schema_path(self):
        """Return the path to the real schema file."""
        # Check if schema file exists
        if not os.path.exists(self.REAL_SCHEMA_PATH):
            raise FileNotFoundError(f"Schema file not found at {self.REAL_SCHEMA_PATH}")
        return self.REAL_SCHEMA_PATH

    @pytest.fixture
    def sqlite_manager(self, temp_db_path, schema_path):
        """Create an SQLiteManager instance pointing to the temporary files."""
        return SQLiteManager(db_path=temp_db_path, schema_path=schema_path)
    
    @pytest.fixture
    def sample_document_file(self):
        """Create a sample DocumentFile instance for testing."""
        return DocumentFile(
            id=None,
            hash="sample_hash_123",
            name="test_document.pdf",
            path="/path/to/test_document.pdf",
            total_pages=2
        )
    
    @pytest.fixture
    def sample_chunk(self):
        """Create a sample Chunk instance for testing."""
        return Chunk(
            id=None,
            document_id=1,  # This will be set correctly in tests
            page_number=1,
            chunk_page_index=0,
            chunk_start_char_position=0,
            content="This is a test chunk content for testing the database insertion."
        )
    
    @pytest.fixture
    def sample_embedding(self):
        """Create a sample Embedding instance for testing."""
        return Embedding(
            id=None,
            chunk_id=1,  # This will be set correctly in tests
            faiss_index_path="/path/to/faiss/index.bin",
            chunk_faiss_index=0,
            dimension=384,
            embedding=np.array([0.1] * 384, dtype=np.float32)
        )
    
    def test_initialization(self, temp_db_path, schema_path):
        """Test SQLiteManager initialization."""
        manager = SQLiteManager(db_path=temp_db_path, schema_path=schema_path)
        assert manager.db_path == temp_db_path
        assert manager.schema_path == schema_path
    
    def test_initialize_database(self, sqlite_manager):
        """Test database initialization."""
        # Initialize the database
        sqlite_manager.initialize_database()
        
        # Verify that the database was created and has the expected tables
        conn = sqlite3.connect(sqlite_manager.db_path)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        
        assert "document_files" in tables
        assert "chunks" in tables
        assert "embeddings" in tables
        
        conn.close()
    
    def test_initialize_database_file_not_found(self, sqlite_manager):
        """Test database initialization with non-existent schema file."""
        # Set an invalid schema path
        sqlite_manager.schema_path = "/nonexistent/path/schema.sql"
        
        # Try to initialize the database
        with pytest.raises(FileNotFoundError):
            sqlite_manager.initialize_database()
    
    def test_get_connection(self, sqlite_manager):
        """Test getting a database connection."""
        conn = sqlite_manager.get_connection()
        assert isinstance(conn, sqlite3.Connection)
        conn.close()
    
    def test_begin_transaction(self, sqlite_manager):
        """Test beginning a transaction."""
        conn = sqlite_manager.get_connection()
        sqlite_manager.begin(conn)
        
        # Check if foreign keys are enabled
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys;")
        foreign_keys_enabled = cursor.fetchone()[0]
        assert foreign_keys_enabled == 1
        
        conn.close()
    
    def test_insert_document_file(self, sqlite_manager, sample_document_file):
        """Test inserting a document file."""
        # Initialize the database
        sqlite_manager.initialize_database()
        
        # Insert the document file
        with sqlite_manager.get_connection() as conn:
            document_id = sqlite_manager.insert_document_file(sample_document_file, conn)
            
            # Verify the document was inserted
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, hash, path, total_pages FROM document_files WHERE id = ?", (document_id,))
            result = cursor.fetchone()
            
            assert result is not None
            assert result[0] == document_id
            assert result[1] == sample_document_file.name
            assert result[2] == sample_document_file.hash
            assert result[3] == sample_document_file.path
            assert result[4] == sample_document_file.total_pages
            
            # Verify the ID was set on the model
            assert sample_document_file.id == document_id
    
    def test_insert_chunk(self, sqlite_manager, sample_document_file, sample_chunk):
        """Test inserting a chunk."""
        # Initialize the database
        sqlite_manager.initialize_database()
        
        # Insert the document file first
        with sqlite_manager.get_connection() as conn:
            document_id = sqlite_manager.insert_document_file(sample_document_file, conn)
            
            # Set the document_id on the chunk
            sample_chunk.document_id = document_id
            
            # Insert the chunk
            chunk_id = sqlite_manager.insert_chunk(sample_chunk, document_id, conn)
            
            # Verify the chunk was inserted
            cursor = conn.cursor()
            cursor.execute("SELECT id, document_id, page_number, content, chunk_page_index FROM chunks WHERE id = ?", (chunk_id,))
            result = cursor.fetchone()
            
            assert result is not None
            assert result[0] == chunk_id
            assert result[1] == document_id
            assert result[2] == sample_chunk.page_number
            assert result[3] == sample_chunk.content
            assert result[4] == sample_chunk.chunk_page_index
            
            # Verify the ID was set on the model
            assert sample_chunk.id == chunk_id
    
    def test_insert_embedding(self, sqlite_manager, sample_document_file, sample_chunk, sample_embedding):
        """Test inserting an embedding."""
        # Initialize the database
        sqlite_manager.initialize_database()
        
        # Insert the document file and chunk first
        with sqlite_manager.get_connection() as conn:
            document_id = sqlite_manager.insert_document_file(sample_document_file, conn)
            
            # Set the document_id on the chunk
            sample_chunk.document_id = document_id
            
            # Insert the chunk
            chunk_id = sqlite_manager.insert_chunk(sample_chunk, document_id, conn)
            
            # Set the chunk_id on the embedding
            sample_embedding.chunk_id = chunk_id
            
            # Insert the embedding
            embedding_id = sqlite_manager.insert_embedding(sample_embedding, conn)
            
            # Verify the embedding was inserted
            cursor = conn.cursor()
            cursor.execute("SELECT id, chunk_id, faiss_index_path, chunk_faiss_index, dimension FROM embeddings WHERE id = ?", (embedding_id,))
            result = cursor.fetchone()
            
            assert result is not None
            assert result[0] == embedding_id
            assert result[1] == chunk_id
            assert result[2] == sample_embedding.faiss_index_path
            assert result[3] == sample_embedding.chunk_faiss_index
            assert result[4] == sample_embedding.dimension
            
            # Verify the ID was set on the model
            assert sample_embedding.id == embedding_id
    
    def test_transaction_rollback(self, sqlite_manager, sample_document_file):
        """Test transaction rollback."""
        # Initialize the database
        sqlite_manager.initialize_database()
        
        # Start a transaction and insert a document
        conn = sqlite_manager.get_connection()
        sqlite_manager.begin(conn)
        
        document_id = sqlite_manager.insert_document_file(sample_document_file, conn)
        
        # Verify the document exists in the transaction
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM document_files WHERE id = ?", (document_id,))
        count = cursor.fetchone()[0]
        assert count == 1
        
        # Rollback the transaction
        conn.rollback()
        
        # Verify the document doesn't exist after rollback
        cursor.execute("SELECT COUNT(*) FROM document_files WHERE id = ?", (document_id,))
        count = cursor.fetchone()[0]
        assert count == 0
        
        conn.close()
    
    def test_transaction_commit(self, sqlite_manager, sample_document_file):
        """Test transaction commit."""
        # Initialize the database
        sqlite_manager.initialize_database()
        
        # Start a transaction and insert a document
        conn = sqlite_manager.get_connection()
        sqlite_manager.begin(conn)
        
        document_id = sqlite_manager.insert_document_file(sample_document_file, conn)
        
        # Commit the transaction
        conn.commit()
        conn.close()
        
        # Open a new connection and verify the document exists
        conn2 = sqlite_manager.get_connection()
        cursor = conn2.cursor()
        cursor.execute("SELECT COUNT(*) FROM document_files WHERE id = ?", (document_id,))
        count = cursor.fetchone()[0]
        assert count == 1
        
        conn2.close()
    
    def test_sqlite_error_handling(self, sqlite_manager, sample_document_file):
        """Test SQLite error handling."""
        # Initialize the database
        sqlite_manager.initialize_database()
        
        # Corrupt the document_files table to force an error
        conn = sqlite_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute("DROP TABLE document_files")
        conn.commit()
        conn.close()
        
        # Try to insert a document to trigger an error
        with sqlite_manager.get_connection() as conn:
            with pytest.raises(sqlite3.Error):
                sqlite_manager.insert_document_file(sample_document_file, conn)
    
    def test_get_connection_with_context_manager(self, sqlite_manager):
        """Test using get_connection with a context manager."""
        # Initialize the database
        sqlite_manager.initialize_database()
        
        # Use the connection in a with block
        with sqlite_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sqlite_version();")
            version = cursor.fetchone()
            assert version is not None
        
        # Connection should be closed after the with block
        assert conn.in_transaction == False  # Verify the connection is not in a transaction