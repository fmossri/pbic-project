import os
import pytest
import sqlite3

from src.models import DocumentFile, Chunk, Domain
from src.utils import SQLiteManager

class TestSQLiteManager:
    """Test suite for SQLiteManager class."""

    REAL_SCHEMA_PATH = os.path.join("storage", "schemas", "schema.sql")
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        # Create a unique path for each test to avoid conflicts
        test_dir = os.path.join("tests", "storage", "domains", "test_domain")
        os.makedirs(test_dir, exist_ok=True)
        
        # Use process ID to make path unique
        temp_db_path = os.path.join(test_dir, f"test_{os.getpid()}.db")
        
        yield temp_db_path
        
        # Clean up after tests
        if os.path.exists(temp_db_path):
            os.unlink(temp_db_path)
    
    @pytest.fixture
    def temp_control_db_path(self):
        """Create a temporary control database path."""
        # Create a unique path for each test to avoid conflicts
        test_dir = os.path.join("tests", "storage", "domains")
        os.makedirs(test_dir, exist_ok=True)
        
        # Use process ID to make path unique
        temp_control_db_path = os.path.join(test_dir, f"control_test_{os.getpid()}.db")
        
        yield temp_control_db_path
        
        # Clean up after tests
        if os.path.exists(temp_control_db_path):
            os.unlink(temp_control_db_path)
    
    @pytest.fixture
    def schema_path(self):
        """Create a temporary schema path with a test schema."""
        # Create a path for the test schema
        schema_dir = os.path.join("tests", "storage", "schemas")
        os.makedirs(schema_dir, exist_ok=True)
        schema_path = os.path.join(schema_dir, "test_schema.sql")
        
        # Write a simple schema to the file
        with open(schema_path, "w") as f:
            f.write("CREATE TABLE IF NOT EXISTS document_files (id INTEGER PRIMARY KEY, hash TEXT, name TEXT, path TEXT, total_pages INTEGER);\n")
            f.write("CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, document_id INTEGER, page_number INTEGER, content TEXT, chunk_page_index INTEGER, faiss_index INTEGER, chunk_start_char_position INTEGER);\n")
            f.write("CREATE TABLE IF NOT EXISTS domains (id INTEGER PRIMARY KEY, name TEXT, description TEXT, keywords TEXT, total_documents INTEGER, db_path TEXT, vector_store_path TEXT, faiss_index INTEGER, embeddings_dimension INTEGER);\n")
            
        
        yield schema_path
        
        # Clean up after tests
        if os.path.exists(schema_path):
            os.unlink(schema_path)
    
    @pytest.fixture
    def control_schema_path(self):
        """Create a temporary control schema path with a test schema."""
        # Create a path for the test control schema
        schema_dir = os.path.join("tests", "storage", "schemas")
        os.makedirs(schema_dir, exist_ok=True)
        control_schema_path = os.path.join(schema_dir, "test_control_schema.sql")
        
        # Write a simple control schema to the file
        with open(control_schema_path, "w") as f:
            f.write("CREATE TABLE IF NOT EXISTS document_files (id INTEGER PRIMARY KEY, hash TEXT, name TEXT, path TEXT, total_pages INTEGER);\n")
            
        yield control_schema_path
        
        # Clean up after tests
        if os.path.exists(control_schema_path):
            os.unlink(control_schema_path)
    
    @pytest.fixture
    def sqlite_manager(self, temp_db_path, schema_path, temp_control_db_path, control_schema_path):
        """Create a SQLiteManager instance for testing."""
        manager = SQLiteManager()
        manager.db_path = temp_db_path
        manager.schema_path = schema_path
        manager.control_db_path = temp_control_db_path
        # Override class constant for testing
        manager.CONTROL_SCHEMA_PATH = control_schema_path
        return manager
    
    @pytest.fixture
    def sample_document_file(self):
        """Create a sample DocumentFile for testing."""
        return DocumentFile(
            id=None,
            hash="test_hash_123",
            name="test_document.pdf",
            path="/path/to/test_document.pdf",
            total_pages=5
        )
    
    @pytest.fixture
    def sample_chunk(self):
        """Create a sample Chunk for testing."""
        return Chunk(
            id=None,
            document_id=1,  # Using default value of 1 to avoid validation error
            page_number=1,
            content="This is a test chunk content.",
            chunk_page_index=0,
            chunk_start_char_position=0,
            faiss_index=None  # Updated to include faiss_index field with None value
        )
    
    def test_initialization(self, temp_db_path, schema_path):
        """Test SQLiteManager initialization."""
        manager = SQLiteManager()
        manager.db_path = temp_db_path
        manager.schema_path = schema_path
        
        assert manager is not None
        assert manager.db_path == temp_db_path
        assert manager.schema_path == schema_path
        
        # Test default paths
        default_manager = SQLiteManager()
        assert default_manager.db_path == SQLiteManager.DEFAULT_DB_PATH
        assert default_manager.schema_path == SQLiteManager.DEFAULT_SCHEMA_PATH
        assert default_manager.control_db_path == SQLiteManager.CONTROL_DB_PATH
    
    def test_initialize_database(self, sqlite_manager):
        """Test database initialization."""
        # Initialize the database
        sqlite_manager.initialize_database()
        
        # Verify the database file was created
        assert os.path.exists(sqlite_manager.db_path)
        
        # Verify tables were created
        with sqlite3.connect(sqlite_manager.db_path) as conn:
            cursor = conn.cursor()
            
            # Check for document_files table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='document_files'")
            assert cursor.fetchone() is not None
            
            # Check for chunks table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'")
            assert cursor.fetchone() is not None
    
    def test_initialize_database_with_control(self, sqlite_manager):
        """Test control database initialization."""
        # Initialize the control database
        sqlite_manager.initialize_database(control=True)
        
        # Verify the control database file was created
        assert os.path.exists(sqlite_manager.db_path)
        assert sqlite_manager.db_path == sqlite_manager.control_db_path
        
        # Verify schema path was set to control schema
        assert sqlite_manager.schema_path == sqlite_manager.CONTROL_SCHEMA_PATH
        
        # Verify tables were created
        with sqlite3.connect(sqlite_manager.db_path) as conn:
            cursor = conn.cursor()
            
            # Check for document_files table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='document_files'")
            assert cursor.fetchone() is not None
            
            # Chunks table should not exist in control DB schema
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'")
            assert cursor.fetchone() is None
    
    def test_initialize_database_file_not_found(self, sqlite_manager):
        """Test error handling when schema file is not found."""
        # Set an invalid schema path
        sqlite_manager.schema_path = "nonexistent_schema.sql"
        
        # Try to initialize the database
        with pytest.raises(FileNotFoundError):
            sqlite_manager.initialize_database()
    
    def test_get_connection(self, sqlite_manager):
        """Test getting a database connection."""
        conn = sqlite_manager.get_connection()
        assert conn is not None
        conn.close()
    
    def test_get_connection_with_control(self, sqlite_manager):
        """Test getting a control database connection."""
        # Get a connection to the control database
        conn = sqlite_manager.get_connection(control=True)
        assert conn is not None
        
        # Verify it's connected to the control database
        db_path = conn.execute("PRAGMA database_list").fetchone()[2]
        # Compare paths regardless of absolute/relative format
        assert os.path.abspath(db_path) == os.path.abspath(sqlite_manager.control_db_path)
        
        # Verify the control database file was created
        assert os.path.exists(sqlite_manager.control_db_path)
        
        conn.close()
    
    def test_begin_transaction(self, sqlite_manager):
        """Test beginning a transaction."""
        # Initialize the database
        sqlite_manager.initialize_database()
        
        # Get a connection
        conn = sqlite_manager.get_connection()
        
        # Begin a transaction
        sqlite_manager.begin(conn)
        
        # Verify transaction is active
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys")
        assert cursor.fetchone()[0] == 1  # Foreign keys should be ON
        
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
            cursor.execute("SELECT id, hash, name, path, total_pages FROM document_files WHERE id = ?", (document_id,))
            result = cursor.fetchone()
            
            assert result is not None
            assert result[0] == document_id
            assert result[1] == sample_document_file.hash
            assert result[2] == sample_document_file.name
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
            
            # Set the document_id on the chunk and faiss_index
            sample_chunk.document_id = document_id
            sample_chunk.faiss_index = 42  # Add a test faiss_index
            
            # Insert the chunk
            sqlite_manager.insert_chunks([sample_chunk], document_id, conn)
            
            # Verify the chunk was inserted
            cursor = conn.cursor()
            cursor.execute("SELECT id, document_id, page_number, content, chunk_page_index, faiss_index, chunk_start_char_position FROM chunks WHERE document_id = ?", (document_id,))
            result = cursor.fetchone()
            
            assert result is not None
            chunk_id = result[0]
            assert result[1] == document_id
            assert result[2] == sample_chunk.page_number
            assert result[3] == sample_chunk.content
            assert result[4] == sample_chunk.chunk_page_index
            assert result[5] == sample_chunk.faiss_index  # Verify faiss_index was inserted correctly
            assert result[6] == sample_chunk.chunk_start_char_position
            
            # Verify the ID was set on the model
            assert sample_chunk.id == chunk_id
    
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
    
    def test_transaction_rollback_control_db(self, sqlite_manager, sample_document_file):
        """Test transaction rollback in control database."""
        # Initialize the control database
        sqlite_manager.initialize_database(control=True)
        
        # Start a transaction and insert a document
        conn = sqlite_manager.get_connection(control=True)
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
    
    def test_get_connection_with_control_and_context_manager(self, sqlite_manager):
        """Test using get_connection with control=True and a context manager."""
        # Initialize the control database
        sqlite_manager.initialize_database(control=True)
        
        # Use the connection in a with block
        with sqlite_manager.get_connection(control=True) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sqlite_version();")
            version = cursor.fetchone()
            assert version is not None
        
        # Connection should be closed after the with block
        assert conn.in_transaction == False  # Verify the connection is not in a transaction
    
    def test_get_chunks_content(self, sqlite_manager, sample_document_file, sample_chunk):
        """Test retrieving chunks content by faiss indices."""
        # Initialize the database
        sqlite_manager.initialize_database()
        
        # Insert a document and a chunk
        with sqlite_manager.get_connection() as conn:
            document_id = sqlite_manager.insert_document_file(sample_document_file, conn)
            
            # Set up multiple chunks with different faiss indices
            chunks = []
            faiss_indices = [10, 20, 30]
            contents = ["Content 1", "Content 2", "Content 3"]
            
            for i in range(3):
                chunk = Chunk(
                    id=None,
                    document_id=document_id,
                    page_number=1,
                    content=contents[i],
                    chunk_page_index=i,
                    chunk_start_char_position=i*100,
                    faiss_index=faiss_indices[i]
                )
                chunks.append(chunk)
            
            # Insert the chunks
            sqlite_manager.insert_chunks(chunks, document_id, conn)
            
            # Retrieve chunks content by faiss indices
            # Request in a different order to test ordering
            request_indices = [30, 10, 20]
            chunks_content = sqlite_manager.get_chunks_content(conn, request_indices)
            
            # Verify the correct content was returned in the correct order
            assert len(chunks_content) == 3
            assert chunks_content[0] == "Content 3"  # faiss_index 30
            assert chunks_content[1] == "Content 1"  # faiss_index 10
            assert chunks_content[2] == "Content 2"  # faiss_index 20
    
    def test_get_domain(self, sqlite_manager):
        """Test retrieving a domain by name."""
        # Initialize the database
        sqlite_manager.initialize_database()
        
        # Create and insert a domain
        domain_name = "test_domain"
        domain = Domain(
            id=None,
            name=domain_name,
            description="Test domain description",
            keywords="test,domain,keywords",
            total_documents=0,
            db_path="path/to/db",
            vector_store_path="path/to/vectors",
            faiss_index=1,
            embeddings_dimension=384
        )
        
        with sqlite_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO domains (name, description, keywords, total_documents, db_path, vector_store_path, faiss_index, embeddings_dimension) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (domain.name, domain.description, domain.keywords, domain.total_documents, domain.db_path, domain.vector_store_path, domain.faiss_index, domain.embeddings_dimension)
            )
            conn.commit()
            
            # Retrieve the domain - this now works because we updated sqlite_manager.py
            retrieved_domain = sqlite_manager.get_domain(conn, domain_name)
            
            # Verify the domain was retrieved correctly
            assert retrieved_domain is not None
            assert retrieved_domain.name == domain_name
            assert retrieved_domain.description == domain.description
            assert retrieved_domain.keywords == domain.keywords
            assert retrieved_domain.total_documents == domain.total_documents
            assert retrieved_domain.db_path == domain.db_path
            assert retrieved_domain.vector_store_path == domain.vector_store_path
            assert retrieved_domain.faiss_index == domain.faiss_index
            assert retrieved_domain.embeddings_dimension == domain.embeddings_dimension
            
            # Test retrieving a non-existent domain
            non_existent = sqlite_manager.get_domain(conn, "non_existent_domain")
            assert non_existent is None