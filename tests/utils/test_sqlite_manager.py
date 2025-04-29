import os
import pytest
import sqlite3
import shutil

from src.models import DocumentFile, Chunk, Domain
from src.utils import SQLiteManager

class TestSQLiteManager:
    """Test suite for SQLiteManager class."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        # Create a unique path for each test to avoid conflicts
        test_dir = os.path.join("tests", "storage", "domains", "test_domain")
        os.makedirs(test_dir, exist_ok=True)
        
        # Use process ID to make path unique
        temp_db_path = os.path.join(test_dir, f"test_{os.getpid()}.db")
        
        yield temp_db_path
        
        # Clean up using shutil.rmtree
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

    @pytest.fixture
    def temp_control_db_path(self):
        """Create a temporary control database path."""
        # Create a unique path for each test to avoid conflicts
        test_dir = os.path.join("tests", "storage", "domains")
        os.makedirs(test_dir, exist_ok=True)
        
        # Use process ID to make path unique
        temp_control_db_path = os.path.join(test_dir, f"control_test_{os.getpid()}.db")
        
        yield temp_control_db_path
        
        # Clean up using shutil.rmtree
        if os.path.exists(test_dir):
            # Remove the specific DB file first if it exists (optional, but safer)
            if os.path.exists(temp_control_db_path):
                os.unlink(temp_control_db_path)
            # Then attempt to remove the directory if empty, otherwise use rmtree if needed

    @pytest.fixture
    def schema_path(self):
        """Provide the actual path to the domain-specific schema file."""
        # Return the default schema path used by the manager
        yield SQLiteManager.DOMAIN_SCHEMA_PATH
        # No cleanup needed as we are not creating a temporary file
    
    @pytest.fixture
    def control_schema_path(self):
        """Provide the actual path to the control schema file."""
        # Return the control schema path used by the manager
        yield SQLiteManager.CONTROL_SCHEMA_PATH
        # No cleanup needed as we are not creating a temporary file
    
    @pytest.fixture
    def sqlite_manager(self, temp_db_path, schema_path, temp_control_db_path, control_schema_path):
        """Create a SQLiteManager instance using temporary DB paths and actual schema paths."""
        manager = SQLiteManager()
        # Set instance paths to temporary test database paths
        manager.db_path = temp_db_path
        manager.control_db_path = temp_control_db_path
        # Set the schema paths the manager instance will use for initialization
        manager.schema_path = schema_path # Uses actual path from schema_path fixture
        manager.CONTROL_SCHEMA_PATH = control_schema_path # Uses actual path from control_schema_path fixture
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
    
    @pytest.fixture
    def sample_domain(self):
        """Create a sample Domain object for testing."""
        return Domain(
            name="test_domain_sample",
            description="Sample description",
            keywords="sample, test",
            total_documents=0,
            db_path=os.path.join("tests", "storage", "domains", "test_domain_sample", "test_domain_sample.db"),
            vector_store_path=os.path.join("tests", "storage", "domains", "test_domain_sample", "vector_store", "test_domain_sample.faiss"),
            embeddings_dimension=0 # Default initial value
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
        assert default_manager.db_path == SQLiteManager.TEST_DB_PATH
        assert default_manager.schema_path == SQLiteManager.DOMAIN_SCHEMA_PATH
        assert default_manager.control_db_path == SQLiteManager.CONTROL_DB_PATH
    
    def test_create_database(self, sqlite_manager):
        """Test domain-specific database initialization using the temporary schema."""
        # Initialize the database
        sqlite_manager.create_database(db_path=sqlite_manager.db_path) # Explicitly pass path
        
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

            # Check that knowledge_domains table does NOT exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge_domains'")
            assert cursor.fetchone() is None, "knowledge_domains table should not exist in domain DB"
    
    def test_create_database_with_control(self, sqlite_manager):
        """Test control database initialization."""
        # Initialize the control database
        sqlite_manager.create_database(control=True)
        
        # Verify the control database file was created
        assert os.path.exists(sqlite_manager.control_db_path)

        # Verify table was created based on the temporary control_schema_path fixture
        with sqlite3.connect(sqlite_manager.control_db_path) as conn:
            cursor = conn.cursor()
            
            # Check for knowledge_domains table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge_domains'")
            assert cursor.fetchone() is not None
    
    def test_create_database_file_not_found(self, sqlite_manager):
        """Test error handling when schema file is not found."""
        invalid_schema_path = "nonexistent_schema.sql"
        
        # Try to initialize the database with an invalid schema path
        with pytest.raises(FileNotFoundError, match=f"Arquivo do schema nao encontrado em {invalid_schema_path}"):
            # Pass both the db_path and the invalid schema_path
            sqlite_manager.create_database(db_path=sqlite_manager.db_path, schema_path=invalid_schema_path)
    
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
        # Initialize the database, explicitly providing the path
        sqlite_manager.create_database(db_path=sqlite_manager.db_path)
        
        # Get a connection, explicitly providing the path
        conn = sqlite_manager.get_connection(db_path=sqlite_manager.db_path)
        
        # Begin a transaction
        sqlite_manager.begin(conn)
        
        # Verify transaction is active
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys")
        assert cursor.fetchone()[0] == 1  # Foreign keys should be ON
        
        conn.close()
    
    def test_insert_document_file(self, sqlite_manager, sample_document_file):
        """Test inserting a document file."""
        # Initialize the database, explicitly providing the path
        sqlite_manager.create_database(db_path=sqlite_manager.db_path)
        
        # Insert the document file using get_connection with path
        with sqlite_manager.get_connection(db_path=sqlite_manager.db_path) as conn:
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
        # Initialize the database, explicitly providing the path
        sqlite_manager.create_database(db_path=sqlite_manager.db_path)
        
        # Insert the document file first using get_connection with path
        with sqlite_manager.get_connection(db_path=sqlite_manager.db_path) as conn:
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
        # Initialize the database, explicitly providing the path
        sqlite_manager.create_database(db_path=sqlite_manager.db_path)
        
        # Start a transaction and insert a document using get_connection with path
        conn = sqlite_manager.get_connection(db_path=sqlite_manager.db_path)
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
    
    def test_transaction_rollback_control_db(self, sqlite_manager, sample_domain):
        """Test transaction rollback in control database using a domain."""
        # Initialize the control database
        sqlite_manager.create_database(control=True)
        
        # Start a transaction and insert a domain
        conn = sqlite_manager.get_connection(control=True)
        sqlite_manager.begin(conn)
        
        # Use insert_domain instead of insert_document_file
        sqlite_manager.insert_domain(sample_domain, conn)
        # Get the ID assigned during insertion (needed for verification)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM knowledge_domains WHERE name = ?", (sample_domain.name,))
        domain_id = cursor.fetchone()[0]
        
        # Verify the domain exists in the transaction
        cursor.execute("SELECT COUNT(*) FROM knowledge_domains WHERE id = ?", (domain_id,))
        count = cursor.fetchone()[0]
        assert count == 1
        
        # Rollback the transaction
        conn.rollback()
        
        # Verify the domain doesn't exist after rollback
        # Need to reopen connection or use the same cursor if still valid after rollback
        cursor.execute("SELECT COUNT(*) FROM knowledge_domains WHERE id = ?", (domain_id,))
        count = cursor.fetchone()[0]
        assert count == 0
        
        conn.close()
    
    def test_transaction_commit(self, sqlite_manager, sample_document_file):
        """Test transaction commit."""
        # Initialize the database, explicitly providing the path
        sqlite_manager.create_database(db_path=sqlite_manager.db_path)
        
        # Start a transaction and insert a document using get_connection with path
        conn = sqlite_manager.get_connection(db_path=sqlite_manager.db_path)
        sqlite_manager.begin(conn)
        
        document_id = sqlite_manager.insert_document_file(sample_document_file, conn)
        
        # Commit the transaction
        conn.commit()
        conn.close()
        
        # Open a new connection and verify the document exists
        conn2 = sqlite_manager.get_connection(db_path=sqlite_manager.db_path)
        cursor = conn2.cursor()
        cursor.execute("SELECT COUNT(*) FROM document_files WHERE id = ?", (document_id,))
        count = cursor.fetchone()[0]
        assert count == 1
        
        conn2.close()
    
    def test_sqlite_error_handling(self, sqlite_manager, sample_document_file):
        """Test SQLite error handling."""
        # Initialize the database, explicitly providing the path
        sqlite_manager.create_database(db_path=sqlite_manager.db_path)
        
        # Corrupt the document_files table to force an error
        conn = sqlite_manager.get_connection(db_path=sqlite_manager.db_path)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE document_files")
        conn.commit()
        conn.close()
        
        # Try to insert a document to trigger an error
        with sqlite_manager.get_connection(db_path=sqlite_manager.db_path) as conn:
            with pytest.raises(sqlite3.Error):
                sqlite_manager.insert_document_file(sample_document_file, conn)
    
    def test_get_connection_with_context_manager(self, sqlite_manager):
        """Test using get_connection with a context manager."""
        # Initialize the database, explicitly providing the path
        sqlite_manager.create_database(db_path=sqlite_manager.db_path)
        
        # Use the connection in a with block, explicitly providing the path
        with sqlite_manager.get_connection(db_path=sqlite_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sqlite_version();")
            version = cursor.fetchone()
            assert version is not None
        
        # Connection should be closed after the with block
        assert conn.in_transaction == False  # Verify the connection is not in a transaction
    
    def test_get_connection_with_control_and_context_manager(self, sqlite_manager):
        """Test using get_connection with control=True and a context manager."""
        # Initialize the control database
        sqlite_manager.create_database(control=True)
        
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
        # Initialize the database, explicitly providing the path
        sqlite_manager.create_database(db_path=sqlite_manager.db_path)
        
        # Insert a document and a chunk using get_connection with path
        with sqlite_manager.get_connection(db_path=sqlite_manager.db_path) as conn:
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
    
    def test_get_domain(self, sqlite_manager, sample_domain):
        """Test retrieving domains from the control database."""
        # Initialize control DB
        sqlite_manager.create_database(control=True)
        
        # Insert a sample domain directly for testing retrieval
        with sqlite_manager.get_connection(control=True) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO knowledge_domains 
                   (name, description, keywords, total_documents, db_path, vector_store_path, embeddings_dimension) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (sample_domain.name, sample_domain.description, sample_domain.keywords, 
                 sample_domain.total_documents, sample_domain.db_path, sample_domain.vector_store_path,
                 sample_domain.embeddings_dimension)
            )
            conn.commit()

            # Test retrieving the specific domain by name
            retrieved_domains = sqlite_manager.get_domain(conn, domain_name=sample_domain.name)
            assert retrieved_domains is not None
            assert len(retrieved_domains) == 1
            retrieved_domain = retrieved_domains[0]
            assert isinstance(retrieved_domain, Domain)
            assert retrieved_domain.name == sample_domain.name
            assert retrieved_domain.description == sample_domain.description
            assert retrieved_domain.db_path == sample_domain.db_path
            assert retrieved_domain.embeddings_dimension == sample_domain.embeddings_dimension

            # Test retrieving a non-existent domain
            non_existent = sqlite_manager.get_domain(conn, domain_name="non_existent_domain")
            assert non_existent is None

    def test_update_domain(self, sqlite_manager, sample_domain):
        """Test updating a domain in the control database."""
        # Initialize control DB and insert sample domain
        sqlite_manager.create_database(control=True)
        with sqlite_manager.get_connection(control=True) as conn:
            sqlite_manager.insert_domain(sample_domain, conn)
            cursor = conn.execute("SELECT id FROM knowledge_domains WHERE name = ?", (sample_domain.name,))
            sample_domain.id = cursor.fetchone()[0]
            conn.commit()

        # Define updates
        updates = {
            "description": "Updated description",
            "keywords": "updated, sample",
            "total_documents": 10
        }

        # Perform the update using the manager method
        with sqlite_manager.get_connection(control=True) as conn:
            sqlite_manager.update_domain(sample_domain, conn, updates)
            conn.commit()

            # Verify the update in the database
            cursor = conn.cursor()
            cursor.execute("SELECT description, keywords, total_documents FROM knowledge_domains WHERE id = ?", (sample_domain.id,))
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == updates["description"]
            assert result[1] == updates["keywords"]
            assert result[2] == updates["total_documents"]

    def test_delete_domain(self, sqlite_manager, sample_domain, mocker):
        """Test deleting a domain from the control database."""
        # Initialize control DB and insert sample domain
        sqlite_manager.create_database(control=True)
        
        # Note: We no longer need to create dummy files as SQLiteManager doesn't delete them.

        with sqlite_manager.get_connection(control=True) as conn:
            sqlite_manager.insert_domain(sample_domain, conn)
            cursor = conn.execute("SELECT id FROM knowledge_domains WHERE name = ?", (sample_domain.name,))
            sample_domain.id = cursor.fetchone()[0]
            conn.commit()

        # Perform the deletion
        with sqlite_manager.get_connection(control=True) as conn:
            sqlite_manager.delete_domain(sample_domain, conn)
            conn.commit()

            # Verify the domain is deleted from the database
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM knowledge_domains WHERE id = ?", (sample_domain.id,))
            assert cursor.fetchone() is None

    def test_insert_domain(self, sqlite_manager, sample_domain):
        """Test inserting a domain into the control database."""
        # Initialize control DB
        sqlite_manager.create_database(control=True)
        
        # Insert the domain using the manager method
        with sqlite_manager.get_connection(control=True) as conn:
            sqlite_manager.insert_domain(sample_domain, conn)
            conn.commit()

            # Verify the domain was inserted correctly
            cursor = conn.cursor()
            cursor.execute(
                """SELECT name, description, keywords, total_documents, db_path, vector_store_path, embeddings_dimension 
                   FROM knowledge_domains WHERE name = ?""", 
                (sample_domain.name,)
            )
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == sample_domain.name
            assert result[1] == sample_domain.description
            assert result[2] == sample_domain.keywords
            assert result[3] == sample_domain.total_documents
            assert result[4] == sample_domain.db_path
            assert result[5] == sample_domain.vector_store_path
            assert result[6] == sample_domain.embeddings_dimension

    def test_get_all_domains(self, sqlite_manager, sample_domain):
        """Test retrieving all domains from the control database."""
        # Initialize control DB
        sqlite_manager.create_database(control=True)
        
        # Insert multiple sample domains
        domain1 = sample_domain
        domain2 = Domain(
            name="another_domain", description="Another desc", keywords="another", total_documents=5,
            db_path="/path/to/another.db", vector_store_path="/path/to/another.faiss", embeddings_dimension=384
        )
        
        with sqlite_manager.get_connection(control=True) as conn:
            sqlite_manager.insert_domain(domain1, conn)
            sqlite_manager.insert_domain(domain2, conn)
            conn.commit()

            # Retrieve all domains
            all_domains = sqlite_manager.get_domain(conn, domain_name=None) # Pass None to get all
            
            assert all_domains is not None
            assert len(all_domains) == 2
            
            # Check if the retrieved domains match the inserted ones (names are unique)
            retrieved_names = {d.name for d in all_domains}
            assert retrieved_names == {domain1.name, domain2.name}
            
            # Verify types
            assert all(isinstance(d, Domain) for d in all_domains)
