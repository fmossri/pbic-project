import os
import pytest
import sqlite3
import shutil
import datetime

from src.models import DocumentFile, Chunk, Domain
from src.utils import SQLiteManager
from src.config.models import SystemConfig

# Session-scoped config fixture moved outside the class
@pytest.fixture(scope="session")
def test_config(request):
    """Fixture to provide a session-scoped SystemConfig using a defined path,
    ensuring cleanup after the session.
    """
    session_storage_base = os.path.abspath(os.path.join("tests", "test_session_storage")) # Unique name
    domains_dir = os.path.join(session_storage_base, "domains")

    # Ensure the base directory is clean before the session
    if os.path.exists(session_storage_base):
        shutil.rmtree(session_storage_base)
    os.makedirs(domains_dir, exist_ok=True)

    config = SystemConfig(
        storage_base_path=session_storage_base,
        control_db_filename="session_control.db"
    )

    def cleanup_session_storage():
        print(f"\nCleaning up session storage: {session_storage_base}")
        if os.path.exists(session_storage_base):
            try:
                shutil.rmtree(session_storage_base)
            except OSError as e:
                print(f"Error removing session storage directory {session_storage_base}: {e}")

    request.addfinalizer(cleanup_session_storage)
    return config

class TestSQLiteManager:
    """Test suite for SQLiteManager class."""

    @pytest.fixture(autouse=True)
    def manager_setup(self, test_config):
        """Ensures the manager instance is available via self.manager"""
        self.manager = SQLiteManager(config=test_config)
        os.makedirs(os.path.dirname(self.manager.control_db_path), exist_ok=True)

    @pytest.fixture
    def sample_domain_db_path(self, test_config, request):
        """Provides a unique, function-scoped path for a test domain database."""
        test_name = request.node.name
        domain_db_dir = os.path.join(test_config.storage_base_path, "domains", test_name)
        db_path = os.path.join(domain_db_dir, f"{test_name}.db")
        os.makedirs(domain_db_dir, exist_ok=True)
        if os.path.exists(db_path):
            os.remove(db_path)
        yield db_path
        if os.path.exists(db_path):
            os.remove(db_path)

    @pytest.fixture
    def sample_document_file(self):
        """Create a sample DocumentFile for testing."""
        now = datetime.datetime.now()
        return DocumentFile(
            id=None,
            hash="test_hash_123",
            name="test_document.pdf",
            path="/path/to/test_document.pdf",
            total_pages=5,
            created_at=now,
            updated_at=now
        )

    @pytest.fixture
    def sample_chunk(self):
        """Create a sample Chunk for testing."""
        now = datetime.datetime.now()
        return Chunk(
            id=None,
            document_id=1,
            page_number=1,
            content="This is a test chunk content.",
            chunk_page_index=0,
            chunk_start_char_position=0,
            created_at=now
        )

    @pytest.fixture
    def sample_domain(self, test_config, sample_domain_db_path):
        """Create a sample Domain object for testing"""
        domain_name = "test_domain_sample" 
        domain_root = os.path.dirname(sample_domain_db_path)
        domain_name_fs = os.path.basename(domain_root)
        vector_store_path = os.path.join(domain_root, "vector_store", f"{domain_name_fs}.faiss")

        return Domain(
            name=domain_name,
            description="Sample description",
            keywords="sample, test",
            total_documents=0,
            db_path=sample_domain_db_path,
            vector_store_path=vector_store_path,
            embeddings_dimension=0
        )

    def test_initialization(self, test_config):
        """Test SQLiteManager initialization."""
        assert self.manager is not None
        assert self.manager.config == test_config
        expected_control_path = os.path.join(test_config.storage_base_path, test_config.control_db_filename)
        assert self.manager.control_db_path == expected_control_path
        assert self.manager.schema_path == SQLiteManager.DOMAIN_SCHEMA_PATH
        assert self.manager.db_path is None

    def test_create_database_schema_not_found(self, sample_domain_db_path):
        """Test error handling when schema file is not found."""
        invalid_schema_path = "nonexistent_schema.sql"
        temp_manager = SQLiteManager(config=self.manager.config)

        with pytest.raises(FileNotFoundError, match=f"Arquivo do schema nao encontrado em {invalid_schema_path}"):
            temp_manager._create_database(db_path=sample_domain_db_path, schema_path=invalid_schema_path)

    def test_begin(self, sample_domain_db_path):
        """Test beginning a transaction on a domain database."""
        with self.manager.get_connection(db_path=sample_domain_db_path) as conn:
            self.manager.begin(conn)
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys")
            assert cursor.fetchone()[0] == 1

    # --- Domain DB Tests ---

    def test_create_domain_database(self, sample_domain_db_path):
        """Test domain-specific database initialization via get_connection."""
        with self.manager.get_connection(db_path=sample_domain_db_path) as conn:
            assert conn is not None
        assert os.path.exists(sample_domain_db_path)

        with sqlite3.connect(sample_domain_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='document_files'")
            assert cursor.fetchone() is not None
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'")
            assert cursor.fetchone() is not None
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge_domains'")
            assert cursor.fetchone() is None

    def test_get_connection_with_db_path(self, sample_domain_db_path):
        """Test getting a domain database connection."""
        conn = self.manager.get_connection(db_path=sample_domain_db_path)
        assert conn is not None
        db_conn_path = conn.execute("PRAGMA database_list").fetchone()[2]
        assert os.path.abspath(db_conn_path) == os.path.abspath(sample_domain_db_path)
        conn.close()

    def test_insert_document_file(self, sample_document_file, sample_domain_db_path):
        """Test inserting a document file into a domain database."""
        with self.manager.get_connection(db_path=sample_domain_db_path) as conn:
            document_id = self.manager.insert_document_file(sample_document_file, conn)
            conn.commit()

            assert document_id is not None
            assert sample_document_file.id == document_id

            cursor = conn.cursor()
            cursor.execute("SELECT id, hash, name, path, total_pages FROM document_files WHERE id = ?", (document_id,))
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == document_id
            assert result[1] == sample_document_file.hash
            assert result[2] == sample_document_file.name
            assert result[3] == sample_document_file.path
            assert result[4] == sample_document_file.total_pages

    def test_insert_chunk(self, sample_document_file, sample_chunk, sample_domain_db_path):
        """Test inserting a chunk into a domain database."""
        chunk_id = None
        with self.manager.get_connection(db_path=sample_domain_db_path) as conn:
            document_id = self.manager.insert_document_file(sample_document_file, conn)
            sample_chunk.document_id = document_id

            self.manager.insert_chunks([sample_chunk], document_id, conn)
            chunk_id = sample_chunk.id
            conn.commit()

            assert chunk_id is not None

            cursor = conn.cursor()
            cursor.execute("SELECT id, document_id, page_number, content, chunk_page_index, chunk_start_char_position FROM chunks WHERE id = ?", (chunk_id,))
            result = cursor.fetchone()

            assert result is not None
            assert result[0] == chunk_id
            assert result[1] == document_id
            assert result[2] == sample_chunk.page_number
            assert result[3] == sample_chunk.content
            assert result[4] == sample_chunk.chunk_page_index
            assert result[5] == sample_chunk.chunk_start_char_position

    def test_get_chunks_by_file_id(self, sample_document_file, sample_chunk, sample_domain_db_path):
        """Test retrieving all chunks associated with a file_id."""
        with self.manager.get_connection(db_path=sample_domain_db_path) as conn:
            doc_id = self.manager.insert_document_file(sample_document_file, conn)
            
            chunk1 = sample_chunk
            chunk1.document_id = doc_id
            chunk1.content = "Content for chunk 1"
            chunk1.chunk_page_index = 0

            chunk2 = Chunk(
                id=None, document_id=doc_id, page_number=1, content="Content for chunk 2", 
                chunk_page_index=1, chunk_start_char_position=100, created_at=datetime.datetime.now()
            )
            chunk3 = Chunk(
                id=None, document_id=doc_id, page_number=2, content="Content for chunk 3", 
                chunk_page_index=0, chunk_start_char_position=200, created_at=datetime.datetime.now()
            )

            self.manager.insert_chunks([chunk1, chunk2, chunk3], doc_id, conn)
            conn.commit()

            retrieved_chunks = self.manager.get_chunks(conn, file_id=doc_id)

            assert retrieved_chunks is not None
            assert len(retrieved_chunks) == 3
            assert all(isinstance(c, Chunk) for c in retrieved_chunks)
            assert all(c.document_id == doc_id for c in retrieved_chunks)
            
            retrieved_contents = {c.content for c in retrieved_chunks}
            assert "Content for chunk 2" in retrieved_contents

    def test_get_chunks_by_chunk_ids(self, sample_document_file, sample_chunk, sample_domain_db_path):
        """Test retrieving specific chunks by their chunk_ids, preserving order."""
        chunk_ids_map = {}
        with self.manager.get_connection(db_path=sample_domain_db_path) as conn:
            doc_id = self.manager.insert_document_file(sample_document_file, conn)
            
            contents = [f"Content {i}" for i in range(5)]
            chunks_to_insert = []
            for i, content in enumerate(contents):
                chunk = Chunk(
                    id=None, document_id=doc_id, page_number=1, content=content,
                    chunk_page_index=i, chunk_start_char_position=i*50, created_at=datetime.datetime.now()
                )
                chunks_to_insert.append(chunk)

            self.manager.insert_chunks(chunks_to_insert, doc_id, conn)
            for chunk in chunks_to_insert:
                 chunk_ids_map[chunk.id] = chunk.content
            conn.commit()
        
        inserted_ids = list(chunk_ids_map.keys())
        assert len(inserted_ids) == 5

        request_ids = [inserted_ids[3], inserted_ids[1], inserted_ids[0]]
        expected_ordered_content = [
            chunk_ids_map[inserted_ids[3]],
            chunk_ids_map[inserted_ids[1]],
            chunk_ids_map[inserted_ids[0]]
        ]

        with self.manager.get_connection(db_path=sample_domain_db_path) as conn:
            retrieved_chunks = self.manager.get_chunks(conn, chunk_ids=request_ids)

            assert retrieved_chunks is not None
            assert len(retrieved_chunks) == 3
            assert all(isinstance(c, Chunk) for c in retrieved_chunks)

            retrieved_ids_ordered = [c.id for c in retrieved_chunks]
            assert retrieved_ids_ordered == request_ids

            retrieved_content_ordered = [c.content for c in retrieved_chunks]
            assert retrieved_content_ordered == expected_ordered_content

    def test_get_chunks_no_match(self, sample_document_file, sample_domain_db_path):
        """Test get_chunks returns empty list when no chunks match."""
        with self.manager.get_connection(db_path=sample_domain_db_path) as conn:
            doc_id = self.manager.insert_document_file(sample_document_file, conn)
            conn.commit()

            retrieved_by_file = self.manager.get_chunks(conn, file_id=doc_id)
            assert retrieved_by_file == []

            retrieved_by_nonexistent_ids = self.manager.get_chunks(conn, chunk_ids=[999, 1000])
            assert retrieved_by_nonexistent_ids == []

            retrieved_with_no_ids = self.manager.get_chunks(conn) 
            assert retrieved_with_no_ids == []


    # --- Control DB Tests ---

    def test_get_connection_with_control(self):
        """Test getting a control database connection."""
        conn = self.manager.get_connection(control=True)
        assert conn is not None
        db_conn_path = conn.execute("PRAGMA database_list").fetchone()[2]
        assert os.path.abspath(db_conn_path) == os.path.abspath(self.manager.control_db_path)
        conn.close()

    def test_create_control_database(self):
        """Test control database initialization via get_connection."""
        if os.path.exists(self.manager.control_db_path):
            os.remove(self.manager.control_db_path)

        with self.manager.get_connection(control=True) as conn:
             assert conn is not None
        assert os.path.exists(self.manager.control_db_path)

        with sqlite3.connect(self.manager.control_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge_domains'")
            assert cursor.fetchone() is not None

    def test_insert_domain(self, sample_domain):
        """Test inserting a domain into the control database."""
        if os.path.exists(self.manager.control_db_path):
            os.remove(self.manager.control_db_path)

        with self.manager.get_connection(control=True) as conn:
            self.manager.insert_domain(sample_domain, conn)
            conn.commit()
            cursor = conn.cursor()
            cursor.execute("SELECT name, db_path, vector_store_path FROM knowledge_domains WHERE name = ?", (sample_domain.name,))
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == sample_domain.name
            assert os.path.abspath(result[1]) == os.path.abspath(sample_domain.db_path)
            assert os.path.abspath(result[2]) == os.path.abspath(sample_domain.vector_store_path)

    def test_get_domain(self, sample_domain):
        """Test retrieving a specific domain."""
        if os.path.exists(self.manager.control_db_path):
            os.remove(self.manager.control_db_path)

        with self.manager.get_connection(control=True) as conn:
            self.manager.insert_domain(sample_domain, conn)
            conn.commit()
            retrieved_domains = self.manager.get_domain(conn, domain_name=sample_domain.name)
            assert retrieved_domains is not None
            assert len(retrieved_domains) == 1
            retrieved = retrieved_domains[0]
            assert isinstance(retrieved, Domain)
            assert retrieved.name == sample_domain.name
            assert os.path.abspath(retrieved.db_path) == os.path.abspath(sample_domain.db_path)

            assert self.manager.get_domain(conn, "non_existent") is None

    def test_get_all_domains(self, sample_domain):
        """Test retrieving all domains."""
        if os.path.exists(self.manager.control_db_path):
            os.remove(self.manager.control_db_path)

        domain_name2 = "another_domain_for_get_all"
        domain_root2 = os.path.join(self.manager.config.storage_base_path, "domains", domain_name2)
        os.makedirs(domain_root2, exist_ok=True)
        domain2 = Domain(
            name=domain_name2, description="Another desc", keywords="another", total_documents=5,
            db_path=os.path.join(domain_root2, f"{domain_name2}.db"),
            vector_store_path=os.path.join(domain_root2, "vector_store", f"{domain_name2}.faiss"),
            embeddings_dimension=384
        )

        with self.manager.get_connection(control=True) as conn:
            self.manager.insert_domain(sample_domain, conn)
            self.manager.insert_domain(domain2, conn)
            conn.commit()
            all_domains = self.manager.get_domain(conn)
            assert all_domains is not None
            assert len(all_domains) == 2
            retrieved_names = {d.name for d in all_domains}
            assert retrieved_names == {sample_domain.name, domain2.name}
            assert all(isinstance(d, Domain) for d in all_domains)

    def test_update_domain(self, sample_domain):
        """Test updating a domain."""
        if os.path.exists(self.manager.control_db_path):
            os.remove(self.manager.control_db_path)

        domain_id = None
        with self.manager.get_connection(control=True) as conn:
            self.manager.insert_domain(sample_domain, conn)
            cursor = conn.execute("SELECT id FROM knowledge_domains WHERE name = ?", (sample_domain.name,))
            domain_id = cursor.fetchone()[0]
            sample_domain.id = domain_id
            conn.commit()
        assert domain_id is not None

        updates = {"description": "Updated Description", "total_documents": 99}
        with self.manager.get_connection(control=True) as conn:
            self.manager.update_domain(sample_domain, conn, updates)
            conn.commit()
            cursor = conn.cursor()
            cursor.execute("SELECT description, total_documents FROM knowledge_domains WHERE id = ?", (domain_id,))
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == updates["description"]
            assert result[1] == updates["total_documents"]

    def test_delete_domain(self, sample_domain):
        """Test deleting a domain."""
        if os.path.exists(self.manager.control_db_path):
            os.remove(self.manager.control_db_path)

        domain_id = None
        with self.manager.get_connection(control=True) as conn:
            self.manager.insert_domain(sample_domain, conn)
            cursor = conn.execute("SELECT id FROM knowledge_domains WHERE name = ?", (sample_domain.name,))
            domain_id = cursor.fetchone()[0]
            sample_domain.id = domain_id
            conn.commit()
        assert domain_id is not None

        with self.manager.get_connection(control=True) as conn:
            self.manager.delete_domain(sample_domain, conn)
            conn.commit()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM knowledge_domains WHERE id = ?", (domain_id,))
            assert cursor.fetchone() is None
