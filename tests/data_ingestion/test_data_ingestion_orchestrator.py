import os
import pytest
import shutil
import numpy as np
from unittest.mock import patch, MagicMock, call
from src.data_ingestion import DataIngestionOrchestrator
from src.models import DocumentFile, Domain
from langchain.schema import Document
from .test_docs.generate_test_pdfs import create_test_pdf

class TestDataIngestionOrchestrator:
    """Suite de testes para a classe DataIngestionOrchestrator."""

    # Class-level fixture to create test paths only once
    test_docs_dir = os.path.join(os.path.dirname(__file__), "test_docs")
    test_pdfs_dir = os.path.join(test_docs_dir, "test_pdfs")
    empty_dir = os.path.join(test_docs_dir, "empty_dir")
    indices_dir = os.path.join("storage", "domains", "test_domain", "vector_store")
    
    # Test-specific control database path
    test_control_db_dir = os.path.join("tests", "storage", "test_control")
    test_control_db_path = os.path.join(test_control_db_dir, "test_control.db")

    @pytest.fixture(scope="class", autouse=True)
    def setup_class(self, request):
        """Setup that runs once for the entire test class"""
        # Create directories only once for all tests
        os.makedirs(self.test_pdfs_dir, exist_ok=True)
        os.makedirs(self.empty_dir, exist_ok=True)
        os.makedirs(self.indices_dir, exist_ok=True)
        os.makedirs(self.test_control_db_dir, exist_ok=True)

        # Create test PDFs only once
        create_test_pdf()
        
        # Patch the SQLiteManager.CONTROL_DB_PATH at the class level
        self.original_control_db_path = patch('src.utils.sqlite_manager.SQLiteManager.CONTROL_DB_PATH', self.test_control_db_path).start()
        
        yield
        
        # Stop the patcher
        patch.stopall()
        
        # Clean up after all tests have completed
        if os.path.exists(self.test_pdfs_dir):
            shutil.rmtree(self.test_pdfs_dir)
        
        if os.path.exists(self.empty_dir):
            shutil.rmtree(self.empty_dir)
            
        if os.path.exists(self.indices_dir):
            for file in os.listdir(self.indices_dir):
                file_path = os.path.join(self.indices_dir, file)
                if os.path.isfile(file_path) and file.endswith(".faiss"):
                    os.unlink(file_path)
                    
        # Clean up the test control database
        if os.path.exists(self.test_control_db_path):
            os.unlink(self.test_control_db_path)

    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        """Setup for each test"""
        # Mock database connection for each test
        self.mock_conn = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_conn.cursor.return_value = self.mock_cursor
        self.mock_cursor.lastrowid = 1
        
        # Mock the logger to avoid serialization issues with MagicMock
        self.mock_logger = MagicMock()
        # Make the logger methods chainable
        self.mock_logger.info.return_value = self.mock_logger
        self.mock_logger.debug.return_value = self.mock_logger
        self.mock_logger.warning.return_value = self.mock_logger
        self.mock_logger.error.return_value = self.mock_logger
        
        # Patch get_logger to return our mock
        self.logger_patcher = mocker.patch('src.utils.logger.get_logger', return_value=self.mock_logger)
        
        # Patch SentenceTransformer to avoid actual model loading
        self.model_patcher = mocker.patch('src.utils.embedding_generator.SentenceTransformer')
        mock_model = MagicMock()
        # Make sure get_sentence_embedding_dimension returns an int, not a MagicMock
        mock_model.get_sentence_embedding_dimension.return_value = 384
        self.model_patcher.return_value = mock_model
        
        # Mock faiss operations
        self.faiss_patcher = mocker.patch('src.utils.faiss_manager.faiss')
        mock_index = MagicMock()
        mock_index.ntotal = 0
        self.faiss_patcher.IndexFlatL2.return_value = mock_index
        
        yield
    
    @pytest.fixture
    def domain_fixture(self, mocker):
        """Fixture para criar um domínio de teste."""
        domain = Domain(
            id=1,
            name="test_domain",
            description="Test domain description",
            keywords="test,domain,keywords",
            total_documents=0,
            db_path="storage/domains/test_domain/test_domain.db",
            vector_store_path="storage/domains/test_domain/vector_store/test_domain.faiss",
            faiss_index=1,
            embeddings_dimension=384
        )
        
        # Mock get_domain to return our test domain
        mocker.patch('src.utils.sqlite_manager.SQLiteManager.get_domain', return_value=domain)
        
        return domain
    
    @pytest.fixture
    def orchestrator(self):
        """Create a pre-configured orchestrator with mocks for faster testing"""
        # Create the orchestrator
        orch = DataIngestionOrchestrator()
        
        # Verify the control_db_path is set to our test path
        assert orch.sqlite_manager.control_db_path == self.test_control_db_path
        
        # Mock heavy initialization of components to speed up tests
        orch.document_processor = MagicMock()
        orch.document_processor.process_document.side_effect = lambda file: file
        
        # Mock normalizer to avoid text processing overhead
        orch.text_normalizer = MagicMock()
        orch.text_normalizer.normalize.return_value = ["normalized text"]
        
        # Make sure embedding generator returns proper numpy arrays, not MagicMocks
        orch.embedding_generator = MagicMock()
        orch.embedding_generator.embedding_dimension = 384
        orch.embedding_generator.generate_embeddings.return_value = np.array([[0.1] * 384], dtype=np.float32)
        
        # Set test paths
        test_db_path = os.path.join("storage", "domains", "test_domain", "test.db")
        test_index_path = os.path.join(self.indices_dir, "test_index.faiss")
        orch.sqlite_manager.db_path = test_db_path
        orch.faiss_manager.index_path = test_index_path
        
        return orch
    
    @pytest.fixture
    def mocked_managers(self, mocker):
        """Fixture para mockar dependências comuns (SQLiteManager, FaissManager)."""
        mocks = {}

        # 1. Mock Connection Context Manager Setup for regular db
        # Use self.mock_conn from the setup fixture
        mock_context_manager = MagicMock(name="ctx_mgr_mock")
        mock_context_manager.__enter__.return_value = self.mock_conn
        mock_context_manager.__exit__.return_value = None
        
        # Create a separate mock connection for control database
        self.mock_control_conn = MagicMock()
        self.mock_control_cursor = MagicMock()
        self.mock_control_conn.cursor.return_value = self.mock_control_cursor
        self.mock_control_cursor.lastrowid = 1
        
        mock_control_context_manager = MagicMock(name="control_ctx_mgr_mock")
        mock_control_context_manager.__enter__.return_value = self.mock_control_conn
        mock_control_context_manager.__exit__.return_value = None
        
        # Mock get_connection to handle both regular and control connections
        def mock_get_connection(control=False, db_path=None):
            if control:
                return mock_control_context_manager
            return mock_context_manager
            
        mocks['get_connection'] = mocker.patch(
            'src.utils.sqlite_manager.SQLiteManager.get_connection',
            side_effect=mock_get_connection
        )

        # 2. Mock other SQLiteManager methods
        mocks['sqlite_begin'] = mocker.patch('src.utils.sqlite_manager.SQLiteManager.begin')
        mocks['insert_doc'] = mocker.patch(
            'src.utils.sqlite_manager.SQLiteManager.insert_document_file', return_value=1
        )
        mocks['insert_chunk'] = mocker.patch(
            'src.utils.sqlite_manager.SQLiteManager.insert_chunks', return_value=[1]
        )
        
        # Mock insert_domain method
        mocks['insert_domain'] = mocker.patch(
            'src.utils.sqlite_manager.SQLiteManager.insert_domain'
        )

        # 3. Mock EmbeddingGenerator method to return a NumPy array
        mock_embeddings = np.array([[0.1] * 384], dtype=np.float32)
        mocks['generate_embeddings'] = mocker.patch(
            'src.utils.embedding_generator.EmbeddingGenerator.generate_embeddings',
            return_value=mock_embeddings
        )

        # 4. Mock FaissManager method to return a list of integers (faiss indices)
        mocks['add_embeddings'] = mocker.patch(
            'src.utils.faiss_manager.FaissManager.add_embeddings',
            return_value=[1]  # Now returns a list of integers
        )

        # Return the dictionary of mocks for tests to use if needed
        return mocks
            
    def test_initialization(self, orchestrator):
        """Testa a inicialização do orquestrador."""
        assert orchestrator is not None
        assert orchestrator.document_processor is not None
        assert orchestrator.text_chunker is not None
        assert orchestrator.text_normalizer is not None
        assert orchestrator.embedding_generator is not None
        assert orchestrator.sqlite_manager is not None
        assert orchestrator.faiss_manager is not None
        assert isinstance(orchestrator.document_hashes, dict)
        
    def test_list_pdf_files(self, orchestrator):
        """Testa a listagem de arquivos PDF usando os.scandir."""
        # Testa listagem de PDFs válidos
        pdf_files = orchestrator._list_pdf_files(self.test_pdfs_dir)
        assert isinstance(pdf_files, list)
        assert len(pdf_files) > 0
        
        for file in pdf_files:
            assert isinstance(file, DocumentFile)
            assert file.name.endswith('.pdf')
        
        # Verifica se os arquivos realmente existem
        with os.scandir(self.test_pdfs_dir) as entries:
            actual_pdfs = {entry.name for entry in entries 
                         if entry.is_file() and entry.name.endswith('.pdf')}
            
        filenames = [file.name for file in pdf_files]
        assert set(filenames) == actual_pdfs
        
    def test_is_duplicate(self, orchestrator):
        """Testa o método de detecção de duplicatas."""
        mock_conn = MagicMock()
        
        # Mock cursor and fetchone for database check
        mock_cursor = MagicMock()
        mock_conn.execute.return_value = mock_cursor
        
        # Case 1: Empty hash dictionary and no DB match
        orchestrator.document_hashes = {}
        mock_cursor.fetchone.return_value = None  # No match in DB
        assert orchestrator._is_duplicate("some_hash", mock_conn) is False
        mock_conn.execute.assert_called_with("SELECT * FROM document_files WHERE hash = ?", ("some_hash",))

        # Case 2: Hash exists in the dictionary
        test_hash = "existing_hash_123"
        orchestrator.document_hashes = {"file1.pdf": test_hash, "file2.pdf": "another_hash"}
        assert orchestrator._is_duplicate(test_hash, mock_conn) is True
        # Should return True before DB check due to in-memory match

        # Case 3: Hash does not exist in dictionary but exists in DB
        orchestrator.document_hashes = {}
        mock_cursor.fetchone.return_value = (1, "db_match_hash")  # Match found in DB
        assert orchestrator._is_duplicate("db_match_hash", mock_conn) is True
        mock_conn.execute.assert_called_with("SELECT * FROM document_files WHERE hash = ?", ("db_match_hash",))

        # Case 4: Hash does not exist in dictionary or DB
        mock_cursor.fetchone.return_value = None  # No match in DB
        assert orchestrator._is_duplicate("new_hash_456", mock_conn) is False

        # Case 5: Check with None or empty hash
        assert orchestrator._is_duplicate(None, mock_conn) is False
        assert orchestrator._is_duplicate("", mock_conn) is False

    def test_process_directory_with_valid_pdfs(
        self, orchestrator, mocked_managers, mocker, domain_fixture
    ):
        """Testa o processamento de um diretório com PDFs válidos (patching methods)."""
        # Reset mocks
        self.mock_conn.reset_mock()
        self.mock_control_conn.reset_mock()

        # Mock document processor
        def mock_process_document(file):
            file.hash = "test_hash_123"
            file.pages = [Document(page_content="Test content", metadata={"page": 1})]
            file.total_pages = 1
            return file
        orchestrator.document_processor.process_document.side_effect = mock_process_document
        
        # Mock _is_duplicate to return True for the second file only
        # The test_pdfs_dir contains two files: "test_document.pdf" and "duplicate_document.pdf"
        calls = 0
        def mock_is_duplicate_func(document_hash, conn):
            nonlocal calls
            calls += 1
            # First file is unique, second file is duplicate
            return calls > 1
            
        mock_is_duplicate = mocker.patch.object(
            DataIngestionOrchestrator, 
            '_is_duplicate', 
            side_effect=mock_is_duplicate_func
        )

        # Execute the method
        orchestrator.process_directory(self.test_pdfs_dir, domain_name="test_domain")

        # Verify calls to databases
        assert mocked_managers['get_connection'].call_count >= 2 # At least 2 calls: control DB and domain DB
        
        # Verify control=True was passed to get_connection
        control_calls = [call for call in mocked_managers['get_connection'].call_args_list 
                        if call.kwargs.get('control') == True]
        assert len(control_calls) >= 1, "get_connection should be called with control=True"
        
        # Verify begin was called for both connections
        assert mocked_managers['sqlite_begin'].call_count >= 2
        
        # Verify mock_is_duplicate was called at least once
        assert mock_is_duplicate.call_count == 2, f"_is_duplicate should be called twice, once for each PDF"
        
        # Verify other operations were performed
        mocked_managers['insert_doc'].assert_called_once()
        mocked_managers['insert_chunk'].assert_called_once()
        mocked_managers['add_embeddings'].assert_called_once()
        
        # Verify commits: one for control DB, one for regular DB
        assert self.mock_control_conn.commit.call_count == 1, "Control DB should be committed once"
        assert self.mock_conn.commit.call_count == 1, "Regular DB should be committed once"
        
        # Verify rollback was called exactly once (for the duplicate file)
        self.mock_conn.rollback.assert_called_once()

    def test_duplicate_handling(self, orchestrator, mocked_managers, mocker, domain_fixture):
        """Testa o tratamento de arquivos duplicados."""
        # Reset mocks
        self.mock_conn.reset_mock()
        
        # --- Setup Scenario ---
        unique_hash = "unique_hash_abc"
        duplicate_hash = "duplicate_hash_123" # This will be pre-loaded
 
        # Actual filenames created by generate_test_pdfs.py
        unique_filename = "test_document.pdf"
        duplicate_filename = "duplicate_document.pdf"
 
        # Pre-load the hash for the file we want to be treated as duplicate
        orchestrator.document_hashes = {duplicate_filename: duplicate_hash}
 
        # --- Mock Document Processor Conditionally ---
        original_files = { # Use actual filenames
            unique_filename: unique_hash,
            duplicate_filename: duplicate_hash, # This hash already exists
        }
        def mock_process_document_conditional(file):
            # Look up the desired hash based on actual filename
            file.hash = original_files.get(file.name, "unexpected_file_hash")
            file.pages = [Document(page_content="Content", metadata={"page": 1})] # Minimal data
            file.total_pages = 1
            return file
        orchestrator.document_processor.process_document.side_effect = mock_process_document_conditional
 
        # --- Mock _is_duplicate method conditionally ---
        def mock_is_duplicate_conditional(hash_value, conn):
            # Return True only for the duplicate hash
            return hash_value == duplicate_hash
 
        mocker.patch.object(
            orchestrator, 
            '_is_duplicate', 
            side_effect=mock_is_duplicate_conditional
        )
 
        # --- Mock _find_original_document ---
        def mock_find_original_document(hash_value, conn):
            # Return a basic document file object for the duplicate
            if hash_value == duplicate_hash:
                return DocumentFile(
                    id=999, # Some fake ID
                    hash=duplicate_hash,
                    name="original_" + duplicate_filename,
                    path="/original/path/" + duplicate_filename,
                    total_pages=1
                )
            return None
 
        mocker.patch.object(
            orchestrator, 
            '_find_original_document', 
            side_effect=mock_find_original_document
        )
 
        # --- Execute Test ---
        result = orchestrator.process_directory(self.test_pdfs_dir, domain_name="test_domain")
 
        # --- Verify Results ---
        assert 'duplicate_files' in result
        assert result['duplicate_files'] > 0
        assert self.mock_conn.rollback.call_count >= 1
 
        # Check metrics for both files (success and duplicate)
        assert unique_filename in result, "Missing metrics for unique file"
        assert duplicate_filename in result, "Missing metrics for duplicate file"
 
        # Check the duplicate flag in metrics
        assert result[duplicate_filename]['is_duplicate'] == True
        assert result[unique_filename]['is_duplicate'] == False
 
        # Verify database operations happened for unique file only
        mocked_managers['insert_doc'].assert_called_once()
        # Rollback happened for duplicate, commit for unique file
        assert self.mock_conn.commit.call_count >= 1

    def test_invalid_directory(self, orchestrator):
        """Testa o tratamento de diretórios inválidos."""
        # Test with non-existent directory
        with pytest.raises(FileNotFoundError):
            orchestrator.process_directory("/nonexistent/directory", domain_name="test_domain")
        
        # Test with a file path instead of a directory
        test_file = os.path.join(self.test_pdfs_dir, "test_document.pdf")
        with open(test_file, 'w') as f:
            f.write("dummy content")
            
        with pytest.raises(NotADirectoryError):
            orchestrator.process_directory(test_file, domain_name="test_domain")

    def test_empty_directory(self, orchestrator, domain_fixture):
        """Testa o tratamento de diretórios vazios."""
        # Test with empty directory
        with pytest.raises(ValueError):
            orchestrator.process_directory(self.empty_dir, domain_name="test_domain")

    def test_add_new_domain(self, orchestrator, mocked_managers, mocker):
        """Testa a adição de um novo domínio."""
        # Reset mocks
        self.mock_conn.reset_mock()
        self.mock_control_conn.reset_mock()
        
        # Mock the get_domain method to return None (domain doesn't exist yet)
        mocker.patch('src.utils.sqlite_manager.SQLiteManager.get_domain', return_value=None)
        
        # Execute the method
        orchestrator.add_new_domain("test_domain", "Test domain description", "test,domain,keywords")
        
        # Verify calls to control database
        assert mocked_managers['get_connection'].call_count >= 1
        # Verify control=True was passed to get_connection
        control_calls = [call for call in mocked_managers['get_connection'].call_args_list 
                        if call.kwargs.get('control') == True]
        assert len(control_calls) >= 1, "get_connection should be called with control=True"
        
        # Verify begin was called
        mocked_managers['sqlite_begin'].assert_called()
        
        # Verify domain creation and insertion
        mocked_managers['insert_domain'].assert_called_once()
        
        # Verify commit on control database
        self.mock_control_conn.commit.assert_called_once() 