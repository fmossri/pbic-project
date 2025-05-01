import os
import pytest
import shutil
import numpy as np
from unittest.mock import patch, MagicMock
from src.data_ingestion import DataIngestionOrchestrator
from src.models import DocumentFile, Domain
from langchain.schema import Document
# Import necessary config models
from src.config.models import AppConfig, SystemConfig, IngestionConfig, EmbeddingConfig, VectorStoreConfig, QueryConfig, TextNormalizerConfig, LLMConfig
from .test_docs.generate_test_pdfs import create_test_pdf

class TestDataIngestionOrchestrator:
    """Suite de testes para a classe DataIngestionOrchestrator."""

    # Class-level fixture to create test paths only once
    test_docs_dir = os.path.join(os.path.dirname(__file__), "test_docs")
    test_pdfs_dir = os.path.join(test_docs_dir, "test_pdfs")
    empty_dir = os.path.join(test_docs_dir, "empty_dir")
    test_storage_base = os.path.join("tests", "test_storage") # Define a base for test storage
    test_domain_dir = os.path.join(test_storage_base, "domains", "test_domain")
    indices_dir = os.path.join(test_domain_dir, "vector_store") # Corrected path based on base
    
    # Test-specific control database path using the test storage base
    test_control_db_dir = os.path.join(test_storage_base, "control") # Directory for control DB
    test_control_db_filename = "test_control.db"
    test_control_db_path = os.path.join(test_control_db_dir, test_control_db_filename) # Full path

    # Domain paths using test base
    test_domain_db_path = os.path.join(test_domain_dir, "test_domain.db")
    test_domain_vector_store_path = os.path.join(indices_dir, "test_domain.faiss")

    @pytest.fixture(scope="class", autouse=True)
    def setup_class(self, request):
        """Setup that runs once for the entire test class"""
        # Create directories only once for all tests
        os.makedirs(self.test_pdfs_dir, exist_ok=True)
        os.makedirs(self.empty_dir, exist_ok=True)
        os.makedirs(self.indices_dir, exist_ok=True)
        os.makedirs(self.test_control_db_dir, exist_ok=True) # Use control DB dir

        # Create test PDFs only once
        create_test_pdf()
        
        # REMOVED: Patch for SQLiteManager.CONTROL_DB_PATH
        # self.original_control_db_path = patch('src.utils.sqlite_manager.SQLiteManager.CONTROL_DB_PATH', self.test_control_db_path).start()
        
        yield
        
        # Stop any remaining patches (though we removed the main one)
        patch.stopall()
        
        # Clean up after all tests have completed
        if os.path.exists(self.test_pdfs_dir):
            shutil.rmtree(self.test_pdfs_dir)
        
        if os.path.exists(self.empty_dir):
            shutil.rmtree(self.empty_dir)
            
        # Clean up the entire test storage base directory
        if os.path.exists(self.test_storage_base):
             shutil.rmtree(self.test_storage_base)

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
    def test_app_config(self) -> AppConfig:
        """Creates an AppConfig instance pointing to test directories."""
        return AppConfig(
            system=SystemConfig(
                storage_base_path=self.test_storage_base, # Use test base path
                control_db_filename=self.test_control_db_filename # Use test filename
            ),
            ingestion=IngestionConfig(chunk_size=500, chunk_overlap=50), # Example test values
            embedding=EmbeddingConfig(model_name="paraphrase-MiniLM-L3-v2"), # Different model for testing?
            vector_store=VectorStoreConfig(),
            query=QueryConfig(),
            llm=LLMConfig(), # Provide a default LLMConfig instance
            text_normalizer=TextNormalizerConfig()
        )

    @pytest.fixture
    def domain_fixture(self, mocker, request):
        """Fixture para criar um domínio de teste com dimensão de embedding opcional."""
        # Get initial dimension from test parameter, default to 384 if not provided
        initial_dimension = getattr(request, "param", 384)

        # Use the actual Domain model definition
        domain = Domain(
            id=1,
            name="test_domain",
            description="Test domain description",
            keywords="test,domain,keywords",
            total_documents=0, # Default or initial value
            db_path=self.test_domain_db_path, # Use test path
            vector_store_path=self.test_domain_vector_store_path, # Use test path
            embeddings_dimension=initial_dimension # Use parameter here
            # created_at is Optional[datetime]=None by default in model
        )

        # Mock get_domain to return this specific domain object in a list
        mock_get_domain = mocker.patch('src.utils.sqlite_manager.SQLiteManager.get_domain', return_value=[domain])

        # Mock update_domain to verify it gets called (use the mock from mocked_managers if preferred,
        # but patching here ensures it's available when fixture is used)
        mock_update_domain = mocker.patch('src.utils.sqlite_manager.SQLiteManager.update_domain')

        # Return the domain and the mocks
        yield domain, mock_get_domain, mock_update_domain
    
    @pytest.fixture
    def configured_orchestrator(self, test_app_config):
        """Creates an orchestrator instance initialized with the test configuration."""
        # Instantiate with the test config - real dependencies will be created
        # but configured for testing (e.g., test paths, embedding model if changed).
        # Mocking of external calls (like HF model download or DB writes)
        # should happen in specific tests or the `mocked_managers` fixture.
        orch = DataIngestionOrchestrator(config=test_app_config)
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

        # 5. Mock update_domain method
        mocks['update_domain'] = mocker.patch(
            'src.utils.sqlite_manager.SQLiteManager.update_domain'
        )

        # Return the dictionary of mocks for tests to use if needed
        return mocks
            
    def test_initialization(self, configured_orchestrator):
        """Testa a inicialização do orquestrador com configuração."""
        orch = configured_orchestrator # Use the configured orchestrator
        assert orch is not None
        assert orch.config is not None # Check if config was stored
        # Check if components were initialized (assuming they are not None after init)
        assert orch.document_processor is not None
        assert orch.text_chunker is not None
        assert orch.text_normalizer is not None
        assert orch.embedding_generator is not None
        assert orch.sqlite_manager is not None
        assert orch.faiss_manager is not None
        # Check if configuration values were applied (example)
        assert orch.text_chunker.config.chunk_size == 500 # From test_app_config
        assert orch.embedding_generator.config.model_name == "paraphrase-MiniLM-L3-v2" # From test_app_config
        assert orch.sqlite_manager.config.storage_base_path == self.test_storage_base # From test_app_config
        
    def test_list_pdf_files(self, configured_orchestrator):
        """Testa a listagem de arquivos PDF usando os.scandir."""
        orch = configured_orchestrator # Use the configured orchestrator
        # Testa listagem de PDFs válidos
        pdf_files = orch._list_pdf_files(self.test_pdfs_dir)
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
        
    def test_is_duplicate(self, configured_orchestrator):
        """Testa o método de detecção de duplicatas."""
        orch = configured_orchestrator # Use the configured orchestrator
        mock_conn = MagicMock()
        
        # Mock cursor and fetchone for database check
        mock_cursor = MagicMock()
        mock_conn.execute.return_value = mock_cursor
        
        # Case 1: Empty hash dictionary and no DB match
        orch.document_hashes = {}
        mock_cursor.fetchone.return_value = None  # No match in DB
        assert orch._is_duplicate("some_hash", mock_conn) is False
        mock_conn.execute.assert_called_with("SELECT * FROM document_files WHERE hash = ?", ("some_hash",))

        # Case 2: Hash exists in the dictionary
        test_hash = "existing_hash_123"
        orch.document_hashes = {"file1.pdf": test_hash, "file2.pdf": "another_hash"}
        assert orch._is_duplicate(test_hash, mock_conn) is True
        # Should return True before DB check due to in-memory match

        # Case 3: Hash does not exist in dictionary but exists in DB
        orch.document_hashes = {}
        mock_cursor.fetchone.return_value = (1, "db_match_hash")  # Match found in DB
        assert orch._is_duplicate("db_match_hash", mock_conn) is True
        mock_conn.execute.assert_called_with("SELECT * FROM document_files WHERE hash = ?", ("db_match_hash",))

        # Case 4: Hash does not exist in dictionary or DB
        mock_cursor.fetchone.return_value = None  # No match in DB
        assert orch._is_duplicate("new_hash_456", mock_conn) is False

        # Case 5: Check with None or empty hash
        assert orch._is_duplicate(None, mock_conn) is False
        assert orch._is_duplicate("", mock_conn) is False

    def test_process_directory_with_valid_pdfs(
        self, configured_orchestrator, mocked_managers, mocker, domain_fixture
    ):
        """Testa o processamento de um diretório com PDFs válidos (patching methods)."""
        orch = configured_orchestrator # Use the configured orchestrator
        # domain_fixture provides domain with dimension=384 by default
        initial_domain, mock_get_domain, mock_update_domain = domain_fixture # Unpack 3 values

        # Reset mocks
        self.mock_conn.reset_mock()
        self.mock_control_conn.reset_mock()
        # Mock update_domain again locally to ensure clean state for assertions
        mock_update_domain = mocker.patch('src.utils.sqlite_manager.SQLiteManager.update_domain')

        # Mock document processor
        def mock_process_document(file):
            file.hash = f"test_hash_{file.name}" # Give unique hash based on name
            file.pages = [Document(page_content=f"Content for {file.name}", metadata={"page": 1})]
            file.total_pages = 1
            return file
        mocker.patch.object(orch.document_processor, 'process_document', side_effect=mock_process_document)
        
        # Mock _is_duplicate to return True for the second file only
        # The test_pdfs_dir contains two files: "test_document.pdf" and "duplicate_document.pdf"
        calls = 0
        def mock_is_duplicate_func(document_hash, conn):
            nonlocal calls
            calls += 1
            # Assume 'duplicate_document.pdf' leads to the second call
            return calls > 1
            
        mock_is_duplicate = mocker.patch.object(
            orch, # Patch the method on the instance
            '_is_duplicate', 
            side_effect=mock_is_duplicate_func
        )

        # Mock _find_original_document needed for duplicate case log/metrics
        def mock_find_original(hash_value, conn):
             if calls > 1: # Corresponds to the duplicate call
                 return DocumentFile(id=1, hash="test_hash_test_document.pdf", name="test_document.pdf", path="/fake/path", total_pages=1)
             return None
        mocker.patch.object(orch, '_find_original_document', side_effect=mock_find_original)

        # Execute the method
        orch.process_directory(self.test_pdfs_dir, domain_name="test_domain")

        # Verify calls to databases
        # At least 3 calls: initial control DB, domain DB for each file, final control DB for update
        assert mocked_managers['get_connection'].call_count >= 3
        
        # Verify control=True was passed to get_connection at least twice (initial and final)
        control_calls = [call for call in mocked_managers['get_connection'].call_args_list 
                        if call.kwargs.get('control') == True]
        assert len(control_calls) >= 2, "get_connection should be called with control=True twice"
        
        # Verify begin was called for each connection
        # Should be called once per file processed (for domain DB)
        assert mocked_managers['sqlite_begin'].call_count >= 2
        
        # Verify mock_is_duplicate was called at least once
        assert mock_is_duplicate.call_count == 2, f"_is_duplicate should be called twice, once for each PDF"
        
        # Verify other operations were performed
        mocked_managers['insert_doc'].assert_called_once()
        mocked_managers['insert_chunk'].assert_called_once()
        mocked_managers['add_embeddings'].assert_called_once()
        
        # Verify domain DB committed once (for successful file)
        assert self.mock_conn.commit.call_count == 1, "Domain DB should be committed once"
        
        # Verify control DB committed twice:
        # 1. After getting the domain (dimension wasn't 0, so no update here)
        # 2. After updating domain metrics (total_docs) at the end
        assert self.mock_control_conn.commit.call_count == 2, f"Control DB should be committed twice (get, update count), got {self.mock_control_conn.commit.call_count}"
        
        # Verify rollback was called exactly once (for the duplicate file)
        self.mock_conn.rollback.assert_called_once()
        
        # Verify update_domain was called only *once* (for total_documents)
        mock_update_domain.assert_called_once()
        
        # Verify the domain was updated with the correct values (total_docs=1)
        call_args_docs_update = mock_update_domain.call_args
        # Check signature: update_domain(domain_object, conn, update_dict)
        assert call_args_docs_update.args[0].id == initial_domain.id, "Incorrect domain object ID for total_docs update"
        assert call_args_docs_update.args[1] == self.mock_control_conn, "Incorrect connection for total_docs update"
        assert call_args_docs_update.args[2] == {"total_documents": 1}, "Incorrect update dict for total_docs"

    def test_duplicate_handling(self, configured_orchestrator, mocked_managers, mocker, domain_fixture):
        """Testa o tratamento de arquivos duplicados."""
        orch = configured_orchestrator # Use the configured orchestrator
        # Reset mocks
        self.mock_conn.reset_mock()
        
        # --- Setup Scenario ---
        unique_hash = "unique_hash_abc"
        duplicate_hash = "duplicate_hash_123" # This will be pre-loaded
 
        # Actual filenames created by generate_test_pdfs.py
        unique_filename = "test_document.pdf"
        duplicate_filename = "duplicate_document.pdf"
 
        # Pre-load the hash *in the database mock* instead of the instance dictionary
        # (Orchestrator clears its internal dict per run)
        # We'll control this via the _is_duplicate mock primarily.

        # --- Mock Document Processor Conditionally ---
        original_files = { # Use actual filenames
            unique_filename: unique_hash,
            duplicate_filename: duplicate_hash, # This hash will trigger duplicate logic
        }
        def mock_process_document_conditional(file):
            # Look up the desired hash based on actual filename
            file.hash = original_files.get(file.name, "unexpected_file_hash")
            file.pages = [Document(page_content="Content", metadata={"page": 1})] # Minimal data
            file.total_pages = 1
            return file
        mocker.patch.object(orch.document_processor, 'process_document', side_effect=mock_process_document_conditional)

        # --- Mock _is_duplicate method conditionally ---
        def mock_is_duplicate_conditional(hash_value, conn):
            # Return True only for the duplicate hash
            return hash_value == duplicate_hash
 
        mocker.patch.object(
            orch, 
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
            orch, 
            '_find_original_document', 
            side_effect=mock_find_original_document
        )
 
        # --- Execute Test ---
        result = orch.process_directory(self.test_pdfs_dir, domain_name="test_domain")
 
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

    def test_invalid_directory(self, configured_orchestrator):
        """Testa o tratamento de diretórios inválidos."""
        orch = configured_orchestrator # Use the configured orchestrator
        # Test with non-existent directory
        with pytest.raises(FileNotFoundError):
            orch.process_directory("/nonexistent/directory", domain_name="test_domain")
        
        # Test with a file path instead of a directory
        test_file = os.path.join(self.test_pdfs_dir, "test_document.pdf")
        with open(test_file, 'w') as f:
            f.write("dummy content")
            
        with pytest.raises(NotADirectoryError):
            orch.process_directory(test_file, domain_name="test_domain")

    def test_empty_directory(self, configured_orchestrator, domain_fixture):
        """Testa o tratamento de diretórios vazios."""
        orch = configured_orchestrator # Use the configured orchestrator
        # Test with empty directory
        with pytest.raises(ValueError, match="Nenhum arquivo PDF encontrado"): # Match specific error
            orch.process_directory(self.empty_dir, domain_name="test_domain")

    @pytest.mark.parametrize("domain_fixture", [0], indirect=True)
    def test_process_directory_updates_embedding_dimension(
        self, configured_orchestrator, mocked_managers, mocker, domain_fixture
    ):
        """
        Testa se process_directory atualiza embeddings_dimension se for 0.
        """
        orch = configured_orchestrator # Use the configured orchestrator
        # domain_fixture now provides the domain with embeddings_dimension=0
        initial_domain, mock_get_domain, mock_update_domain = domain_fixture
        # Get expected dimension from the orchestrator's generator instance
        expected_final_dimension = orch.embedding_generator.embedding_dimension

        # Reset mocks used within the test
        self.mock_control_conn.reset_mock()
        mock_update_domain.reset_mock() # Reset the specific mock we want to check

        # Mock document processing to simulate one successful file
        def mock_process_document(file):
            file.hash = "unique_hash_for_dim_test"
            file.pages = [Document(page_content="Test content dim update", metadata={"page": 1})]
            file.total_pages = 1
            return file
        mocker.patch.object(orch.document_processor, 'process_document', side_effect=mock_process_document)

        # Mock _is_duplicate to always return False for this test
        mocker.patch.object(orch,'_is_duplicate', return_value=False)

        # --- Execute ---
        orch.process_directory(self.test_pdfs_dir, domain_name="test_domain")

        # --- Assertions ---
        # Check that get_domain was called initially
        mock_get_domain.assert_called_once_with(self.mock_control_conn, "test_domain")

        # Check that update_domain was called at least once (for the dimension update)
        mock_update_domain.assert_called() # We expect at least the dimension update call

        # Check the *first* call to update_domain (for embedding dimension)
        # Ensure call_args_list is not empty before accessing index 0
        assert len(mock_update_domain.call_args_list) >= 1, "update_domain should have been called at least once"
        first_call_args = mock_update_domain.call_args_list[0].args
        assert first_call_args[0].id == initial_domain.id, "Incorrect domain object ID for dimension update"
        assert first_call_args[1] == self.mock_control_conn, "Incorrect connection for dimension update"
        # Expect both dimension and model name in the update dict
        expected_update_dict = {
            "embeddings_dimension": expected_final_dimension,
            "embeddings_model": orch.embedding_generator.config.model_name
        }
        assert first_call_args[2] == expected_update_dict, "Incorrect update dict for embedding dimension/model"

        # Check if the second call (for total_documents) also happened
        assert len(mock_update_domain.call_args_list) >= 2, "Expected update_domain call for total_documents as well"
        second_call_args = mock_update_domain.call_args_list[1].args
        assert second_call_args[0].id == initial_domain.id, "Incorrect domain object ID for total_docs update"
        assert second_call_args[1] == self.mock_control_conn, "Incorrect connection for total_docs update"
        assert second_call_args[2] == {"total_documents": len(os.listdir(self.test_pdfs_dir))}, "Incorrect update dict for total_docs" # Should process all files

