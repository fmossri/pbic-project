import os
import pytest
import shutil
import numpy as np
from unittest.mock import patch, MagicMock
from components.data_ingestion import DataIngestionOrchestrator
from components.models import DocumentFile, Chunk, Embedding
from langchain.schema import Document
from .test_docs.generate_test_pdfs import create_test_pdf

class TestDataIngestionOrchestrator:
    """Suite de testes para a classe DataIngestionOrchestrator."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Configura o ambiente de teste."""
        print("\nSetting up test environment...")
        
        # Obtém o caminho para o diretório test_docs
        test_docs_dir = os.path.join(os.path.dirname(__file__), "test_docs")
        self.test_pdfs_dir = os.path.join(test_docs_dir, "test_pdfs")
        self.empty_dir = os.path.join(test_docs_dir, "empty_dir")
        self.indices_dir = os.path.join("indices")
        
        # Cria o diretório de teste se não existir
        os.makedirs(self.test_pdfs_dir, exist_ok=True)
        os.makedirs(self.empty_dir, exist_ok=True)
        
        # Cria os arquivos PDF de teste
        create_test_pdf()

        # Mock do banco de dados
        self.mock_conn = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_conn.cursor.return_value = self.mock_cursor
        self.mock_cursor.lastrowid = 1
        
        yield
        
        # Limpa os arquivos de teste após os testes
        if os.path.exists(self.test_pdfs_dir):
            shutil.rmtree(self.test_pdfs_dir)
        
        if os.path.exists(self.empty_dir):
            shutil.rmtree(self.empty_dir)

        for file in os.listdir(self.indices_dir):
            file_path = os.path.join(self.indices_dir, file)
            if os.path.isfile(file_path) and file.endswith(".faiss"):
                os.unlink(file_path)

    
    @pytest.fixture
    def mocked_managers(self, mocker):
        """Fixture para mockar dependências comuns (SQLiteManager, FaissManager)."""
        mocks = {}

        # 1. Mock Connection Context Manager Setup
        # Use self.mock_conn from the setup fixture
        mock_context_manager = MagicMock(name="ctx_mgr_mock")
        mock_context_manager.__enter__.return_value = self.mock_conn
        mock_context_manager.__exit__.return_value = None
        mocks['get_connection'] = mocker.patch(
            'components.storage.sqlite_manager.SQLiteManager.get_connection',
            return_value=mock_context_manager
        )

        # 2. Mock other SQLiteManager methods
        mocks['sqlite_begin'] = mocker.patch('components.storage.sqlite_manager.SQLiteManager.begin')
        mocks['insert_doc'] = mocker.patch(
            'components.storage.sqlite_manager.SQLiteManager.insert_document_file', return_value=1
        )
        mocks['insert_chunk'] = mocker.patch(
            'components.storage.sqlite_manager.SQLiteManager.insert_chunk', return_value=1
        )
        mocks['insert_embedding'] = mocker.patch('components.storage.sqlite_manager.SQLiteManager.insert_embedding')

        # 3. Mock FaissManager method
        mock_embedding_data = Embedding(
            id=None, chunk_id=1, dimension=384, faiss_index_path="test/path",
            chunk_faiss_index=0, embedding=np.array([0.1] * 384)
        )
        mocks['add_embeddings'] = mocker.patch(
            'components.storage.faiss_manager.FaissManager.add_embeddings',
            return_value=[mock_embedding_data]
        )

        # Return the dictionary of mocks for tests to use if needed
        return mocks
            
    def test_initialization(self):
        """Testa a inicialização do orquestrador."""
        orchestrator = DataIngestionOrchestrator()
        assert orchestrator is not None
        assert orchestrator.document_processor is not None
        assert orchestrator.text_chunker is not None
        assert orchestrator.text_normalizer is not None
        assert orchestrator.embedding_generator is not None
        assert orchestrator.sqlite_manager is not None
        assert orchestrator.faiss_manager is not None
        assert isinstance(orchestrator.document_hashes, dict)

    def test_list_pdf_files(self):
        """Testa a listagem de arquivos PDF usando os.scandir."""
        orchestrator = DataIngestionOrchestrator()
        
        # Testa listagem de PDFs válidos
        pdf_files = orchestrator.list_pdf_files(self.test_pdfs_dir)
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

    def test_is_duplicate(self):
        """Testa o método de detecção de duplicatas."""
        orchestrator = DataIngestionOrchestrator()

        # Case 1: Empty hash dictionary
        orchestrator.document_hashes = {}
        assert orchestrator._is_duplicate("some_hash") is False

        # Case 2: Hash exists in the dictionary
        test_hash = "existing_hash_123"
        orchestrator.document_hashes = {"file1.pdf": test_hash, "file2.pdf": "another_hash"}
        assert orchestrator._is_duplicate(test_hash) is True

        # Case 3: Hash does not exist in the dictionary
        assert orchestrator._is_duplicate("new_hash_456") is False

        # Case 4: Check with None or empty hash (optional edge case)
        assert orchestrator._is_duplicate(None) is False
        assert orchestrator._is_duplicate("") is False # Assuming empty hash is not considered duplicate


    def test_process_directory_with_valid_pdfs(
        self, mocked_managers
    ):
        """Testa o processamento de um diretório com PDFs válidos (patching methods)."""


        orchestrator = DataIngestionOrchestrator()

        orchestrator.process_directory(self.test_pdfs_dir)

        mocked_managers['get_connection'].assert_called_once()
        mocked_managers['sqlite_begin'].assert_called_once_with(self.mock_conn)
        mocked_managers['insert_doc'].assert_called()
        mocked_managers['insert_chunk'].assert_called()
        mocked_managers['insert_embedding'].assert_called()
        mocked_managers['add_embeddings'].assert_called_once()

        self.mock_conn.commit.assert_called_once()



    def test_duplicate_handling(self, mocked_managers): # Renamed from test_process_directory_handles_mixed_files
        """Testa o processamento com um arquivo único e um duplicado."""

        # --- Setup Scenario ---
        orchestrator = DataIngestionOrchestrator()
        unique_hash = "unique_hash_abc"
        duplicate_hash = "duplicate_hash_123" # This will be pre-loaded

        # Actual filenames created by generate_test_pdfs.py
        unique_filename = "test_document.pdf"
        duplicate_filename = "duplicate_document.pdf"

        # Pre-load the hash for the file we want to be treated as duplicate
        orchestrator.document_hashes = {duplicate_filename: duplicate_hash}

        # --- Mock Document Processor Conditionally ---
        mock_processor_instance = MagicMock()
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
        mock_processor_instance.process_document = mock_process_document_conditional
        orchestrator.document_processor = mock_processor_instance

        # --- Process Directory ---
        orchestrator.process_directory(self.test_pdfs_dir) # Contains the two actual files

        # --- Assertions ---
        mocked_managers['get_connection'].assert_called_once()

        # Rollback should happen exactly once (for duplicate_document.pdf)
        self.mock_conn.rollback.assert_called_once()

        # Commit should happen exactly once (for test_document.pdf)
        self.mock_conn.commit.assert_called_once() # Expecting 1 commit now

        # Check inserts happened for the unique file
        # We can be more specific about calls with the unique file's expected ID if needed
        mocked_managers['insert_doc'].assert_called_once()
        mocked_managers['insert_chunk'].assert_called() # >= 1 chunk
        mocked_managers['insert_embedding'].assert_called() # >= 1 embedding
        mocked_managers['add_embeddings'].assert_called_once() # Called once for the unique file

        # Check final hashes (ensure unique one was added, duplicate remains)
        assert len(orchestrator.document_hashes) == 2 # Initial duplicate + 1 unique
        assert orchestrator.document_hashes[unique_filename] == unique_hash
        assert orchestrator.document_hashes[duplicate_filename] == duplicate_hash # Original duplicate entry remains
        
    def test_invalid_directory(self):
        """Testa o processamento de um diretório inválido."""
        orchestrator = DataIngestionOrchestrator()
        
        # Testa diretório inexistente
        with pytest.raises(FileNotFoundError):
            orchestrator.process_directory("nonexistent_directory")
            
        # Testa caminho que não é um diretório
        test_file = os.path.join(self.test_pdfs_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
            
        with pytest.raises(NotADirectoryError):
            orchestrator.process_directory(test_file)
            
        # Limpa o arquivo de teste
        os.remove(test_file)

    def test_empty_directory(self):
        """Testa o processamento de um diretório vazio."""
        orchestrator = DataIngestionOrchestrator()
        
        # Testa listagem de PDFs em diretório vazio
        with pytest.raises(ValueError) as exc_info:
            orchestrator.list_pdf_files(self.empty_dir)
        assert "Nenhum arquivo PDF encontrado" in str(exc_info.value)
        
        # Testa processamento de diretório vazio
        with pytest.raises(ValueError) as exc_info:
            orchestrator.process_directory(self.empty_dir)
        assert "Nenhum arquivo PDF encontrado" in str(exc_info.value) 