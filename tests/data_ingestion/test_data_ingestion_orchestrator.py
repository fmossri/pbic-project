import os
import pytest
import shutil
import numpy as np

from unittest.mock import patch, MagicMock
from .test_docs.generate_test_pdfs import create_test_pdf

from src.data_ingestion import DataIngestionOrchestrator

from src.config.models import AppConfig, SystemConfig, IngestionConfig, EmbeddingConfig, VectorStoreConfig, QueryConfig, TextNormalizerConfig, LLMConfig
from src.models import DocumentFile, Domain
from langchain.schema import Document

class TestDataIngestionOrchestrator:
    """Suite de testes para a classe DataIngestionOrchestrator."""

    test_docs_dir = os.path.join(os.path.dirname(__file__), "test_docs")
    test_pdfs_dir = os.path.join(test_docs_dir, "test_pdfs")
    empty_dir = os.path.join(test_docs_dir, "empty_dir")
    test_storage_base = os.path.join("tests", "test_storage")
    test_domain_dir = os.path.join(test_storage_base, "domains", "test_domain")
    indices_dir = os.path.join(test_domain_dir, "vector_store")
    
    test_control_db_dir = os.path.join(test_storage_base, "control")
    test_control_db_filename = "test_control.db"
    test_control_db_path = os.path.join(test_control_db_dir, test_control_db_filename)

    test_domain_db_path = os.path.join(test_domain_dir, "test_domain.db")
    test_domain_vector_store_path = os.path.join(indices_dir, "test_domain.faiss")

    @pytest.fixture(scope="class", autouse=True)
    def setup_class(self, request):
        """Setup that runs once for the entire test class"""
        # Create directórios apenas uma vez para todos os testes
        os.makedirs(self.test_pdfs_dir, exist_ok=True)
        os.makedirs(self.empty_dir, exist_ok=True)
        os.makedirs(self.indices_dir, exist_ok=True)
        os.makedirs(self.test_control_db_dir, exist_ok=True)

        # Cria PDFs de teste apenas uma vez
        create_test_pdf()
      
        yield
        
        patch.stopall()
        
        # Limpa após todos os testes terem sido completados
        if os.path.exists(self.test_pdfs_dir):
            shutil.rmtree(self.test_pdfs_dir)
        
        if os.path.exists(self.empty_dir):
            shutil.rmtree(self.empty_dir)
            
        # Limpa o diretório de armazenamento de teste inteiro
        if os.path.exists(self.test_storage_base):
             shutil.rmtree(self.test_storage_base)

    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        """Setup for each test"""
        # Mocka a conexão do banco de dados para cada teste
        self.mock_conn = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_conn.cursor.return_value = self.mock_cursor
        self.mock_cursor.lastrowid = 1
        
        # Mocka o logger para evitar problemas de serialização com MagicMock
        self.mock_logger = MagicMock()
        # Torna os métodos do logger encadeáveis
        self.mock_logger.info.return_value = self.mock_logger
        self.mock_logger.debug.return_value = self.mock_logger
        self.mock_logger.warning.return_value = self.mock_logger
        self.mock_logger.error.return_value = self.mock_logger
        
        # Patcha o get_logger para retornar nosso mock
        self.logger_patcher = mocker.patch('src.utils.logger.get_logger', return_value=self.mock_logger)
        
        # Patcha SentenceTransformer para evitar carregamento do modelo real
        self.model_patcher = mocker.patch('src.utils.embedding_generator.SentenceTransformer')
        mock_model = MagicMock()
 
        mock_model.get_sentence_embedding_dimension.return_value = 384
        self.model_patcher.return_value = mock_model
        
        # Mock operações faiss
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
                storage_base_path=self.test_storage_base,
                control_db_filename=self.test_control_db_filename
            ),
            ingestion=IngestionConfig(chunk_size=500, chunk_overlap=50),
            embedding=EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            vector_store=VectorStoreConfig(),
            query=QueryConfig(),
            llm=LLMConfig(),
            text_normalizer=TextNormalizerConfig()
        )

    @pytest.fixture
    def domain_fixture(self, mocker, request):
        """Fixture para criar um domínio de teste com dimensão de embedding opcional."""
        initial_dimension = getattr(request, "param", 384)

        domain = Domain(
            id=1,
            name="test_domain",
            description="Test domain description",
            keywords="test,domain,keywords",
            total_documents=0,
            db_path=self.test_domain_db_path, 
            vector_store_path=self.test_domain_vector_store_path, 
            embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
            embeddings_dimension=initial_dimension,
            faiss_index_type="IndexFlatL2",
        )

        mock_get_domain = mocker.patch('src.utils.sqlite_manager.SQLiteManager.get_domain', return_value=[domain])

        mock_update_domain = mocker.patch('src.utils.sqlite_manager.SQLiteManager.update_domain')

        yield domain, mock_get_domain, mock_update_domain
    
    @pytest.fixture
    def configured_orchestrator(self, test_app_config):
        """Cria uma instância do orquestrador inicializada com a configuração de teste."""
        orch = DataIngestionOrchestrator(config=test_app_config)
        return orch
    
    @pytest.fixture
    def mocked_managers(self, mocker):
        """Fixture para mockar dependências comuns (SQLiteManager, FaissManager)."""
        mocks = {}

        mock_context_manager = MagicMock(name="ctx_mgr_mock")
        mock_context_manager.__enter__.return_value = self.mock_conn
        mock_context_manager.__exit__.return_value = None
        
        self.mock_control_conn = MagicMock()
        self.mock_control_cursor = MagicMock()
        self.mock_control_conn.cursor.return_value = self.mock_control_cursor
        self.mock_control_cursor.lastrowid = 1
        
        mock_control_context_manager = MagicMock(name="control_ctx_mgr_mock")
        mock_control_context_manager.__enter__.return_value = self.mock_control_conn
        mock_control_context_manager.__exit__.return_value = None
        
        def mock_get_connection(control=False, db_path=None):
            if control:
                return mock_control_context_manager
            return mock_context_manager
            
        mocks['get_connection'] = mocker.patch(
            'src.utils.sqlite_manager.SQLiteManager.get_connection',
            side_effect=mock_get_connection
        )

        mocks['sqlite_begin'] = mocker.patch('src.utils.sqlite_manager.SQLiteManager.begin')
        mocks['insert_doc'] = mocker.patch(
            'src.utils.sqlite_manager.SQLiteManager.insert_document_file', return_value=1
        )
        mocks['insert_chunk'] = mocker.patch(
            'src.utils.sqlite_manager.SQLiteManager.insert_chunks', return_value=[1]
        )

        mock_embeddings = np.array([[0.1] * 384], dtype=np.float32)
        mocks['generate_embeddings'] = mocker.patch(
            'src.utils.embedding_generator.EmbeddingGenerator.generate_embeddings',
            return_value=mock_embeddings
        )

        mocks['add_embeddings'] = mocker.patch(
            'src.utils.faiss_manager.FaissManager.add_embeddings',
        )

        mocks['update_domain'] = mocker.patch(
            'src.utils.sqlite_manager.SQLiteManager.update_domain'
        )

        return mocks
            
    def test_initialization(self, configured_orchestrator):
        """Testa a inicialização do orquestrador com configuração."""
        orch = configured_orchestrator
        assert orch is not None
        assert orch.config is not None
        assert orch.document_processor is not None
        assert orch.text_chunker is not None
        assert orch.text_normalizer is not None
        assert orch.embedding_generator is not None
        assert orch.sqlite_manager is not None
        assert orch.faiss_manager is not None
        assert orch.text_chunker.config.chunk_size == 500
        assert orch.embedding_generator.config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert orch.sqlite_manager.config.storage_base_path == self.test_storage_base
        
    def test_list_pdf_files(self, configured_orchestrator):
        """Testa a listagem de arquivos PDF usando os.scandir."""
        orch = configured_orchestrator
        pdf_files = orch._list_pdf_files(self.test_pdfs_dir)
        assert isinstance(pdf_files, list)
        assert len(pdf_files) > 0
        
        for file in pdf_files:
            assert isinstance(file, DocumentFile)
            assert file.name.endswith('.pdf')
        
        with os.scandir(self.test_pdfs_dir) as entries:
            actual_pdfs = {entry.name for entry in entries 
                         if entry.is_file() and entry.name.endswith('.pdf')}
            
        filenames = [file.name for file in pdf_files]
        assert set(filenames) == actual_pdfs
        
    def test_is_duplicate(self, configured_orchestrator):
        """Testa o método de detecção de duplicatas."""
        orch = configured_orchestrator
        mock_conn = MagicMock()
        
        mock_cursor = MagicMock()
        mock_conn.execute.return_value = mock_cursor
        
        orch.document_hashes = {}
        mock_cursor.fetchone.return_value = None
        assert orch._is_duplicate("some_hash", mock_conn) is False
        mock_conn.execute.assert_called_with("SELECT * FROM document_files WHERE hash = ?", ("some_hash",))

        test_hash = "existing_hash_123"
        orch.document_hashes = {"file1.pdf": test_hash, "file2.pdf": "another_hash"}
        assert orch._is_duplicate(test_hash, mock_conn) is True
        orch.document_hashes = {}
        mock_cursor.fetchone.return_value = (1, "db_match_hash")
        assert orch._is_duplicate("db_match_hash", mock_conn) is True
        mock_conn.execute.assert_called_with("SELECT * FROM document_files WHERE hash = ?", ("db_match_hash",))

        mock_cursor.fetchone.return_value = None
        assert orch._is_duplicate("new_hash_456", mock_conn) is False

        assert orch._is_duplicate(None, mock_conn) is False
        assert orch._is_duplicate("", mock_conn) is False

    def test_process_directory_with_valid_pdfs(
        self, configured_orchestrator, mocked_managers, mocker, domain_fixture
    ):
        """Testa o processamento de um diretório com PDFs válidos (patching methods)."""
        orch = configured_orchestrator
        initial_domain, mock_get_domain, mock_update_domain = domain_fixture

        self.mock_conn.reset_mock()
        self.mock_control_conn.reset_mock()
        mock_update_domain = mocker.patch('src.utils.sqlite_manager.SQLiteManager.update_domain')

        def mock_process_document(file):
            file.hash = f"test_hash_{file.name}"
            file.pages = [Document(page_content=f"Content for {file.name}", metadata={"page": 1})]
            file.total_pages = 1
            return file
        mocker.patch.object(orch.document_processor, 'process_document', side_effect=mock_process_document)
        
        calls = 0
        def mock_is_duplicate_func(document_hash, conn):
            nonlocal calls
            calls += 1
            return calls > 1
            
        mock_is_duplicate = mocker.patch.object(
            orch, 
            '_is_duplicate', 
            side_effect=mock_is_duplicate_func
        )

        def mock_find_original(hash_value, conn):
            if calls > 1:
                return DocumentFile(id=1, hash="test_hash_test_document.pdf", name="test_document.pdf", path="/fake/path", total_pages=1)
            return None
        mocker.patch.object(orch, '_find_original_document', side_effect=mock_find_original)

        orch.process_directory(self.test_pdfs_dir, domain_name="test_domain")

        assert mocked_managers['get_connection'].call_count >= 3
        
        control_calls = [call for call in mocked_managers['get_connection'].call_args_list 
                        if call.kwargs.get('control') == True]
        assert len(control_calls) >= 2, "get_connection deveria ser chamado com control=True duas vezes"
        
        assert mocked_managers['sqlite_begin'].call_count >= 2
        
        assert mock_is_duplicate.call_count == 2, f"_is_duplicate deveria ser chamado duas vezes, uma para cada PDF"
        
        mocked_managers['insert_doc'].assert_called_once()
        mocked_managers['insert_chunk'].assert_called_once()
        mocked_managers['add_embeddings'].assert_called_once()
        
        assert self.mock_conn.commit.call_count == 1, "Domain DB deveria ser commitado uma vez"
        
        assert self.mock_control_conn.commit.call_count == 2, f"Control DB deveria ser commitado duas vezes (get, update count), recebeu {self.mock_control_conn.commit.call_count}"
        
        self.mock_conn.rollback.assert_called_once()
        
        mock_update_domain.assert_called_once()
        
        call_args_docs_update = mock_update_domain.call_args
        assert call_args_docs_update.args[0].id == initial_domain.id, "ID do objeto dominio incorreto para atualizacao de total de documentos"
        assert call_args_docs_update.args[1] == self.mock_control_conn, "conexao incorreta para atualizacao de total de documentos"
        assert call_args_docs_update.args[2] == {"total_documents": 1}, "update dict incorreto para atualizacao de total de documentos"

    def test_duplicate_handling(self, configured_orchestrator, mocked_managers, mocker, domain_fixture):
        """Testa o tratamento de arquivos duplicados."""
        orch = configured_orchestrator

        self.mock_conn.reset_mock()
        
        unique_hash = "unique_hash_abc"
        duplicate_hash = "duplicate_hash_123"
 
        unique_filename = "test_document.pdf"
        duplicate_filename = "duplicate_document.pdf"
 
        original_files = {
            unique_filename: unique_hash,
            duplicate_filename: duplicate_hash,
        }
        def mock_process_document_conditional(file):
            file.hash = original_files.get(file.name, "unexpected_file_hash")
            file.pages = [Document(page_content="Content", metadata={"page": 1})]
            file.total_pages = 1
            return file
        mocker.patch.object(orch.document_processor, 'process_document', side_effect=mock_process_document_conditional)

        def mock_is_duplicate_conditional(hash_value, conn):
            return hash_value == duplicate_hash
 
        mocker.patch.object(
            orch, 
            '_is_duplicate', 
            side_effect=mock_is_duplicate_conditional
        )
 
        def mock_find_original_document(hash_value, conn):
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
 
        result = orch.process_directory(self.test_pdfs_dir, domain_name="test_domain")
 
        assert 'duplicate_files' in result
        assert result['duplicate_files'] > 0
        assert self.mock_conn.rollback.call_count >= 1
 
        assert unique_filename in result, "Missing metrics for unique file"
        assert duplicate_filename in result, "Missing metrics for duplicate file"
 
        assert result[duplicate_filename]['is_duplicate'] == True
        assert result[unique_filename]['is_duplicate'] == False
 
        mocked_managers['insert_doc'].assert_called_once()
        
        assert self.mock_conn.commit.call_count >= 1

    def test_invalid_directory(self, configured_orchestrator):
        """Testa o tratamento de diretórios inválidos."""
        orch = configured_orchestrator
        with pytest.raises(FileNotFoundError):
            orch.process_directory("/nonexistent/directory", domain_name="test_domain")
    
        test_file = os.path.join(self.test_pdfs_dir, "test_document.pdf")
        with open(test_file, 'w') as f:
            f.write("dummy content")
            
        with pytest.raises(NotADirectoryError):
            orch.process_directory(test_file, domain_name="test_domain")

    def test_empty_directory(self, configured_orchestrator, domain_fixture):
        """Testa o tratamento de diretórios vazios."""
        orch = configured_orchestrator
        
        with pytest.raises(ValueError, match="Nenhum arquivo PDF encontrado"):
            orch.process_directory(self.empty_dir, domain_name="test_domain")

    @pytest.mark.parametrize("domain_fixture", [0], indirect=True)
    def test_process_directory_updates_embedding_dimension(
        self, configured_orchestrator, mocked_managers, mocker, domain_fixture
    ):
        """
        Testa se process_directory atualiza embeddings_dimension se for 0.
        """
        orch = configured_orchestrator

        initial_domain, mock_get_domain, mock_update_domain = domain_fixture
        expected_final_dimension = orch.embedding_generator.embedding_dimension

        self.mock_control_conn.reset_mock()
        mock_update_domain.reset_mock()

        def mock_process_document(file):
            file.hash = "unique_hash_for_dim_test"
            file.pages = [Document(page_content="Test content dim update", metadata={"page": 1})]
            file.total_pages = 1
            return file
        mocker.patch.object(orch.document_processor, 'process_document', side_effect=mock_process_document)

        mocker.patch.object(orch,'_is_duplicate', return_value=False)

        orch.process_directory(self.test_pdfs_dir, domain_name="test_domain")

        mock_get_domain.assert_called_once_with(self.mock_control_conn, "test_domain")

        mock_update_domain.assert_called()

        assert len(mock_update_domain.call_args_list) >= 1, "update_domain deveria ter sido chamado pelo menos uma vez"
        
        first_call_args = mock_update_domain.call_args_list[0].args
        
        assert first_call_args[0].id == initial_domain.id, "ID do objeto dominio incorreto para atualizacao de dimensao de embeddings"
        assert first_call_args[1] == self.mock_control_conn, "conexao incorreta para atualizacao de dimensao de embeddings"

        expected_update_dict = {
            "embeddings_dimension": expected_final_dimension,
        }

        assert first_call_args[2] == expected_update_dict, "update dict incorreto para atualizacao de dimensao de embeddings" 
        assert len(mock_update_domain.call_args_list) >= 2, "chamada de update_domain para total de documentos esperada"
        
        second_call_args = mock_update_domain.call_args_list[1].args
        
        assert second_call_args[0].id == initial_domain.id, "ID do objeto dominio incorreto para atualizacao de total de documentos"
        assert second_call_args[1] == self.mock_control_conn, "conexao incorreta para atualizacao de total de documentos"
        assert second_call_args[2] == {"total_documents": len(os.listdir(self.test_pdfs_dir))}, "update dict incorreto para atualizacao de total de documentos" # Should process all files

    def test_update_config_no_change(self, configured_orchestrator, test_app_config, mocker):
        """Teste do update_config: Sem alterações."""
        orchestrator = configured_orchestrator

        mock_tc_update = mocker.patch.object(orchestrator.text_chunker, 'update_config', autospec=True)
        mock_eg_update = mocker.patch.object(orchestrator.embedding_generator, 'update_config', autospec=True)
        mock_fm_update = mocker.patch.object(orchestrator.faiss_manager, 'update_config', autospec=True)
        mock_tn_update = mocker.patch.object(orchestrator.text_normalizer, 'update_config', autospec=True)
        mock_sm_update = mocker.patch.object(orchestrator.sqlite_manager, 'update_config', autospec=True)

        initial_config_ref = orchestrator.config
        new_config = test_app_config.model_copy()

        orchestrator.update_config(new_config)

        assert orchestrator.config is initial_config_ref

        mock_tc_update.assert_not_called()
        mock_eg_update.assert_not_called()
        mock_fm_update.assert_not_called()
        mock_tn_update.assert_not_called()
        mock_sm_update.assert_not_called()

    def test_update_config_single_change(self, configured_orchestrator, test_app_config, mocker):
        """Teste do update_config: Uma alteração (seção 'ingestion')."""
        orchestrator = configured_orchestrator

        mock_tc_update = mocker.patch.object(orchestrator.text_chunker, 'update_config', autospec=True)
        mock_eg_update = mocker.patch.object(orchestrator.embedding_generator, 'update_config', autospec=True)
        mock_fm_update = mocker.patch.object(orchestrator.faiss_manager, 'update_config', autospec=True)
        mock_tn_update = mocker.patch.object(orchestrator.text_normalizer, 'update_config', autospec=True)
        mock_sm_update = mocker.patch.object(orchestrator.sqlite_manager, 'update_config', autospec=True)

        new_ingestion_config = IngestionConfig(chunk_size=9999, chunk_overlap=999) # Different values
        new_config = test_app_config.model_copy(update={"ingestion": new_ingestion_config})
        initial_config_ref = orchestrator.config

        orchestrator.update_config(new_config)

        # Verifica se o método update_config foi chamado com a nova config
        mock_tc_update.assert_called_once_with(new_ingestion_config)
        mock_eg_update.assert_not_called()
        mock_fm_update.assert_not_called()
        mock_tn_update.assert_not_called()
        mock_sm_update.assert_not_called()
        # Verifica se o objeto config foi atualizado
        assert orchestrator.config is new_config
        assert orchestrator.config != initial_config_ref

    def test_update_config_multiple_changes(self, configured_orchestrator, test_app_config, mocker):
        """Teste do update_config: Múltiplas alterações (seções 'system' e 'embedding')."""
        orchestrator = configured_orchestrator

        mock_tc_update = mocker.patch.object(orchestrator.text_chunker, 'update_config', autospec=True)
        mock_eg_update = mocker.patch.object(orchestrator.embedding_generator, 'update_config', autospec=True)
        mock_fm_update = mocker.patch.object(orchestrator.faiss_manager, 'update_config', autospec=True)
        mock_tn_update = mocker.patch.object(orchestrator.text_normalizer, 'update_config', autospec=True)
        mock_sm_update = mocker.patch.object(orchestrator.sqlite_manager, 'update_config', autospec=True)

        new_system_config = SystemConfig(storage_base_path="/multi/change/simplified")
        new_embedding_config = EmbeddingConfig(model_name="sentence-transformers/all-mpnet-base-v2") # Different model
        new_config = test_app_config.model_copy(update={
            "system": new_system_config,
            "embedding": new_embedding_config
        })
        initial_config_ref = orchestrator.config

        orchestrator.update_config(new_config)

        # Verifica que o método update_config foi chamado com a nova config
        mock_sm_update.assert_called_once_with(new_system_config)
        mock_eg_update.assert_called_once_with(new_embedding_config)
        mock_tc_update.assert_not_called()
        mock_fm_update.assert_not_called()
        mock_tn_update.assert_not_called()
        # Assert config object reference was updated
        assert orchestrator.config is new_config
        assert orchestrator.config != initial_config_ref

