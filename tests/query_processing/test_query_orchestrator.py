import pytest
import numpy as np
from unittest.mock import MagicMock

# Import necessary components and models
from src.query_processing.query_orchestrator import QueryOrchestrator
from src.config.models import AppConfig, SystemConfig, IngestionConfig, EmbeddingConfig, VectorStoreConfig, QueryConfig, TextNormalizerConfig, LLMConfig
from src.models import Chunk, Domain # Import Chunk and Domain for type hints/mocking

class TestQueryOrchestrator:
    """Suite de testes para a classe QueryOrchestrator."""
    
    @pytest.fixture
    def test_app_config(self) -> AppConfig:
        """Creates a default AppConfig for testing query orchestration."""
        return AppConfig(
            system=SystemConfig(),
            ingestion=IngestionConfig(),
            embedding=EmbeddingConfig(model_name="test-embedding-model"), # Specify a test model name
            vector_store=VectorStoreConfig(),
            query=QueryConfig(retrieval_k=3), # Specify test k
            llm=LLMConfig(model_repo_id="test-llm-model", prompt_template="Context:{context} Question:{query} Answer:"), # Specify test LLM and template
            text_normalizer=TextNormalizerConfig()
        )

    @pytest.fixture
    def orchestrator(self, mocker, test_app_config):
        """Fixture que fornece uma instância do QueryOrchestrator com config e mocks."""
        
        # Mock initializations of dependencies *before* creating orchestrator
        # This prevents external calls (like model downloads) during init
        mock_text_norm_init = mocker.patch('src.utils.text_normalizer.TextNormalizer.__init__', return_value=None)
        mock_embed_init = mocker.patch('src.utils.embedding_generator.EmbeddingGenerator.__init__', return_value=None)
        mock_faiss_init = mocker.patch('src.utils.faiss_manager.FaissManager.__init__', return_value=None)
        mock_sql_init = mocker.patch('src.utils.sqlite_manager.SQLiteManager.__init__', return_value=None)
        mock_hf_init = mocker.patch('src.query_processing.hugging_face_manager.HuggingFaceManager.__init__', return_value=None)
        
        # Create orchestrator with the test config
        orch = QueryOrchestrator(config=test_app_config)
        
        # Assign mock instances to the orchestrator's attributes *after* initialization
        # This allows us to control their behavior in tests
        orch.text_normalizer = MagicMock(spec=orch.text_normalizer)
        orch.embedding_generator = MagicMock(spec=orch.embedding_generator)
        orch.faiss_manager = MagicMock(spec=orch.faiss_manager)
        orch.sqlite_manager = MagicMock(spec=orch.sqlite_manager)
        orch.hugging_face_manager = MagicMock(spec=orch.hugging_face_manager)
        
        # Mock logger to avoid actual logging
        orch.logger = MagicMock()

        # Configure mocks that need specific values from the config or defaults
        orch.embedding_generator.config = test_app_config.embedding
        orch.embedding_generator.embedding_dimension = 384 # Example dimension
        orch.faiss_manager.vector_config = test_app_config.vector_store
        orch.faiss_manager.query_config = test_app_config.query
        orch.hugging_face_manager.config = test_app_config.llm
        
        return orch
    
    def test_initialization(self, orchestrator, test_app_config): # Pass orchestrator fixture
        """Testa a inicialização do QueryOrchestrator."""
        assert orchestrator is not None
        # Check if mocks were assigned (or real instances if not mocked in fixture)
        assert orchestrator.text_normalizer is not None 
        assert orchestrator.embedding_generator is not None
        assert orchestrator.faiss_manager is not None
        assert orchestrator.sqlite_manager is not None
        assert orchestrator.hugging_face_manager is not None
        # Check if config was used (example)
        assert orchestrator.embedding_generator.config.model_name == "test-embedding-model"
        assert orchestrator.hugging_face_manager.config.model_repo_id == "test-llm-model"
    
    def test_empty_query(self, orchestrator): # Pass orchestrator fixture
        """Testa o comportamento com uma query vazia."""
        with pytest.raises(ValueError) as exc_info:
            orchestrator._process_query("") # Use the fixture
        assert "Query vazia ou inválida" in str(exc_info.value)
    
    def test_process_query(self, orchestrator, mocker): # Pass orchestrator fixture
        """Testa o processamento de uma query válida."""
        # Configure mocks on the fixture instance
        orchestrator.text_normalizer.normalize.return_value = "query normalizada"
        mock_embeddings = np.array([[0.1] * 384], dtype=np.float32) 
        orchestrator.embedding_generator.generate_embeddings.return_value = mock_embeddings
        
        result = orchestrator._process_query("teste de query")
        
        # Verifica se os métodos foram chamados corretamente
        orchestrator.text_normalizer.normalize.assert_called_once_with("teste de query")
        orchestrator.embedding_generator.generate_embeddings.assert_called_once_with("query normalizada")
        
        # Verifica se o resultado é o esperado
        assert np.array_equal(result, mock_embeddings)
    
    def test_embedding_error(self, orchestrator, mocker): # Pass orchestrator fixture
        """Testa o comportamento quando o embedding não pode ser gerado."""
        # Configure mocks on the fixture instance
        orchestrator.text_normalizer.normalize.return_value = "query normalizada"
        orchestrator.embedding_generator.generate_embeddings.return_value = np.array([])
        
        with pytest.raises(ValueError) as exc_info:
            orchestrator._process_query("teste de query")
        
        # Verify mocks were called
        orchestrator.text_normalizer.normalize.assert_called_once_with("teste de query")
        orchestrator.embedding_generator.generate_embeddings.assert_called_once_with("query normalizada")
        assert "Erro ao gerar o embedding da query" in str(exc_info.value)
    
    def test_retrieve_documents(self, orchestrator, mocker): # Pass orchestrator fixture
        """Testa a recuperação de documentos."""
        mock_embedding = np.array([[0.1] * 384], dtype=np.float32)
        mock_domain = Domain(
            id=1, name="domain1", description="d", keywords="k", 
            db_path="path/to/domain1.db", vector_store_path="path/to/domain1.faiss",
            embeddings_dimension=384 # Match mock dimension
        )
        
        # Configure mocks on the fixture instance
        orchestrator.faiss_manager.search_faiss_index.return_value = (np.array([[0.8]]), np.array([[101]])) # Return mock distances and IDs
        mock_conn = MagicMock()
        orchestrator.sqlite_manager.get_connection.return_value.__enter__.return_value = mock_conn
        mock_chunks = [Chunk(id=101, document_id=1, page_number=1, chunk_page_index=0, chunk_start_char_position=0, content="Chunk 101")]
        orchestrator.sqlite_manager.get_chunks.return_value = mock_chunks # Mock get_chunks
        
        # Call the method
        result = orchestrator._retrieve_documents(mock_embedding, mock_domain)
        
        # Verify calls
        orchestrator.faiss_manager.search_faiss_index.assert_called_once_with(
            query_embedding=mock_embedding, 
            index_path=mock_domain.vector_store_path,
            dimension=mock_domain.embeddings_dimension,
        )
        # Check get_chunks call (note: arg name is chunk_ids)
        orchestrator.sqlite_manager.get_chunks.assert_called_once_with(mock_conn, [101]) # Check with positional argument
        
        # Verify result
        assert result == mock_chunks
    
    def test_retrieve_documents_empty_embedding(self, orchestrator): # Pass orchestrator fixture
        """Testa a recuperação de documentos com embedding vazio."""
        mock_domain = Domain(id=1, name="d", description="d", keywords="k", db_path="p", vector_store_path="p", embeddings_dimension=1) 
        
        with pytest.raises(ValueError) as exc_info:
            orchestrator._retrieve_documents(None, mock_domain)
        assert "Vetor de embedding vazio ou inválido" in str(exc_info.value)
    
    def test_prepare_context_prompt(self, orchestrator, test_app_config): # Pass orchestrator fixture
        """Testa a preparação do prompt de contexto usando o template."""
        query = "Qual é a capital?"
        # Use Chunk objects
        chunks = [
            Chunk(id=1, document_id=1, page_number=1, chunk_page_index=0, chunk_start_char_position=0, content="Brasília é a capital."),
            Chunk(id=2, document_id=1, page_number=2, chunk_page_index=0, chunk_start_char_position=0, content="Fica no planalto central.")
        ]
        expected_context = "Brasília é a capital.\n\nFica no planalto central."
        
        # Get the template from the config used by the orchestrator
        template = test_app_config.llm.prompt_template 
        expected_prompt = template.format(context=expected_context, query=query)
        
        # Configure the mock hf_manager on the orchestrator to have the config
        orchestrator.hugging_face_manager.config = test_app_config.llm

        prompt = orchestrator._prepare_context_prompt(query, chunks)
        
        assert prompt == expected_prompt
        assert "Qual é a capital?" in prompt
        assert "Brasília é a capital." in prompt
        assert "Fica no planalto central." in prompt
        
    def test_query_llm(self, orchestrator, mocker, test_app_config): # Pass orchestrator fixture
        """Testa o fluxo completo de processamento de query."""
        test_query = "Teste de query"
        
        # Mock internal method calls on the orchestrator instance
        mock_embedding = np.array([[0.1] * 384], dtype=np.float32)
        mocker.patch.object(orchestrator, '_process_query', return_value=mock_embedding)
        
        mock_domain = Domain(
             id=1, name="test_domain", description="d", keywords="k",
             db_path="p", vector_store_path="p", embeddings_dimension=384
        )
        mocker.patch.object(orchestrator, '_select_domains', return_value=[mock_domain])
        
        mock_chunks = [Chunk(id=1, document_id=1, page_number=1, chunk_page_index=0, chunk_start_char_position=0, content="Chunk 1")]
        mocker.patch.object(orchestrator, '_retrieve_documents', return_value=mock_chunks)
        
        expected_context = "Chunk 1"
        expected_prompt = test_app_config.llm.prompt_template.format(context=expected_context, query=test_query)
        # We don't need to mock _prepare_context_prompt if we trust its implementation based on the previous test
        # mocker.patch.object(orchestrator, '_prepare_context_prompt', return_value=expected_prompt)
        
        # Mock the final call to the HF manager
        mock_answer = "Esta é a resposta gerada pelo modelo."
        orchestrator.hugging_face_manager.generate_answer.return_value = mock_answer
        
        # Call the method
        result = orchestrator.query_llm(test_query)
        
        # Verify internal calls (mocks on orchestrator object)
        orchestrator._select_domains.assert_called_once_with(test_query, None)
        orchestrator._process_query.assert_called_once_with(test_query)
        orchestrator._retrieve_documents.assert_called_once_with(mock_embedding, mock_domain)
        # orchestrator._prepare_context_prompt.assert_called_once_with(test_query, mock_chunks)
        
        # Verify the call to the external dependency
        orchestrator.hugging_face_manager.generate_answer.assert_called_once_with(test_query, expected_prompt)
        
        # Verify result is a dictionary with expected content
        assert isinstance(result, dict)
        assert result["answer"] == mock_answer
        assert result["question"] == test_query
        assert result["success"] == True
    
    def test_query_llm_empty_query(self, orchestrator): # Pass orchestrator fixture
        """Testa o comportamento com uma query vazia no fluxo completo."""
        with pytest.raises(ValueError) as exc_info:
            orchestrator.query_llm("")
        assert "Query vazia ou inválida" in str(exc_info.value)
    
    def test_query_llm_error_handling(self, orchestrator, mocker): # Pass orchestrator fixture
        """Testa o tratamento de erros no fluxo completo."""
        test_query = "Teste de query"
        error_message = "Erro de teste"
        
        # Mock an internal method to raise an exception
        mocker.patch.object(orchestrator, '_select_domains', side_effect=Exception(error_message))
        
        # Test that the exception is propagated
        with pytest.raises(Exception) as exc_info:
            orchestrator.query_llm(test_query)
            
        # Check if the raised exception contains the original message
        assert error_message in str(exc_info.value)
            
        # Verify that the metrics data captures the failure
        assert orchestrator.metrics_data["success"] == False 