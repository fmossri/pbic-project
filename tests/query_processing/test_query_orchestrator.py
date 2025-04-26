import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.query_processing.query_orchestrator import QueryOrchestrator

class TestQueryOrchestrator:
    """Suite de testes para a classe QueryOrchestrator."""
    
    @pytest.fixture
    def query_orchestrator(self, mocker):
        """Fixture que fornece uma instância do QueryOrchestrator com componentes mockados."""
        # Mock heavy components before initialization
        mocker.patch('src.utils.text_normalizer.TextNormalizer.__init__', return_value=None)
        mocker.patch('src.utils.embedding_generator.EmbeddingGenerator.__init__', return_value=None)
        mocker.patch('src.utils.embedding_generator.EmbeddingGenerator.embedding_dimension', 384, create=True)
        mocker.patch('src.utils.embedding_generator.EmbeddingGenerator.model_name', "mock-model", create=True)
        mocker.patch('src.utils.faiss_manager.FaissManager.__init__', return_value=None)
        mocker.patch('src.utils.faiss_manager.FaissManager.index_path', "mock/path.faiss", create=True)
        mocker.patch('src.utils.faiss_manager.FaissManager.dimension', 384, create=True)
        mocker.patch('src.utils.faiss_manager.FaissManager.index', MagicMock(), create=True)
        mocker.patch('src.utils.sqlite_manager.SQLiteManager.__init__', return_value=None)
        mocker.patch('src.utils.sqlite_manager.SQLiteManager.db_path', "mock/path.db", create=True)
        mocker.patch('src.utils.sqlite_manager.SQLiteManager.control_db_path', "mock/control.db", create=True)
        mocker.patch('src.query_processing.hugging_face_manager.HuggingFaceManager.__init__', return_value=None)
        mocker.patch('src.query_processing.hugging_face_manager.HuggingFaceManager.client', MagicMock(), create=True)
        
        orchestrator = QueryOrchestrator()
        
        # Mock logger to avoid actual logging
        orchestrator.logger = MagicMock()
        
        return orchestrator
    
    def test_initialization(self, mocker):
        """Testa a inicialização do QueryOrchestrator."""
        # Mock initializations to make the test faster
        mocker.patch('src.utils.text_normalizer.TextNormalizer.__init__', return_value=None)
        mocker.patch('src.utils.embedding_generator.EmbeddingGenerator.__init__', return_value=None)
        mocker.patch('src.utils.faiss_manager.FaissManager.__init__', return_value=None)
        mocker.patch('src.utils.sqlite_manager.SQLiteManager.__init__', return_value=None) 
        mocker.patch('src.query_processing.hugging_face_manager.HuggingFaceManager.__init__', return_value=None)
        
        orchestrator = QueryOrchestrator()
        assert orchestrator is not None
        assert orchestrator.text_normalizer is not None
        assert orchestrator.embedding_generator is not None
        assert orchestrator.faiss_manager is not None
        assert orchestrator.sqlite_manager is not None
        assert orchestrator.hugging_face_manager is not None
    
    def test_empty_query(self, query_orchestrator):
        """Testa o comportamento com uma query vazia."""
        with pytest.raises(ValueError) as exc_info:
            query_orchestrator._process_query("")
        assert "Query vazia ou inválida" in str(exc_info.value)
    
    def test_process_query(self, query_orchestrator, mocker):
        """Testa o processamento de uma query válida."""
        # Mock do TextNormalizer
        mock_normalize = mocker.patch.object(
            query_orchestrator.text_normalizer,
            'normalize',
            return_value="query normalizada"
        )
        
        # Mock do EmbeddingGenerator
        mock_embeddings = np.array([[0.1] * 384], dtype=np.float32)
        mock_generate_embeddings = mocker.patch.object(
            query_orchestrator.embedding_generator,
            'generate_embeddings',
            return_value=mock_embeddings
        )
        
        result = query_orchestrator._process_query("teste de query")
        
        # Verifica se os métodos foram chamados corretamente
        mock_normalize.assert_called_once_with("teste de query")
        mock_generate_embeddings.assert_called_once_with("query normalizada")
        
        # Verifica se o resultado é o esperado
        assert np.array_equal(result, mock_embeddings)
    
    def test_embedding_error(self, query_orchestrator, mocker):
        """Testa o comportamento quando o embedding não pode ser gerado."""
        # Mock do TextNormalizer
        mock_normalize = mocker.patch.object(
            query_orchestrator.text_normalizer,
            'normalize',
            return_value="query normalizada"
        )
        
        # Mock do EmbeddingGenerator para retornar um array vazio
        mock_generate_embeddings = mocker.patch.object(
            query_orchestrator.embedding_generator,
            'generate_embeddings',
            return_value=np.array([])
        )
        
        with pytest.raises(ValueError) as exc_info:
            query_orchestrator._process_query("teste de query")
        
        # Verify mocks were called
        mock_normalize.assert_called_once_with("teste de query")
        mock_generate_embeddings.assert_called_once_with("query normalizada")
        assert "Erro ao gerar o embedding da query" in str(exc_info.value)
    
    def test_retrieve_documents(self, query_orchestrator, mocker):
        """Testa a recuperação de documentos de múltiplos domínios."""
        # Mock embedding
        mock_embedding = np.array([[0.1] * 384], dtype=np.float32)
        
        # Create mock domain
        mock_domain = MagicMock(name="domain1", db_path="path/to/domain1.db", vector_store_path="path/to/domain1.faiss")
        
        # Mock FaissManager.search_faiss_index
        mock_search = mocker.patch.object(
            query_orchestrator.faiss_manager,
            'search_faiss_index',
            return_value=(np.array([[0.8, 0.7, 0.6]]), np.array([[1, 2, 3]]))
        )
        
        # Mock SQLiteManager.get_connection
        mock_conn = MagicMock()
        mock_get_connection = mocker.patch.object(
            query_orchestrator.sqlite_manager,
            'get_connection'
        )
        mock_get_connection.return_value.__enter__.return_value = mock_conn
        
        # Mock SQLiteManager.get_chunks_content
        mock_chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        mock_get_chunks = mocker.patch.object(
            query_orchestrator.sqlite_manager,
            'get_chunks_content',
            return_value=mock_chunks
        )
        
        # Call the method
        result = query_orchestrator._retrieve_documents(mock_embedding, mock_domain)
        
        # Verify calls
        mock_search.assert_called_once_with(mock_embedding, mock_domain.vector_store_path)
        mock_get_chunks.assert_called_once_with(mock_conn, [1, 2, 3])
        
        # Verify result
        assert result == mock_chunks
    
    def test_retrieve_documents_empty_embedding(self, query_orchestrator):
        """Testa a recuperação de documentos com embedding vazio."""
        # Create mock domain
        mock_domain = MagicMock(name="domain1", db_path="path/to/domain1.db", vector_store_path="path/to/domain1.faiss")
        
        with pytest.raises(ValueError) as exc_info:
            query_orchestrator._retrieve_documents(None, mock_domain)
        assert "Vetor de embedding vazio ou inválido" in str(exc_info.value)
    
    def test_prepare_context_prompt(self, query_orchestrator):
        """Testa a preparação do prompt de contexto."""
        query = "Qual é a capital do Brasil?"
        chunks = ["O Brasil é um país na América do Sul.", "Brasília é a capital federal do Brasil."]
        
        prompt = query_orchestrator._prepare_context_prompt(query, chunks)
        
        # Verifica se o prompt contém a query e os chunks
        assert "Qual é a capital do Brasil?" in prompt
        assert "O Brasil é um país na América do Sul." in prompt
        assert "Brasília é a capital federal do Brasil." in prompt
    
    def test_query_llm(self, query_orchestrator, mocker):
        """Testa o fluxo completo de processamento de query."""
        # Mock para _process_query
        mock_embedding = np.array([[0.1] * 384], dtype=np.float32)
        mock_process_query = mocker.patch.object(
            query_orchestrator,
            '_process_query',
            return_value=mock_embedding
        )
        
        # Create a mock domain
        mock_domain = MagicMock(name="test_domain", db_path="path/to/test_domain.db", vector_store_path="path/to/test_domain.faiss")
        
        # Mock _select_domains to return our test domain
        mock_select_domains = mocker.patch.object(
            query_orchestrator,
            '_select_domains',
            return_value=[mock_domain]
        )
        
        # Mock para _retrieve_documents
        mock_chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        mock_retrieve_docs = mocker.patch.object(
            query_orchestrator,
            '_retrieve_documents',
            return_value=mock_chunks
        )
        
        # Mock para _prepare_context_prompt
        mock_prompt = "Prompt preparado com contexto e query"
        mock_prepare_prompt = mocker.patch.object(
            query_orchestrator,
            '_prepare_context_prompt',
            return_value=mock_prompt
        )
        
        # Mock para HuggingFaceManager.generate_answer
        mock_answer = "Esta é a resposta gerada pelo modelo."
        mock_generate_answer = mocker.patch.object(
            query_orchestrator.hugging_face_manager,
            'generate_answer',
            return_value=mock_answer
        )
        
        # Call the method
        test_query = "Teste de query"
        result = query_orchestrator.query_llm(test_query)
        
        # Verify calls
        mock_select_domains.assert_called_once_with(test_query)
        mock_process_query.assert_called_once_with(test_query)
        mock_retrieve_docs.assert_called_once_with(mock_embedding, mock_domain)
        mock_prepare_prompt.assert_called_once_with(test_query, mock_chunks)
        mock_generate_answer.assert_called_once_with(test_query, mock_prompt)
        
        # Verify result is a dictionary with expected content
        assert isinstance(result, dict)
        assert result["answer"] == mock_answer
        assert result["question"] == test_query
        assert result["success"] == True
    
    def test_query_llm_empty_query(self, query_orchestrator):
        """Testa o comportamento com uma query vazia no fluxo completo."""
        with pytest.raises(ValueError) as exc_info:
            query_orchestrator.query_llm("")
        assert "Query vazia ou inválida" in str(exc_info.value)
    
    def test_query_llm_error_handling(self, query_orchestrator, mocker):
        """Testa o tratamento de erros no fluxo completo."""
        # Mock _select_domains to raise an exception
        error_message = "Erro de teste"
        mocker.patch.object(
            query_orchestrator,
            '_select_domains',
            side_effect=Exception(error_message)
        )
        
        # Test that an exception is propagated
        with pytest.raises(Exception):
            query_orchestrator.query_llm("Teste de query")
            
        # Verify that the metrics data captures the failure
        assert query_orchestrator.metrics_data["success"] == False 