import pytest
import numpy as np
from unittest.mock import MagicMock

from src.query_processing.query_orchestrator import QueryOrchestrator
from src.config.models import AppConfig, SystemConfig, IngestionConfig, EmbeddingConfig, VectorStoreConfig, QueryConfig, TextNormalizerConfig, LLMConfig
from src.models import Chunk, Domain 

class TestQueryOrchestrator:
    """Suite de testes para a classe QueryOrchestrator."""
    
    @pytest.fixture
    def test_app_config(self) -> AppConfig:
        """Creates a default AppConfig for testing query orchestration."""
        return AppConfig(
            system=SystemConfig(),
            ingestion=IngestionConfig(),
            embedding=EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"), 
            vector_store=VectorStoreConfig(),
            query=QueryConfig(retrieval_k=3),
            llm=LLMConfig(model_repo_id="test-llm-model", prompt_template="Context:{context} Question:{query} Answer:"), 
            text_normalizer=TextNormalizerConfig()
        )

    @pytest.fixture
    def mock_domains(self):
        """Fixture para criar um mock de Domain."""
        return [
            Domain(
                id=1, 
                name="mock_domain", 
                description="d", 
                keywords="k", 
                db_path="p", 
                vector_store_path="p", 
                embeddings_model="sentence-transformers/all-MiniLM-L6-v2", 
                embeddings_dimension=384,
                faiss_index_type="IndexFlatL2"
            ),
            Domain(
                id=2, 
                name="mock_domain2", 
                description="d2", 
                keywords="k2", 
                db_path="p2", 
                vector_store_path="p2", 
                embeddings_model="sentence-transformers/all-MiniLM-L6-v2", 
                embeddings_dimension=384,
                faiss_index_type="IndexFlatL2"
            )
        ]

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
        assert orchestrator.embedding_generator.config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert orchestrator.hugging_face_manager.config.model_repo_id == "test-llm-model"
    
    def test_empty_query(self, orchestrator, mock_domains): # Pass orchestrator fixture
        """Testa o comportamento com uma query vazia."""
        with pytest.raises(ValueError) as exc_info:
            orchestrator._process_query("", mock_domains[0]) # Use the fixture
        assert "Query vazia ou inválida" in str(exc_info.value)
    
    def test_process_query(self, orchestrator, mocker, mock_domains): # Pass orchestrator fixture
        """Testa o processamento de uma query válida."""
        # Configure mocks on the fixture instance
        orchestrator.text_normalizer.normalize.return_value = "query normalizada"
        mock_embeddings = np.array([[0.1] * 384], dtype=np.float32) 
        orchestrator.embedding_generator.generate_embeddings.return_value = mock_embeddings
        
        result = orchestrator._process_query("teste de query", mock_domains[0])
        
        # Verifica se os métodos foram chamados corretamente
        orchestrator.text_normalizer.normalize.assert_called_once_with("teste de query")
        orchestrator.embedding_generator.generate_embeddings.assert_called_once_with("query normalizada")
        
        # Verifica se o resultado é o esperado
        assert np.array_equal(result, mock_embeddings)
    
    def test_embedding_error(self, orchestrator, mocker, mock_domains): # Pass orchestrator fixture
        """Testa o comportamento quando o embedding não pode ser gerado."""
        # Configure mocks on the fixture instance
        orchestrator.text_normalizer.normalize.return_value = "query normalizada"
        orchestrator.embedding_generator.generate_embeddings.return_value = np.array([])
        
        with pytest.raises(ValueError) as exc_info:
            orchestrator._process_query("teste de query", mock_domains[0])
        
        # Verify mocks were called
        orchestrator.text_normalizer.normalize.assert_called_once_with("teste de query")
        orchestrator.embedding_generator.generate_embeddings.assert_called_once_with("query normalizada")
        assert "Erro ao gerar o embedding da query" in str(exc_info.value)
    
    def test_retrieve_documents(self, orchestrator, mocker, mock_domains): # Pass orchestrator fixture
        """Testa a recuperação de documentos de múltiplos domínios."""
        mock_embedding = np.array([[0.1] * 384], dtype=np.float32)
        
        mock_domain = mock_domains[0]

        mock_faiss_results = [
            (np.array([[0.8]]), np.array([[101]])), 
            (np.array([[0.9]]), np.array([[202]]))  
        ]
        orchestrator.faiss_manager.search_faiss_index.side_effect = mock_faiss_results
        
        mock_conn = MagicMock()
        orchestrator.sqlite_manager.get_connection.return_value.__enter__.return_value = mock_conn
        
        mock_db_chunk = [Chunk(id=101, document_id=1, page_number=1, chunk_page_index=0, chunk_start_char_position=0, content="Chunk 101 from Domain 1")]
        
        orchestrator.sqlite_manager.get_chunks.return_value = mock_db_chunk
    
        # Initialize metrics data required by the method
        orchestrator.metrics_data = {"retrieved_chunks": 0}

        result = orchestrator._retrieve_documents(mock_embedding, mock_domain)
        
        assert orchestrator.faiss_manager.search_faiss_index.call_count == 1
        orchestrator.faiss_manager.search_faiss_index.assert_any_call(
            query_embedding=mock_embedding, 
            index_path=mock_domain.vector_store_path,
            dimension=mock_domain.embeddings_dimension,
        )

        assert orchestrator.sqlite_manager.get_chunks.call_count == 1
        orchestrator.sqlite_manager.get_chunks.assert_any_call(mock_conn, [101]) 
        
        assert result == mock_db_chunk
        assert len(result) == 1
    
    def test_retrieve_documents_empty_embedding(self, orchestrator, mock_domains): 
        """Testa a recuperação de documentos com embedding vazio."""
        mock_domain = mock_domains[0]
        
        with pytest.raises(ValueError) as exc_info:
            orchestrator._retrieve_documents(None, mock_domain)
        assert "Vetor de embedding vazio ou inválido" in str(exc_info.value)
    
    def test_prepare_context_prompt(self, orchestrator, test_app_config): 
        """Testa a preparação do prompt de contexto usando o template."""
        query = "Qual é a capital?"
        chunks = [
            Chunk(id=1, document_id=1, page_number=1, chunk_page_index=0, chunk_start_char_position=0, content="Brasília é a capital."),
            Chunk(id=2, document_id=1, page_number=2, chunk_page_index=0, chunk_start_char_position=0, content="Fica no planalto central.")
        ]
        expected_context = "Brasília é a capital.\n\nFica no planalto central."
        
        template = test_app_config.llm.prompt_template 

        # Define the fixed structure string correctly
        fixed_structure = (
            "\n\nContexto:\n"
            "{context}\n"
            "\nPergunta:\n"
            "{query}\n"
            "\nResposta útil:"
        )
        expected_prompt = template + fixed_structure.format(context=expected_context, query=query)
        
        # Configure the mock hf_manager on the orchestrator to have the config

        prompt = orchestrator._prepare_context_prompt(query, chunks)
        
        assert prompt == expected_prompt
        assert "Qual é a capital?" in prompt
        assert "Brasília é a capital." in prompt
        assert "Fica no planalto central." in prompt
        
    def test_query_llm(self, orchestrator, mocker, test_app_config, mock_domains): 
        """Testa o fluxo completo de processamento de query."""
        test_query = "Teste de query"
        
        mock_embedding = np.array([[0.1] * 384], dtype=np.float32)
        mocker.patch.object(orchestrator, '_process_query', return_value=mock_embedding)
        
        mock_domain = mock_domains[0]
        mocker.patch.object(orchestrator, '_select_domains', return_value=[mock_domain])
        
        mock_chunks = [Chunk(id=1, document_id=1, page_number=1, chunk_page_index=0, chunk_start_char_position=0, content="Chunk 1")]
        mocker.patch.object(orchestrator, '_retrieve_documents', return_value=mock_chunks)
        
        expected_context = "Chunk 1"
        # Calculate expected prompt using the new logic
        template = test_app_config.llm.prompt_template
        fixed_structure = (
            "\n\nContexto:\n"
            "{context}\n"
            "\nPergunta:\n"
            "{query}\n"
            "\nResposta útil:"
        )
        expected_prompt = template + fixed_structure.format(context=expected_context, query=test_query)
        #expected_prompt = test_app_config.llm.prompt_template.format(context=expected_context, query=test_query)
        
        mock_answer = "Esta é a resposta gerada pelo modelo."
        orchestrator.hugging_face_manager.generate_answer.return_value = mock_answer
        
        result = orchestrator.query_llm(test_query)
        
        orchestrator._select_domains.assert_called_once_with(test_query, None)
        orchestrator._process_query.assert_called_once_with(test_query, mock_domain)
        orchestrator._retrieve_documents.assert_called_once_with(mock_embedding, mock_domain)
        
        orchestrator.hugging_face_manager.generate_answer.assert_called_once_with(test_query, expected_prompt)
        
        assert isinstance(result, dict)
        assert result["answer"] == mock_answer
        assert result["question"] == test_query
        assert result["success"] == True
    
    def test_query_llm_empty_query(self, orchestrator): 
        """Testa query_llm com uma query vazia."""
        with pytest.raises(ValueError, match="Query vazia ou inválida"):
            orchestrator.query_llm("")
    
    def test_query_llm_error_handling(self, orchestrator, mocker):
        """Testa o tratamento de erros no fluxo completo."""
        test_query = "Teste de query"
        error_message = "Erro de teste"
        
        mocker.patch.object(orchestrator, '_select_domains', side_effect=Exception(error_message))
        
        with pytest.raises(Exception) as exc_info:
            orchestrator.query_llm(test_query)
            
        assert error_message in str(exc_info.value)
            
        assert orchestrator.metrics_data["success"] == False 

    def test_update_config_no_change(self, orchestrator, test_app_config, mocker):
        """Testa update_config: Sem alterações."""
        # Componentes já são mocks devido à fixture 'orchestrator'
        # Acessamos seus métodos .update_config diretamente para assertions
        mock_hf_update = orchestrator.hugging_face_manager.update_config
        mock_eg_update = orchestrator.embedding_generator.update_config
        mock_fm_update = orchestrator.faiss_manager.update_config
        mock_tn_update = orchestrator.text_normalizer.update_config
        mock_sm_update = orchestrator.sqlite_manager.update_config

        # Resetar mocks antes da chamada
        mock_hf_update.reset_mock()
        mock_eg_update.reset_mock()
        mock_fm_update.reset_mock()
        mock_tn_update.reset_mock()
        mock_sm_update.reset_mock()

        initial_config_ref = orchestrator.config
        new_config = test_app_config.model_copy() # Config idêntica

        orchestrator.update_config(new_config)

        assert orchestrator.config is initial_config_ref
        # Verifica se nenhum método de componente foi chamado
        mock_hf_update.assert_not_called()
        mock_eg_update.assert_not_called()
        mock_fm_update.assert_not_called()
        mock_tn_update.assert_not_called()
        mock_sm_update.assert_not_called()

    def test_update_config_single_change(self, orchestrator, test_app_config, mocker):
        """Testa update_config: Uma alteração (seção 'llm')."""
        # Acessamos os métodos .update_config diretamente
        mock_hf_update = orchestrator.hugging_face_manager.update_config
        mock_eg_update = orchestrator.embedding_generator.update_config
        mock_fm_update = orchestrator.faiss_manager.update_config
        mock_tn_update = orchestrator.text_normalizer.update_config
        mock_sm_update = orchestrator.sqlite_manager.update_config

        # Resetar mocks antes da chamada
        mock_hf_update.reset_mock()
        mock_eg_update.reset_mock()
        mock_fm_update.reset_mock()
        mock_tn_update.reset_mock()
        mock_sm_update.reset_mock()

        new_llm_config = LLMConfig(model_repo_id="new-test-model") # Config alterada
        new_config = test_app_config.model_copy(update={"llm": new_llm_config})
        initial_config_ref = orchestrator.config

        orchestrator.update_config(new_config)

        # Verifica se APENAS o update do HFManager foi chamado
        mock_hf_update.assert_called_once_with(new_llm_config)
        mock_eg_update.assert_not_called()
        mock_fm_update.assert_not_called()
        mock_tn_update.assert_not_called()
        mock_sm_update.assert_not_called()
        # Verifica se a referência do config foi atualizada
        assert orchestrator.config is new_config
        assert orchestrator.config != initial_config_ref

    def test_update_config_multiple_changes(self, orchestrator, test_app_config, mocker):
        """Testa update_config: Múltiplas alterações ('text_normalizer', 'vector_store')."""
        # Acessamos os métodos .update_config diretamente
        mock_hf_update = orchestrator.hugging_face_manager.update_config
        mock_eg_update = orchestrator.embedding_generator.update_config
        mock_fm_update = orchestrator.faiss_manager.update_config
        mock_tn_update = orchestrator.text_normalizer.update_config
        mock_sm_update = orchestrator.sqlite_manager.update_config

        # Resetar mocks antes da chamada
        mock_hf_update.reset_mock()
        mock_eg_update.reset_mock()
        mock_fm_update.reset_mock()
        mock_tn_update.reset_mock()
        mock_sm_update.reset_mock()

        new_tn_config = TextNormalizerConfig(use_lowercase=False) # Alterada
        new_vs_config = VectorStoreConfig(index_params={"nprobe": 16}) # Alterada
        new_config = test_app_config.model_copy(update={
            "text_normalizer": new_tn_config,
            "vector_store": new_vs_config
        })
        initial_config_ref = orchestrator.config

        orchestrator.update_config(new_config)

        # Verifica se APENAS os updates do TextNormalizer e FaissManager foram chamados
        mock_tn_update.assert_called_once_with(new_tn_config)
        mock_fm_update.assert_called_once_with(new_vs_config)
        mock_hf_update.assert_not_called()
        mock_eg_update.assert_not_called()
        mock_sm_update.assert_not_called()
        # Verifica se a referência do config foi atualizada
        assert orchestrator.config is new_config
        assert orchestrator.config != initial_config_ref 