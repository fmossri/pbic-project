import pytest
import numpy as np
from src.utils.embedding_generator import EmbeddingGenerator
from src.config.models import EmbeddingConfig
from unittest.mock import MagicMock


class TestEmbeddingGenerator:

    @pytest.fixture(scope="module")
    def embedding_generator(self):
        """Fixture que fornece uma instância do EmbeddingGenerator.
        Carrega o modelo uma vez por módulo de teste."""
        # Create a default config for testing
        test_config = EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            batch_size=16,
            normalize_embeddings=True
        )
        return EmbeddingGenerator(config=test_config, log_domain="test_domain")

    @pytest.fixture
    def sample_texts(self):
        """Fixture que fornece textos de exemplo para os testes."""
        return [
            "O desenvolvimento de sistemas de IA tem evoluído significativamente nos últimos anos.",
            "A análise de documentos e textos tem se tornado uma tarefa cada vez mais importante.",
            "O processamento de documentos PDF é uma funcionalidade essencial para muitas aplicações.",
            "A geração de embeddings é uma etapa fundamental no processamento de texto.",
        ]

    @pytest.fixture
    def empty_texts(self):
        """Fixture que fornece uma lista vazia de textos."""
        return []

    @pytest.fixture
    def single_text(self):
        """Fixture que fornece um único texto para teste."""
        return ["O processamento de documentos PDF é uma funcionalidade essencial para muitas aplicações."]

    @pytest.fixture
    def initial_config(self):
        """Initial config for the generator in these tests."""
        return EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2", 
            device="cpu",
            batch_size=32,
            normalize_embeddings=True
            )

    @pytest.fixture
    def generator_for_update(self, initial_config, mocker):
        """Fixture to create a generator instance with mocked dependencies for update tests."""
        # --- Corrected Patching for SentenceTransformer --- #
        # Mock the SentenceTransformer class itself
        mock_st_class = mocker.patch('src.utils.embedding_generator.SentenceTransformer', autospec=True)
        
        # Create a mock instance that the patched class will return
        mock_st_instance = MagicMock(name="initial_st_instance")
        mock_st_instance.get_sentence_embedding_dimension.return_value = 384 # Dimension for MiniLM
        mock_st_instance.to = MagicMock(return_value=mock_st_instance, name="initial_to_method")
            
        # Configure the patched class's __call__ (or __init__) to return the mock instance
        mock_st_class.return_value = mock_st_instance 
        # --- End Correction --- #

        # Initialize the generator - this will call the patched SentenceTransformer
        generator = EmbeddingGenerator(config=initial_config, log_domain="test_update")
        generator.logger = MagicMock()

        # Reset mocks that were called during init 
        mock_st_class.reset_mock() # Reset the class mock itself
        mock_st_instance.to.reset_mock() # Reset the .to method mock on the instance
            
        # Store mocks/patches for use in tests
        generator._initial_model_mock = mock_st_instance # Store the first instance
        generator._st_class_patch = mock_st_class # Store the patch for the class

        return generator

    def test_initialization(self, embedding_generator):
        """Testa a inicialização do EmbeddingGenerator."""
        assert isinstance(embedding_generator, EmbeddingGenerator)
        assert embedding_generator.config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert embedding_generator.embedding_dimension > 0
        assert embedding_generator.config is not None

    def test_empty_texts(self, embedding_generator):
        """Testa o comportamento com lista vazia de textos."""
        embeddings = embedding_generator.generate_embeddings([])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.size == 0

    def test_single_text(self, embedding_generator, single_text):
        """Testa o processamento de um único texto."""
        embeddings = embedding_generator.generate_embeddings(single_text)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] == embedding_generator.embedding_dimension
        assert embeddings.dtype == np.float32

    def test_multiple_texts(self, embedding_generator, sample_texts):
        """Testa o processamento de múltiplos textos."""
        embeddings = embedding_generator.generate_embeddings(sample_texts)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(sample_texts)
        assert embeddings.shape[1] == embedding_generator.embedding_dimension
        assert embeddings.dtype == np.float32

    def test_embedding_values_consistency(self, embedding_generator):
        """Testa se os valores dos embeddings são consistentes."""
        test_text = "Chunk de teste para verificar consistência."
        
        # Gera embeddings para o mesmo texto duas vezes
        embeddings1 = embedding_generator.generate_embeddings([test_text])
        embeddings2 = embedding_generator.generate_embeddings([test_text])
        
        # Verifica se os vetores de embedding são iguais
        np.testing.assert_array_almost_equal(embeddings1[0], embeddings2[0], decimal=5)

    def test_batch_processing(self, embedding_generator):
        """Testa se o processamento em batch funciona como esperado."""
        # Cria textos variados
        texts = [f"Chunk de teste {i}" for i in range(embedding_generator.config.batch_size + 5)]
        
        # Gera embeddings usando o tamanho do lote interno
        embeddings = embedding_generator.generate_embeddings(texts)

        # Verifica se a forma do output é correta
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == embedding_generator.embedding_dimension


    def test_embedding_dimension(self, embedding_generator):
        """Testa se a dimensão dos embeddings está correta."""
        text = "Teste de dimensão do embedding"
        embeddings = embedding_generator.generate_embeddings([text])
        
        # A dimensão do embedding deve corresponder à do modelo
        assert embeddings.shape[1] == embedding_generator.embedding_dimension

    def test_update_no_change(self, generator_for_update, initial_config):
        """Testa update_config quando a nova configuração é idêntica."""
        mock_st_class = generator_for_update._st_class_patch # Classe patchada
        mock_to = generator_for_update.model.to 

        new_config = initial_config.model_copy()
        generator_for_update.update_config(new_config)

        # SentenceTransformer(...) deve NÃO ser chamado novamente
        mock_st_class.assert_not_called() 
        mock_to.assert_not_called() # Não deve mover o dispositivo
        assert generator_for_update.config == new_config
        assert generator_for_update.model == generator_for_update._initial_model_mock 

    def test_update_device_only(self, generator_for_update, initial_config):
        """Testa update_config quando apenas o dispositivo muda."""
        mock_st_class = generator_for_update._st_class_patch
        mock_to = generator_for_update.model.to # .to of the initial model instance

        new_config = initial_config.model_copy()
        new_config.device = "cuda"
        generator_for_update.update_config(new_config)

        mock_st_class.assert_not_called() # Should not re-initialize the class
        mock_to.assert_called_once_with("cuda") # Should call .to on the existing instance
        assert generator_for_update.config == new_config
        assert generator_for_update.config.device == "cuda"
        assert generator_for_update.model == generator_for_update._initial_model_mock

    def test_update_model_only(self, generator_for_update, initial_config):
        """Testa update_config quando apenas o modelo muda."""
        mock_st_class = generator_for_update._st_class_patch
        original_model_mock = generator_for_update._initial_model_mock
        original_mock_to = original_model_mock.to

        # Prepara uma nova instância mock para a próxima chamada a SentenceTransformer(...)
        new_model_mock = MagicMock(name="new_st_instance")
        new_model_mock.get_sentence_embedding_dimension.return_value = 768 # Dimension for mpnet
        new_model_mock.to = MagicMock(return_value=new_model_mock, name="new_to_method")
        # Configura a classe patchada para retornar a nova instância na próxima chamada
        mock_st_class.return_value = new_model_mock

        new_config = initial_config.model_copy()
        new_model_name = "sentence-transformers/all-mpnet-base-v2"
        new_config.model_name = new_model_name
        generator_for_update.update_config(new_config)

        # SentenceTransformer deve ser chamado uma vez para carregar o novo modelo
        mock_st_class.assert_called_once_with(new_model_name, device=initial_config.device)
        # Verifica que .to não foi chamado (já que o dispositivo não mudou, é tratado em init)
        new_model_mock.to.assert_not_called() 
        original_mock_to.assert_not_called()
        assert generator_for_update.config == new_config
        assert generator_for_update.config.model_name == new_model_name
        assert generator_for_update.model == new_model_mock
        assert generator_for_update.embedding_dimension == 768

    def test_update_model_and_device(self, generator_for_update, initial_config):
        """Testa update_config quando o modelo e o dispositivo mudam."""
        mock_st_class = generator_for_update._st_class_patch
        original_model_mock = generator_for_update._initial_model_mock
        original_mock_to = original_model_mock.to

        new_model_mock = MagicMock(name="new_st_instance")
        new_model_mock.get_sentence_embedding_dimension.return_value = 768
        new_model_mock.to = MagicMock(return_value=new_model_mock, name="new_to_method")
        mock_st_class.return_value = new_model_mock

        new_config = initial_config.model_copy()
        new_model_name = "sentence-transformers/all-mpnet-base-v2"
        new_config.model_name = new_model_name
        new_config.device = "cuda"
        generator_for_update.update_config(new_config)

        # Deve inicializar SentenceTransformer com o novo modelo e novo dispositivo
        mock_st_class.assert_called_once_with(new_model_name, device="cuda")
        
        new_model_mock.to.assert_not_called() 
        
        original_mock_to.assert_not_called()
        assert generator_for_update.config == new_config
        assert generator_for_update.config.model_name == new_model_name
        assert generator_for_update.config.device == "cuda"
        assert generator_for_update.model == new_model_mock
        assert generator_for_update.embedding_dimension == 768

    def test_update_other_params_only(self, generator_for_update, initial_config):
        """Testa update_config quando apenas parâmetros não críticos como batch_size mudam."""
        mock_st_class = generator_for_update._st_class_patch
        mock_to = generator_for_update.model.to

        new_config = initial_config.model_copy()
        new_config.batch_size = 64
        new_config.normalize_embeddings = False
        generator_for_update.update_config(new_config)

        mock_st_class.assert_not_called()
        mock_to.assert_not_called()
        assert generator_for_update.config == new_config
        assert generator_for_update.config.batch_size == 64
        assert generator_for_update.config.normalize_embeddings == False
        assert generator_for_update.model == generator_for_update._initial_model_mock