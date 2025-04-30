import pytest
import numpy as np
from src.utils.embedding_generator import EmbeddingGenerator
from src.config.models import EmbeddingConfig

@pytest.fixture(scope="module")
def embedding_generator():
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
def sample_texts():
    """Fixture que fornece textos de exemplo para os testes."""
    return [
        "O desenvolvimento de sistemas de IA tem evoluído significativamente nos últimos anos.",
        "A análise de documentos e textos tem se tornado uma tarefa cada vez mais importante.",
        "O processamento de documentos PDF é uma funcionalidade essencial para muitas aplicações.",
        "A geração de embeddings é uma etapa fundamental no processamento de texto.",
    ]

@pytest.fixture
def empty_texts():
    """Fixture que fornece uma lista vazia de textos."""
    return []

@pytest.fixture
def single_text():
    """Fixture que fornece um único texto para teste."""
    return ["O processamento de documentos PDF é uma funcionalidade essencial para muitas aplicações."]

def test_initialization(embedding_generator):
    """Testa a inicialização do EmbeddingGenerator."""
    assert isinstance(embedding_generator, EmbeddingGenerator)
    assert embedding_generator.config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert embedding_generator.embedding_dimension > 0
    assert embedding_generator.config is not None

def test_empty_texts(embedding_generator):
    """Testa o comportamento com lista vazia de textos."""
    embeddings = embedding_generator.generate_embeddings([])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.size == 0

def test_single_text(embedding_generator, single_text):
    """Testa o processamento de um único texto."""
    embeddings = embedding_generator.generate_embeddings(single_text)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] == embedding_generator.embedding_dimension
    assert embeddings.dtype == np.float32

def test_multiple_texts(embedding_generator, sample_texts):
    """Testa o processamento de múltiplos textos."""
    embeddings = embedding_generator.generate_embeddings(sample_texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(sample_texts)
    assert embeddings.shape[1] == embedding_generator.embedding_dimension
    assert embeddings.dtype == np.float32

def test_embedding_values_consistency(embedding_generator):
    """Testa se os valores dos embeddings são consistentes."""
    test_text = "Chunk de teste para verificar consistência."
    
    # Gera embeddings para o mesmo texto duas vezes
    embeddings1 = embedding_generator.generate_embeddings([test_text])
    embeddings2 = embedding_generator.generate_embeddings([test_text])
    
    # Verifica se os vetores de embedding são iguais
    np.testing.assert_array_almost_equal(embeddings1[0], embeddings2[0], decimal=5)

def test_batch_processing(embedding_generator):
    """Testa se o processamento em batch funciona como esperado."""
    # Cria textos variados
    texts = [f"Chunk de teste {i}" for i in range(embedding_generator.config.batch_size + 5)]
    
    # Gera embeddings usando o tamanho do lote interno
    embeddings = embedding_generator.generate_embeddings(texts)

    # Verifica se a forma do output é correta
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] == embedding_generator.embedding_dimension

    # A lógica de comparação anterior dependia de alterar o parâmetro batch_size,
    # que não é mais possível. Este teste agora verifica principalmente se a execução 
    # com o tamanho do lote interno funciona sem erro e produz a forma de output correta.
    
    # Opcional: Poderia mockar SentenceTransformer.encode para verificar chamadas se necessário,
    # mas por enquanto, verificar a execução com sucesso é suficiente.

def test_embedding_dimension(embedding_generator):
    """Testa se a dimensão dos embeddings está correta."""
    text = "Teste de dimensão do embedding"
    embeddings = embedding_generator.generate_embeddings([text])
    
    # A dimensão do embedding deve corresponder à do modelo
    assert embeddings.shape[1] == embedding_generator.embedding_dimension
    
# Opcional: manter um teste de benchmark se necessário
@pytest.mark.skip(reason="Benchmark test should be run manually")
def test_performance(embedding_generator, benchmark):
    """Testa a performance do processamento de embeddings."""
    # Cria uma lista de textos para teste de performance
    large_texts = [f"Chunk de teste {i}" for i in range(100)]
    
    def process_texts():
        return embedding_generator.generate_embeddings(large_texts)
    
    # Executa o benchmark
    result = benchmark(process_texts)
    assert result is not None