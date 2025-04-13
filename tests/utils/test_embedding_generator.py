import pytest
import numpy as np
from src.utils.embedding_generator import EmbeddingGenerator

@pytest.fixture
def embedding_generator():
    """Fixture que fornece uma instância do EmbeddingGenerator."""
    return EmbeddingGenerator(log_domain="test_domain")

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

def test_initialization():
    """Testa a inicialização do EmbeddingGenerator."""
    generator = EmbeddingGenerator(log_domain="test_domain")
    assert isinstance(generator, EmbeddingGenerator)
    assert generator.model_name == "all-MiniLM-L6-v2"
    assert generator.embedding_dimension > 0

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
    np.testing.assert_array_almost_equal(embeddings1[0], embeddings2[0])

def test_batch_processing(embedding_generator):
    """Testa se o processamento em batch produz os mesmos resultados."""
    # Cria textos variados
    texts = [f"Chunk de teste {i}" for i in range(10)]
    
    # Processa sem batch (batch_size=1)
    embeddings_no_batch = embedding_generator.generate_embeddings(texts, batch_size=1)
    
    # Processa com batch (batch_size=3)
    embeddings_with_batch = embedding_generator.generate_embeddings(texts, batch_size=3)
    
    # Verifica se os resultados são iguais
    assert embeddings_no_batch.shape == embeddings_with_batch.shape
    
    for i in range(len(texts)):
        np.testing.assert_array_almost_equal(embeddings_no_batch[i], embeddings_with_batch[i])

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