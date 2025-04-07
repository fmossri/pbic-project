import pytest
import numpy as np
from components.shared.embedding_generator import EmbeddingGenerator
from components.models.embedding import Embedding

@pytest.fixture
def embedding_generator():
    """Fixture que fornece uma instância do EmbeddingGenerator."""
    return EmbeddingGenerator()

@pytest.fixture
def sample_chunks():
    """Fixture que fornece chunks de exemplo com seus IDs para os testes."""
    return [
        ("O desenvolvimento de sistemas de IA tem evoluído significativamente nos últimos anos.", 1),
        ("A análise de documentos e textos tem se tornado uma tarefa cada vez mais importante.", 2),
        ("O processamento de documentos PDF é uma funcionalidade essencial para muitas aplicações.", 3),
        ("A geração de embeddings é uma etapa fundamental no processamento de texto.", 4),
    ]

@pytest.fixture
def empty_chunks():
    """Fixture que fornece uma lista vazia de chunks."""
    return []

@pytest.fixture
def single_chunk():
    """Fixture que fornece um único chunk com ID para teste."""
    return [("O processamento de documentos PDF é uma funcionalidade essencial para muitas aplicações.", 1)]

def test_initialization():
    """Testa a inicialização do EmbeddingGenerator."""
    generator = EmbeddingGenerator()
    assert isinstance(generator, EmbeddingGenerator)
    assert generator.model_name == "all-MiniLM-L6-v2"
    assert generator.embedding_dimension > 0

def test_empty_chunks(embedding_generator):
    """Testa o comportamento com lista vazia de chunks."""
    embeddings = embedding_generator.generate_embeddings([])
    assert isinstance(embeddings, list)
    assert len(embeddings) == 0

def test_single_chunk(embedding_generator, single_chunk):
    """Testa o processamento de um único chunk."""
    embeddings = embedding_generator.generate_embeddings(single_chunk)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 1
    
    # Verifica se o retorno é um objeto Embedding
    embedding = embeddings[0]
    assert isinstance(embedding, Embedding)
    
    # Verifica os atributos do objeto Embedding
    assert embedding.id is None
    assert embedding.chunk_id == single_chunk[0][1]  # Deve corresponder ao ID fornecido
    assert embedding.dimension == embedding_generator.embedding_dimension
    assert embedding.faiss_index_path is None
    assert embedding.chunk_faiss_index is None
    assert isinstance(embedding.embedding, np.ndarray)
    assert embedding.embedding.shape[0] == embedding_generator.embedding_dimension

def test_multiple_chunks(embedding_generator, sample_chunks):
    """Testa o processamento de múltiplos chunks."""
    embeddings = embedding_generator.generate_embeddings(sample_chunks)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_chunks)
    
    # Verifica todos os embeddings gerados
    for i, embedding in enumerate(embeddings):
        assert isinstance(embedding, Embedding)
        assert embedding.id is None
        assert embedding.chunk_id == sample_chunks[i][1]  # Deve corresponder ao ID fornecido
        assert embedding.dimension == embedding_generator.embedding_dimension
        assert embedding.faiss_index_path is None
        assert embedding.chunk_faiss_index is None
        assert isinstance(embedding.embedding, np.ndarray)
        assert embedding.embedding.shape[0] == embedding_generator.embedding_dimension

def test_embedding_values_consistency(embedding_generator):
    """Testa se os valores dos embeddings são consistentes."""
    test_chunk = ("Chunk de teste para verificar consistência.", 1)
    
    # Gera embeddings para o mesmo texto duas vezes
    embeddings1 = embedding_generator.generate_embeddings([test_chunk])
    embeddings2 = embedding_generator.generate_embeddings([test_chunk])
    
    # Verifica se os vetores de embedding são iguais
    np.testing.assert_array_almost_equal(
        embeddings1[0].embedding,
        embeddings2[0].embedding
    )

def test_batch_processing(embedding_generator):
    """Testa se o processamento em batch produz os mesmos resultados."""
    # Cria chunks com textos variados e IDs sequenciais
    chunks = [(f"Chunk de teste {i}", i) for i in range(10)]
    
    # Processa sem batch (batch_size=1)
    embeddings_no_batch = embedding_generator.generate_embeddings(chunks, batch_size=1)
    
    # Processa com batch (batch_size=3)
    embeddings_with_batch = embedding_generator.generate_embeddings(chunks, batch_size=3)
    
    # Verifica se os resultados são iguais
    assert len(embeddings_no_batch) == len(embeddings_with_batch)
    
    for emb1, emb2 in zip(embeddings_no_batch, embeddings_with_batch):
        np.testing.assert_array_almost_equal(emb1.embedding, emb2.embedding)
        assert emb1.chunk_id == emb2.chunk_id

def test_chunk_id_preservation(embedding_generator):
    """Testa se os IDs dos chunks são preservados nos embeddings."""
    # Cria chunks com IDs não sequenciais para testar preservação
    chunks = [
        ("Texto 1", 100),
        ("Texto 2", 200),
        ("Texto 3", 300)
    ]
    
    embeddings = embedding_generator.generate_embeddings(chunks)
    
    # Verifica se os IDs foram preservados na ordem correta
    for i, embedding in enumerate(embeddings):
        assert embedding.chunk_id == chunks[i][1]

def test_embedding_dimension(embedding_generator):
    """Testa se a dimensão dos embeddings está correta."""
    chunk = ("Teste de dimensão do embedding", 1)
    embeddings = embedding_generator.generate_embeddings([chunk])
    
    # A dimensão do embedding deve corresponder à do modelo
    assert embeddings[0].dimension == embedding_generator.embedding_dimension
    assert embeddings[0].embedding.shape[0] == embedding_generator.embedding_dimension

# Opcional: manter um teste de benchmark se necessário
@pytest.mark.skip(reason="Benchmark test should be run manually")
def test_performance(embedding_generator, benchmark):
    """Testa a performance do processamento de embeddings."""
    # Cria uma lista de chunks com IDs para teste de performance
    large_chunks = [(f"Chunk de teste {i}", i) for i in range(100)]
    
    def process_chunks():
        return embedding_generator.generate_embeddings(large_chunks)
    
    # Executa o benchmark
    result = benchmark(process_chunks)
    assert result is not None