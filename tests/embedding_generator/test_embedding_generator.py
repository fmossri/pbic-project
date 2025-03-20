import pytest
import numpy as np
from components.embedding_generator.embedding_generator import EmbeddingGenerator

@pytest.fixture
def embedding_generator():
    """Fixture que fornece uma instância do EmbeddingGenerator para os testes."""
    return EmbeddingGenerator()

@pytest.fixture
def sample_chunks():
    """Fixture que fornece chunks de exemplo para os testes com tamanhos realistas."""
    return [
        """O desenvolvimento de sistemas de IA tem evoluído significativamente nos últimos anos, 
        com avanços notáveis em processamento de linguagem natural e visão computacional. 
        A integração dessas tecnologias em aplicações empresariais tem permitido a automação 
        de processos complexos e a extração de insights valiosos de grandes volumes de dados.""",  # ~400 chars
        
        """A análise de documentos e textos tem se tornado uma tarefa cada vez mais importante 
        para organizações que precisam processar e entender grandes volumes de informação. 
        Com o advento de técnicas avançadas de processamento de linguagem natural, 
        sistemas automatizados podem agora extrair significado e contexto de documentos 
        de forma mais eficiente e precisa.""",  # ~300 chars
        
        """O processamento de documentos PDF é uma funcionalidade essencial para muitas 
        aplicações empresariais, permitindo a extração e análise de informações contidas 
        em relatórios, contratos e outros documentos estruturados. A capacidade de 
        processar esses documentos de forma eficiente e precisa é crucial para 
        a automação de processos e a tomada de decisões baseada em dados.""",  # ~350 chars
        
        """A geração de embeddings é uma etapa fundamental no processamento de texto, 
        permitindo a representação vetorial de documentos e a realização de operações 
        de similaridade e busca semântica. Essas representações numéricas capturam 
        o significado semântico do texto e permitem a comparação eficiente entre 
        diferentes documentos.""",  # ~250 chars
    ]

@pytest.fixture
def empty_chunks():
    """Fixture que fornece uma lista vazia de chunks."""
    return []

@pytest.fixture
def single_chunk():
    """Fixture que fornece um único chunk para teste."""
    return ["""O processamento de documentos PDF é uma funcionalidade essencial para muitas 
    aplicações empresariais, permitindo a extração e análise de informações contidas 
    em relatórios, contratos e outros documentos estruturados. A capacidade de 
    processar esses documentos de forma eficiente e precisa é crucial para 
    a automação de processos e a tomada de decisões baseada em dados."""]  # ~350 chars

def test_initialization(embedding_generator):
    """Testa a inicialização correta do EmbeddingGenerator."""
    assert embedding_generator.model_name == "all-MiniLM-L6-v2"
    assert embedding_generator.embedding_dimension > 0
    assert embedding_generator.model is not None

def test_empty_chunks(embedding_generator, empty_chunks):
    """Testa o comportamento quando uma lista vazia de chunks é fornecida."""
    embeddings = embedding_generator.calculate_embeddings(empty_chunks)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.size == 0

def test_single_chunk(embedding_generator, single_chunk):
    """Testa o processamento de um único chunk."""
    embeddings = embedding_generator.calculate_embeddings(single_chunk)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1, embedding_generator.embedding_dimension)

def test_multiple_chunks(embedding_generator, sample_chunks):
    """Testa o processamento de múltiplos chunks."""
    embeddings = embedding_generator.calculate_embeddings(sample_chunks)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(sample_chunks), embedding_generator.embedding_dimension)

def test_embedding_values(embedding_generator, sample_chunks):
    """Testa se os valores dos embeddings estão dentro do intervalo esperado."""
    embeddings = embedding_generator.calculate_embeddings(sample_chunks)
    # Verifica se os valores estão entre -1 e 1 (embeddings normalizados)
    assert np.all(embeddings >= -1) and np.all(embeddings <= 1)

def test_batch_processing(embedding_generator, sample_chunks):
    """Testa o processamento em batch com diferentes tamanhos."""
    # Processamento sem batch
    embeddings_no_batch = embedding_generator.calculate_embeddings(sample_chunks)
    
    # Processamento com batch
    embeddings_with_batch = embedding_generator.calculate_embeddings(sample_chunks, batch_size=2)
    
    # Verifica se os resultados são iguais
    np.testing.assert_array_almost_equal(embeddings_no_batch, embeddings_with_batch)

def test_model_info(embedding_generator):
    """Testa se as informações do modelo estão corretas."""
    info = embedding_generator.get_model_info()
    assert isinstance(info, dict)
    assert "model_name" in info
    assert "embedding_dimension" in info
    assert "max_sequence_length" in info
    assert info["model_name"] == "all-MiniLM-L6-v2"
    assert info["embedding_dimension"] == embedding_generator.embedding_dimension

def test_embedding_dimension(embedding_generator):
    """Testa se a dimensão do embedding está correta."""
    dimension = embedding_generator.get_embedding_dimension()
    assert isinstance(dimension, int)
    assert dimension > 0
    assert dimension == embedding_generator.embedding_dimension

@pytest.mark.benchmark
def test_performance(embedding_generator, benchmark):
    """Testa a performance do processamento de embeddings."""
    # Cria uma lista maior de chunks para teste de performance
    large_chunks = ["Chunk de teste " + str(i) for i in range(100)]
    
    def process_chunks():
        return embedding_generator.calculate_embeddings(large_chunks)
    
    # Executa o benchmark
    result = benchmark(process_chunks)
    assert result is not None 