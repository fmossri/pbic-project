import pytest
import numpy as np
from components.embedding_generator import EmbeddingGenerator

@pytest.fixture
def embedding_generator():
    """Fixture que fornece uma instância do EmbeddingGenerator."""
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

def test_initialization():
    """Testa a inicialização do EmbeddingGenerator."""
    generator = EmbeddingGenerator()
    assert isinstance(generator, EmbeddingGenerator)
    assert generator.model_name == "all-MiniLM-L6-v2"

def test_empty_chunks(embedding_generator):
    """Testa o comportamento com lista vazia de chunks."""
    embeddings = embedding_generator.calculate_embeddings([])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.size == 0

def test_single_chunk(embedding_generator):
    """Testa o processamento de um único chunk."""
    chunk = "Este é um chunk de teste."
    embeddings = embedding_generator.calculate_embeddings([chunk])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] == embedding_generator.embedding_dimension

def test_multiple_chunks(embedding_generator):
    """Testa o processamento de múltiplos chunks."""
    chunks = [
        "Primeiro chunk de teste.",
        "Segundo chunk de teste.",
        "Terceiro chunk de teste."
    ]
    embeddings = embedding_generator.calculate_embeddings(chunks)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] == embedding_generator.embedding_dimension

def test_embedding_values(embedding_generator):
    """Testa se os valores dos embeddings são consistentes."""
    chunk = "Chunk de teste para verificar consistência."
    embeddings1 = embedding_generator.calculate_embeddings([chunk])
    embeddings2 = embedding_generator.calculate_embeddings([chunk])
    np.testing.assert_array_almost_equal(embeddings1, embeddings2)

def test_batch_processing(embedding_generator):
    """Testa se o processamento em batch produz os mesmos resultados."""
    chunks = ["Chunk " + str(i) for i in range(10)]
    
    # Processa sem batch (batch_size=1)
    embeddings_no_batch = embedding_generator.calculate_embeddings(chunks, batch_size=1)
    
    # Processa com batch (batch_size=3)
    embeddings_with_batch = embedding_generator.calculate_embeddings(chunks, batch_size=3)
    
    # Verifica se os resultados são iguais
    np.testing.assert_array_almost_equal(embeddings_no_batch, embeddings_with_batch)

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