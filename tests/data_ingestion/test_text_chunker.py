import pytest
from src.data_ingestion.text_chunker import TextChunker
from src.models import Chunk
from src.config.models import IngestionConfig
from langchain.schema import Document

class TestTextChunker:
    """Suite de testes para a classe TextChunker."""

    @pytest.fixture
    def default_config(self):
        """Provides a default IngestionConfig for most tests."""
        return IngestionConfig(chunk_size=500, chunk_overlap=50, chunk_strategy="recursive")

    @pytest.fixture
    def custom_config(self):
        """Provides a custom IngestionConfig for specific tests."""
        return IngestionConfig(chunk_size=50, chunk_overlap=10, chunk_strategy="recursive")

    @pytest.fixture
    def chunker(self, default_config):
        """Fixture to provide a TextChunker instance with default config."""
        return TextChunker(config=default_config, log_domain="test_domain")

    def test_empty_text(self, chunker):
        """Testa o chunking de texto vazio."""
        docs = chunker._chunk_text("")
        assert isinstance(docs, list)
        assert len(docs) == 0

    def test_small_text(self, chunker):
        """Testa o chunking de texto menor que o tamanho do chunk."""
        text = "Este é um texto pequeno."
        docs = chunker._chunk_text(text)
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].page_content == text

    def test_default_chunking(self, chunker, default_config):
        """Testa o chunking padrão com texto maior que o tamanho do chunk."""
        text = "Este é um texto maior que será dividido em chunks. " * 20
        docs = chunker._chunk_text(text)
        assert len(docs) > 1
        assert all(isinstance(doc, Document) for doc in docs)
        assert all(len(doc.page_content) <= default_config.chunk_size for doc in docs)

    def test_custom_chunk_size(self, custom_config):
        """Testa o chunking com tamanho de chunk personalizado."""
        custom_chunker = TextChunker(config=custom_config, log_domain="test_domain")
        text = "Este é um texto que será dividido em chunks menores. " * 3
        docs = custom_chunker._chunk_text(text)
        assert len(docs) > 1
        assert all(isinstance(doc, Document) for doc in docs)
        assert all(len(doc.page_content) <= custom_config.chunk_size for doc in docs)

    def test_overlap_content(self, chunker):
        """Testa se há sobreposição adequada entre chunks."""
        text = "Este é um texto que será dividido em chunks com sobreposição. " * 15
        docs = chunker._chunk_text(text)
        assert len(docs) > 1

        for i in range(len(docs) - 1):
            current_chunk_content = docs[i].page_content
            next_chunk_content = docs[i+1].page_content
            overlap_start_index = next_chunk_content.find(current_chunk_content[-chunker.config.chunk_overlap:])
            assert overlap_start_index != -1, f"Overlap not found between chunk {i} and {i+1}"

    def test_natural_breaks(self, chunker, default_config):
        """Testa se o chunking respeita quebras naturais do texto quando possível."""
        # Cria parágrafos realistas considerando chunk_size=500:
        text = """A IA tem revolucionado diversos setores da sociedade moderna. Os avanços em processamento de dados abrem novas possibilidades. Pesquisadores buscam aplicações inovadoras.\n\n

Com avanços em PLN e visão computacional, seu impacto é significativo. As aplicações práticas podem ser vistas em diversos setores. A velocidade dessas mudanças surpreende especialistas.\n\n

O aprendizado de máquina permite que sistemas melhorem com a experiência. Através de algoritmos e dados, estas ferramentas identificam padrões complexos que seriam impossíveis de detectar manualmente. A precisão aumenta constantemente, tornando as previsões cada vez mais confiáveis. Os modelos se adaptam a novos cenários com uma flexibilidade impressionante. Esta característica é fundamental para aplicações em ambientes dinâmicos. As redes neurais profundas, em particular, demonstram capacidade excepcional de generalização. Sua arquitetura em camadas permite a extração hierárquica de características, desde as mais simples até as mais abstratas. O processo de treinamento iterativo refina gradualmente os pesos das conexões, melhorando o desempenho do modelo em tarefas complexas. A capacidade de processamento paralelo torna possível a análise de grandes volumes de dados em tempo real.\n\n

As implicações éticas não podem ser ignoradas neste contexto. A sociedade precisa discutir ativamente os limites e as diretrizes para o uso responsável dessas tecnologias. O futuro da IA depende de um equilíbrio entre inovação e responsabilidade.\n\n"""

        docs = chunker._chunk_text(text)
        chunk_contents = [doc.page_content for doc in docs]
        # Verifica se parágrafos pequenos consecutivos (P1 e P2) ficam juntos quando cabem
        p1_p2_juntos = False
        for content in chunk_contents:
            if "A IA tem revolucionado" in content and "Com avanços em PLN" in content:
                p1_p2_juntos = True
                break
        assert p1_p2_juntos, "Parágrafos pequenos consecutivos não ficaram juntos"
        
        # Verifica se o parágrafo grande (P3) foi dividido
        p3_dividido = False
        for content in chunk_contents:
            if "O aprendizado de máquina" in content:
                if not ("análise de grandes volumes de dados em tempo real" in content):
                    p3_dividido = True
                    break
        assert p3_dividido, "Parágrafos maiores que chunk_size devem ser divididos"
        
        for content in chunk_contents:
            assert len(content) <= default_config.chunk_size, f"Chunk excede tamanho máximo: {len(content)} > {default_config.chunk_size}"

    def test_create_chunks_metadata(self, chunker):
        """Testa se a função create_chunks adiciona metadados e retorna objetos Chunk."""
        text = "Texto para testar create_chunks e metadados. " * 10
        metadata = {"document_id": 123, "page_number": 42}
        
        chunks = chunker.create_chunks(text, metadata)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        
        for i, chunk in enumerate(chunks):
            assert chunk.document_id == 123
            assert chunk.page_number == 42
            assert chunk.chunk_page_index == i
            assert chunk.chunk_start_char_position != -1
            assert chunk.content is not None and len(chunk.content) > 0

    def test_create_chunks_no_metadata(self, chunker):
        """Testa create_chunks quando nenhum metadado é fornecido."""
        text = "Texto sem metadados para testar. " * 5
        chunks = chunker.create_chunks(text, None)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        
        for i, chunk in enumerate(chunks):
            assert chunk.document_id == 0
            assert chunk.page_number == 0
            assert chunk.chunk_page_index == i
            assert chunk.chunk_start_char_position != -1 
            assert chunk.content is not None

    def test_chunk_object_properties(self, chunker):
        """Testa se as propriedades do objeto Chunk estão corretas após create_chunks."""
        text = "Este é um texto para testar os metadados. " * 10
        metadata = {"document_id": 1, "page_number": 1}
        chunks = chunker.create_chunks(text, metadata)
        
        assert len(chunks) > 0
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, Chunk)
            assert hasattr(chunk, 'id')
            assert hasattr(chunk, 'document_id')
            assert hasattr(chunk, 'page_number')
            assert hasattr(chunk, 'chunk_page_index')
            assert hasattr(chunk, 'chunk_start_char_position')
            assert hasattr(chunk, 'content')

            assert chunk.id is None
            assert chunk.content is not None and len(chunk.content) > 0
            assert chunk.document_id == metadata["document_id"]
            assert chunk.page_number == metadata["page_number"]
            assert chunk.chunk_page_index == i
            assert isinstance(chunk.chunk_start_char_position, int)
            assert chunk.chunk_start_char_position >= 0
