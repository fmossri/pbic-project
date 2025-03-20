import pytest
from components.data_ingestion.text_chunker import TextChunker

class TestTextChunker:
    """Suite de testes para a classe TextChunker."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Configura o chunker para os testes."""
        self.chunker = TextChunker(chunk_size=500, overlap=100)

    def test_empty_text(self):
        """Testa o chunking de texto vazio."""
        chunks = self.chunker.chunk_text("")
        assert len(chunks) == 0

    def test_small_text(self):
        """Testa o chunking de texto menor que o tamanho do chunk."""
        text = "Este é um texto pequeno."
        chunks = self.chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0]["content"] == text
        assert chunks[0]["metadata"]["start_index"] == 0

    def test_default_chunking(self):
        """Testa o chunking padrão com texto maior que o tamanho do chunk."""
        text = "Este é um texto maior que será dividido em chunks. Precisamos garantir que o texto seja longo o suficiente para forçar múltiplos chunks com o tamanho atual. " * 10
        chunks = self.chunker.chunk_text(text)
        assert len(chunks) > 1
        assert all(len(chunk["content"]) <= 500 for chunk in chunks)
        assert all(chunk["metadata"]["start_index"] >= 0 for chunk in chunks)

    def test_custom_chunk_size(self):
        """Testa o chunking com tamanho de chunk personalizado."""
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "Este é um texto que será dividido em chunks menores. " * 3
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1
        assert all(len(chunk["content"]) <= 50 for chunk in chunks)

    def test_overlap_content(self):
        """Testa se o conteúdo sobreposto está presente nos chunks."""
        text = "Este é um texto que será dividido em chunks com sobreposição. " * 3
        chunks = self.chunker.chunk_text(text)
        
        # Verifica se há sobreposição entre chunks consecutivos
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]["content"]
            next_chunk = chunks[i + 1]["content"]
            overlap = set(current_chunk.split()) & set(next_chunk.split())
            assert len(overlap) > 0, "Deve haver sobreposição entre chunks consecutivos"

    def test_natural_breaks(self):
        """Testa se o chunking respeita quebras naturais do texto quando possível."""
        # Cria parágrafos realistas considerando chunk_size=500:
        text = """A IA tem revolucionado diversos setores da sociedade moderna. Os avanços em processamento de dados abrem novas possibilidades. Pesquisadores buscam aplicações inovadoras.\n\n

Com avanços em PLN e visão computacional, seu impacto é significativo. As aplicações práticas podem ser vistas em diversos setores. A velocidade dessas mudanças surpreende especialistas.\n\n

O aprendizado de máquina permite que sistemas melhorem com a experiência. Através de algoritmos e dados, estas ferramentas identificam padrões complexos que seriam impossíveis de detectar manualmente. A precisão aumenta constantemente, tornando as previsões cada vez mais confiáveis. Os modelos se adaptam a novos cenários com uma flexibilidade impressionante. Esta característica é fundamental para aplicações em ambientes dinâmicos. As redes neurais profundas, em particular, demonstram capacidade excepcional de generalização. Sua arquitetura em camadas permite a extração hierárquica de características, desde as mais simples até as mais abstratas. O processo de treinamento iterativo refina gradualmente os pesos das conexões, melhorando o desempenho do modelo em tarefas complexas. A capacidade de processamento paralelo torna possível a análise de grandes volumes de dados em tempo real.\n\n

As implicações éticas não podem ser ignoradas neste contexto. A sociedade precisa discutir ativamente os limites e as diretrizes para o uso responsável dessas tecnologias. O futuro da IA depende de um equilíbrio entre inovação e responsabilidade.\n\n"""

        chunks = self.chunker.chunk_text(text)
        
        # Verifica se parágrafos pequenos consecutivos (P1 e P2) ficam juntos quando cabem
        p1_p2_juntos = False
        for chunk in chunks:
            if "A IA tem revolucionado" in chunk["content"] and "Com avanços em PLN" in chunk["content"]:
                p1_p2_juntos = True
                break
        assert p1_p2_juntos, "Parágrafos pequenos que cabem juntos devem permanecer no mesmo chunk"
        
        # Verifica se o parágrafo grande (P3) foi dividido
        p3_dividido = False
        for chunk in chunks:
            if "O aprendizado de máquina" in chunk["content"]:
                if not ("análise de grandes volumes de dados em tempo real" in chunk["content"]):
                    p3_dividido = True
                    break
        assert p3_dividido, "Parágrafos maiores que chunk_size devem ser divididos"
        
        # Verifica se todos os chunks respeitam o tamanho máximo
        for chunk in chunks:
            assert len(chunk["content"]) <= 500, f"Chunk excede tamanho máximo: {len(chunk['content'])} > 500"

    def test_metadata(self):
        """Testa se os metadados estão sendo corretamente incluídos."""
        text = "Este é um texto para testar os metadados. " * 3
        chunks = self.chunker.chunk_text(text)
        
        for i, chunk in enumerate(chunks):
            assert "metadata" in chunk
            assert "start_index" in chunk["metadata"]
            assert chunk["metadata"]["start_index"] >= 0
            if i > 0:
                assert chunk["metadata"]["start_index"] > chunks[i-1]["metadata"]["start_index"] 