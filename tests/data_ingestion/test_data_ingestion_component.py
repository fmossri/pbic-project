import os
import pytest
import shutil
from components.data_ingestion.data_ingestion_component import DataIngestionComponent
from .test_docs.test_pdf_generator import create_test_pdf

class TestDataIngestionComponent:
    """Suite de testes para a classe DataIngestionComponent."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Configura o ambiente de teste."""
        print("\nSetting up test environment...")
        
        # Obtém o caminho para o diretório test_docs
        test_docs_dir = os.path.join(os.path.dirname(__file__), "test_docs")
        self.test_pdfs_dir = os.path.join(test_docs_dir, "test_pdfs")
        self.empty_dir = os.path.join(test_docs_dir, "empty_dir")
        
        print(f"Test PDFs directory: {self.test_pdfs_dir}")
        
        # Cria o diretório de teste se não existir
        os.makedirs(self.test_pdfs_dir, exist_ok=True)
        os.makedirs(self.empty_dir, exist_ok=True)
        
        print("Creating test PDFs...")
        # Cria os arquivos PDF de teste
        create_test_pdf()
        
        # Verifica se os arquivos foram criados
        pdf_files = [f for f in os.listdir(self.test_pdfs_dir) if f.endswith('.pdf')]
        print(f"PDFs created: {pdf_files}")
        
        yield
        
        # Limpa os arquivos de teste após os testes
        
        if os.path.exists(self.test_pdfs_dir):
            shutil.rmtree(self.test_pdfs_dir)
        
        if os.path.exists(self.empty_dir):
            shutil.rmtree(self.empty_dir)
            
    def test_initialization(self):
        """Testa a inicialização do componente."""
        component = DataIngestionComponent()
        assert component is not None
        assert component.text_chunker is not None
        assert component.document_processor is not None

    def test_process_directory_with_valid_pdfs(self):
        """Testa o processamento de um diretório com PDFs válidos."""
        component = DataIngestionComponent()
        results = component.process_directory(self.test_pdfs_dir)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Verifica a estrutura dos resultados
        for doc_id, chunks in results.items():
            assert isinstance(chunks, list)
            assert len(chunks) > 0
            
            # Verifica a estrutura de cada chunk
            for chunk in chunks:
                assert isinstance(chunk, dict)
                assert "content" in chunk
                assert "metadata" in chunk
                assert "start_index" in chunk["metadata"]
                assert "page" in chunk["metadata"]

    def test_handle_duplicates(self):
        """Testa o tratamento de documentos duplicados."""
        component = DataIngestionComponent()
        
        # Processa o diretório com arquivos duplicados
        results = component.process_directory(self.test_pdfs_dir)
        
        # Verifica se apenas um documento foi processado (devido à detecção de duplicatas)
        assert len(results) == 1
        
        # Verifica se o documento processado tem chunks
        for doc_id, chunks in results.items():
            assert len(chunks) > 0

    def test_invalid_directory(self):
        """Testa o processamento de um diretório inválido."""
        component = DataIngestionComponent()
        
        # Testa diretório inexistente
        with pytest.raises(FileNotFoundError):
            component.process_directory("nonexistent_directory")
            
        # Testa caminho que não é um diretório
        test_file = os.path.join(self.test_pdfs_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
            
        with pytest.raises(NotADirectoryError):
            component.process_directory(test_file)
            
        # Limpa o arquivo de teste
        os.remove(test_file)

    def test_empty_directory(self):
        """Testa o processamento de um diretório vazio."""
        component = DataIngestionComponent()
        
        # Testa listagem de PDFs em diretório vazio
        with pytest.raises(ValueError) as exc_info:
            component.list_pdf_files(self.empty_dir)
        assert "Nenhum arquivo PDF encontrado" in str(exc_info.value)
        
        # Testa processamento de diretório vazio
        with pytest.raises(ValueError) as exc_info:
            component.process_directory(self.empty_dir)
        assert "Nenhum arquivo PDF encontrado" in str(exc_info.value)

    def test_chunk_overlap(self):
        """Testa se há sobreposição adequada entre chunks de páginas diferentes."""
        component = DataIngestionComponent()
        results = component.process_directory(self.test_pdfs_dir)
        
        # Verifica se há sobreposição entre chunks de páginas diferentes
        for doc_id, chunks in results.items():
            # Agrupa chunks por página
            page_chunks = {}
            for chunk in chunks:
                page_num = chunk["metadata"]["page"]
                if page_num not in page_chunks:
                    page_chunks[page_num] = []
                page_chunks[page_num].append(chunk)
            
            # Verifica sobreposição entre páginas consecutivas
            for page_num in sorted(page_chunks.keys())[:-1]:
                current_page_chunks = page_chunks[page_num]
                next_page_chunks = page_chunks[page_num + 1]
                
                # Verifica se o último chunk da página atual sobrepõe com o primeiro da próxima
                if current_page_chunks and next_page_chunks:
                    last_chunk = current_page_chunks[-1]["content"]
                    first_chunk = next_page_chunks[0]["content"]
                    overlap = set(last_chunk.split()) & set(first_chunk.split())
                    assert len(overlap) > 0, f"Deve haver sobreposição entre páginas {page_num} e {page_num + 1}" 