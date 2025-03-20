import os
import pytest
import shutil
from pypdf.errors import PdfStreamError
from .test_docs.test_pdf_generator import create_test_pdf
from components.data_ingestion.document_processor import DocumentProcessor

class TestDocumentProcessor:
    """Suite de testes para a classe DocumentProcessor."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Configura os caminhos dos arquivos de teste."""
        # Obtém o caminho para o diretório test_docs
        test_docs_dir = os.path.join(os.path.dirname(__file__), "test_docs")
        self.test_pdfs_dir = os.path.join(test_docs_dir, "test_pdfs")
        
        # Cria o diretório de teste se não existir
        os.makedirs(self.test_pdfs_dir, exist_ok=True)
        
        # Cria os arquivos PDF de teste
        create_test_pdf()
        
        # Define os caminhos dos arquivos de teste
        self.sample_pdf_path = os.path.join(self.test_pdfs_dir, "test_document.pdf")
        self.non_pdf_path = os.path.join(self.test_pdfs_dir, "sample.txt")
        
        # Cria um arquivo não-PDF para teste
        with open(self.non_pdf_path, "w") as text_file:
            text_file.write("Este não é um arquivo PDF")
            
        yield
        
        # Limpa os arquivos de teste após os testes
        if os.path.exists(self.test_pdfs_dir):
            shutil.rmtree(self.test_pdfs_dir)
        
    def test_calculate_hash_with_valid_pdf(self):
        """Testa o cálculo do hash para um arquivo PDF válido."""
        processor = DocumentProcessor()
        hash_value = processor.calculate_hash(self.sample_pdf_path)
        
        assert hash_value is not None
        assert isinstance(hash_value, str)
        assert len(hash_value) == 32  # Comprimento do hash MD5
        
    def test_calculate_hash_with_nonexistent_file(self):
        """Testa o cálculo do hash para um arquivo inexistente."""
        processor = DocumentProcessor()
        with pytest.raises(FileNotFoundError):
            processor.calculate_hash("nonexistent.pdf")
        
    def test_extract_text_from_pdf(self):
        """Testa a extração de texto de um arquivo PDF válido."""
        processor = DocumentProcessor()
        pages = processor.extract_text(self.sample_pdf_path)
        
        assert len(pages) > 0
        assert isinstance(pages, list)
        
        # Verifica a estrutura das páginas
        for page_num, page_text in pages:
            assert isinstance(page_num, int)
            assert isinstance(page_text, str)
            assert page_text.strip() != ""
        
        # Verifica o conteúdo específico
        all_text = " ".join(text for _, text in pages)
        assert "Documento de Teste" in all_text
        assert "Este é um documento de teste abrangente" in all_text
        assert "Página 2 - Conteúdo Adicional" in all_text
        assert "FIM DO DOCUMENTO DE TESTE" in all_text
        
        # Verifica a numeração das páginas
        page_numbers = [num for num, _ in pages]
        assert page_numbers == list(range(1, len(pages) + 1))
        
    def test_extract_text_from_non_pdf(self):
        """Testa a extração de texto de um arquivo não-PDF."""
        # Cria um arquivo não-PDF para teste
        with open(self.non_pdf_path, "w") as text_file:
            text_file.write("Este não é um arquivo PDF")

        try:
            processor = DocumentProcessor()
            with pytest.raises(PdfStreamError) as exc_info:
                processor.extract_text(self.non_pdf_path)
            
            # Verifica se o erro está relacionado ao stream do PDF
            assert "Stream has ended unexpectedly" in str(exc_info.value)
        finally:
            # Garante que o arquivo é removido após o teste
            if os.path.exists(self.non_pdf_path):
                os.remove(self.non_pdf_path)