import os
import pytest
import shutil
from pypdf.errors import PdfStreamError
from .test_docs.generate_test_pdfs import create_test_pdf
from src.data_ingestion.document_processor import DocumentProcessor
from langchain.schema import Document

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

    def test_calculate_hash_with_text(self):
        """Testa o cálculo do hash para um texto."""
        processor = DocumentProcessor(log_domain="test_domain")
        text_content = "Este é um texto de teste"
        hash_value = processor._calculate_hash(text_content)
        
        # Verifica o formato do hash
        assert hash_value is not None
        assert isinstance(hash_value, str)
        assert len(hash_value) == 32  # Comprimento do hash MD5
        
        # Verifica se o mesmo texto gera o mesmo hash
        assert processor._calculate_hash(text_content) == hash_value
        
    def test_calculate_hash_consistency(self):
        """Testa se diferentes textos geram hashes diferentes."""
        processor = DocumentProcessor(log_domain="test_domain")
        text1 = "Este é o primeiro texto"
        text2 = "Este é o segundo texto"
        
        hash1 = processor._calculate_hash(text1)
        hash2 = processor._calculate_hash(text2)
        
        # Verifica se textos diferentes geram hashes diferentes
        assert hash1 != hash2

    def test_calculate_hash_empty_text(self):
        """Testa o cálculo do hash para texto vazio."""
        processor = DocumentProcessor(log_domain="test_domain")
        hash_value = processor._calculate_hash("")
        
        # Verifica se texto vazio gera um hash válido
        assert hash_value is not None
        assert isinstance(hash_value, str)
        assert len(hash_value) == 32
        
    def test_extract_text(self):
        """Testa a extração de texto de um arquivo PDF válido."""
        processor = DocumentProcessor(log_domain="test_domain")
        pages = processor._extract_text(self.sample_pdf_path)
    
        assert len(pages) > 0
        assert isinstance(pages, list)
    
        # Verifica a estrutura das páginas
        for page in pages:
            assert isinstance(page, Document)
            assert hasattr(page, 'page_content')
            assert hasattr(page, 'metadata')
            assert 'page' in page.metadata
            assert 'source' in page.metadata
            print(page.metadata)
        
        # Verifica o conteúdo específico
        all_text = " ".join(page.page_content for page in pages)
        assert "Documento de Teste" in all_text
        assert "Este é um documento de teste abrangente" in all_text
        assert "Página 2 - Continuação" in all_text
        assert "FIM DO DOCUMENTO DE TESTE" in all_text
        
        # Verifica a numeração das páginas
        page_numbers = [page.metadata['page'] for page in pages]
        assert page_numbers == list(range(len(pages)))
        
    def test_extract_text_from_non_pdf(self):
        """Testa a extração de texto de um arquivo não-PDF."""
        # Cria um arquivo não-PDF para teste
        with open(self.non_pdf_path, "w") as text_file:
            text_file.write("Este não é um arquivo PDF")

        try:
            processor = DocumentProcessor(log_domain="test_domain")
            with pytest.raises(PdfStreamError) as exc_info:
                processor._extract_text(self.non_pdf_path)
            
            # Verifica se o erro está relacionado ao stream do PDF
            assert "Stream has ended unexpectedly" in str(exc_info.value)
        finally:
            # Garante que o arquivo é removido após o teste
            if os.path.exists(self.non_pdf_path):
                os.remove(self.non_pdf_path)