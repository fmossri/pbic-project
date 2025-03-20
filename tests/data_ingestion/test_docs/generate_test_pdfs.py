from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import os

def create_test_pdf():
    """Cria um PDF de teste abrangente com vários tipos de conteúdo."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_pdfs_dir = os.path.join(current_dir, "test_pdfs")
    
    # Cria os diretórios necessários
    os.makedirs(test_pdfs_dir, exist_ok=True)
    
    # Cria o primeiro documento
    pdf_path = os.path.join(test_pdfs_dir, "test_document.pdf")
    create_pdf_content(pdf_path)
    
    # Cria o documento duplicado
    duplicate_path = os.path.join(test_pdfs_dir, "duplicate_document.pdf")
    create_pdf_content(duplicate_path)

def create_pdf_content(pdf_path: str):
    """Cria o conteúdo do PDF."""
    c = canvas.Canvas(pdf_path, pagesize=letter)
    
    # Primeira página - Texto básico e formatação
    c.setFont("Helvetica", 24)
    c.drawString(1*inch, 10*inch, "Documento de Teste")
    
    c.setFont("Helvetica", 12)
    c.drawString(1*inch, 9*inch, "Este é um documento de teste abrangente para processamento de PDF.")
    
    # Adiciona múltiplos parágrafos
    text = [
        "Este documento contém múltiplos parágrafos e páginas para testar a extração de texto.",
        "Ele inclui vários tipos de conteúdo que podemos encontrar em documentos reais.",
        "O processador de documentos deve ser capaz de lidar com todo esse conteúdo corretamente.",
        "",
        "Alguns caracteres especiais: @ # $ % & * ( ) _ + < > ? \" '",
        "Números e datas: 12345 67890, 20/03/2024",
        "",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor",
        "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis",
        "nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
    ]
    
    y = 8*inch
    for line in text:
        c.drawString(1*inch, y, line)
        y -= 0.3*inch
    
    c.showPage()
    
    # Segunda página - Conteúdo adicional
    c.setFont("Helvetica", 14)
    c.drawString(1*inch, 10*inch, "Página 2 - Conteúdo Adicional")
    
    more_text = [
        "Esta segunda página garante que podemos lidar com documentos de múltiplas páginas.",
        "",
        "Termos técnicos: PDF, ASCII, UTF-8, Unicode, Base64",
        "Extensões de arquivo: .pdf, .txt, .doc, .docx",
        "",
        "Testando quebras",
        "de linha",
        "e diferentes",
        "padrões de",
        "espaçamento",
        "",
        "FIM DO DOCUMENTO DE TESTE"
    ]
    
    y = 9*inch
    for line in more_text:
        c.drawString(1*inch, y, line)
        y -= 0.3*inch
    
    c.save()

if __name__ == "__main__":
    create_test_pdf() 