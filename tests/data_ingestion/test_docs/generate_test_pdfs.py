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
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
        "",
        "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt.",
        "",
        "At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga. Et harum quidem rerum facilis est et expedita distinctio.",
        "",
        "Nam libero tempore, cum soluta nobis est eligendi optio cumque nihil impedit quo minus id quod maxime placeat facere possimus, omnis voluptas assumenda est, omnis dolor repellendus. Temporibus autem quibusdam et aut officiis debitis aut rerum necessitatibus saepe eveniet ut et voluptates repudiandae sint et molestiae non recusandae."
    ]
    
    y = 8*inch
    for line in text:
        c.drawString(1*inch, y, line)
        y -= 0.3*inch
    
    c.showPage()
    
    # Segunda página - Continuação do texto
    c.setFont("Helvetica", 14)
    c.drawString(1*inch, 10*inch, "Página 2 - Continuação")
    
    more_text = [
        "Itaque earum rerum hic tenetur a sapiente delectus, ut aut reiciendis voluptatibus maiores alias consequatur aut perferendis doloribus asperiores repellat. Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?",
        "",
        "Temporibus autem quibusdam et aut officiis debitis aut rerum necessitatibus saepe eveniet ut et voluptates repudiandae sint et molestiae non recusandae. Itaque earum rerum hic tenetur a sapiente delectus, ut aut reiciendis voluptatibus maiores alias consequatur aut perferendis doloribus asperiores repellat.",
        "",
        "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit.",
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