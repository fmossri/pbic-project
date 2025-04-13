import sys
import time
import os
from typing import Dict, List
from src.data_ingestion import DataIngestionOrchestrator
from src.query_processing import QueryOrchestrator
from src.utils.logger import setup_logging, get_logger
from langchain.schema import Document

# Setup logging
setup_logging(log_dir="logs", show_logs=True)
logger = get_logger("main")

def print_metrics(
    directory_path: str, 
    pdf_files: List[str], 
    results: Dict[str, List[Document]], 
    processing_time: float,
    embedding_time: float
) -> None:
    """Imprime métricas do processamento.
    
    Args:
        directory_path (str): Caminho para o diretório processado
        pdf_files (List[str]): Lista de arquivos PDF encontrados
        results (Dict[str, List[Document]]): Resultados do processamento
        processing_time (float): Tempo total de processamento em segundos
        embedding_time (float): Tempo de geração dos embeddings em segundos
    """
    total_chunks = sum(len(chunks) for chunks in results.values())
    processed_files = len(results)
    total_files = len(pdf_files)
    avg_chunks_per_doc = total_chunks / processed_files if processed_files > 0 else 0
    
    # Calcula métricas adicionais
    total_pages = sum(
        max(chunk.metadata["page"] for chunk in chunks)
        for chunks in results.values()
    )
    avg_chunks_per_page = total_chunks / total_pages if total_pages > 0 else 0
    
    # Calcula tamanho médio dos chunks
    chunk_sizes = [
        len(chunk.page_content)
        for chunks in results.values()
        for chunk in chunks
    ]
    avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    
    # Pega a dimensão do embedding do primeiro chunk (todos são iguais)
    first_chunk = next(iter(iter(results.values())))
    embedding_dimension = len(first_chunk[0].metadata["embedding"])
    
    # Log metrics
    logger.info("Métricas do processamento", 
                directory_path=directory_path,
                total_files=total_files,
                processed_files=processed_files,
                total_pages=total_pages,
                total_chunks=total_chunks,
                avg_chunks_per_doc=avg_chunks_per_doc,
                avg_chunks_per_page=avg_chunks_per_page,
                avg_chunk_size=avg_chunk_size,
                embedding_dimension=embedding_dimension,
                embedding_time=embedding_time,
                processing_time=processing_time)
    
    print("\n" + "=" * 80)
    print("MÉTRICAS DO PROCESSAMENTO")
    print("=" * 80)
    print(f"Diretório processado: {directory_path}")
    print(f"Total de PDFs encontrados: {total_files:,}")
    print(f"Documentos processados com sucesso: {processed_files:,}")
    print(f"Total de páginas processadas: {total_pages:,}")
    print(f"Total de chunks gerados: {total_chunks:,}")
    print(f"Média de chunks por documento: {avg_chunks_per_doc:.1f}")
    print(f"Média de chunks por página: {avg_chunks_per_page:.1f}")
    print(f"Tamanho médio dos chunks: {avg_chunk_size:.0f} caracteres")
    print("-" * 80)
    print("MÉTRICAS DE EMBEDDINGS")
    print(f"Total de embeddings gerados: {total_chunks:,}")
    print(f"Dimensão dos embeddings: {embedding_dimension}")
    print(f"Tempo de geração dos embeddings: {embedding_time:.1f} segundos")
    print(f"Velocidade de embeddings: {total_chunks/embedding_time:.1f} embeddings/segundo")
    print("-" * 80)
    print(f"Tempo total de processamento: {processing_time:.1f} segundos")
    print(f"Velocidade de processamento: {processed_files/processing_time:.1f} docs/segundo")
    print("=" * 80)

def ingest_data(directory_path: str) -> None:
    """
    Chama o processo de ingestão de dados a partir de um diretório de arquivos PDF.

    Args:
        directory_path (str): Caminho para o diretório contendo os arquivos PDF.
    """
    logger.info("Iniciando o processo de ingestão de dados")
    
    if not os.path.exists(directory_path):
        logger.error(f"O diretório {directory_path} não existe")
        raise FileNotFoundError(f"O diretório {directory_path} não existe")
    
    ingestion = DataIngestionOrchestrator()
    
    try:
        # Lista os arquivos PDF
        pdf_files = ingestion.list_pdf_files(directory_path)
        logger.info(f"{len(pdf_files)} arquivos PDF encontrados. Iniciando processamento...")
        
        # Processa os documentos e mede o tempo
        start_time = time.time()
        ingestion.process_directory(directory_path)
        total_time = time.time() - start_time
        logger.info("Ingestão de dados concluída", processing_time=total_time)
        
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        logger.error("Erro durante a ingestão de dados", error=str(e))
        raise e
    except KeyboardInterrupt as e:
        logger.warning("Ingestão de dados interrompida pelo usuário")
        raise e
    except Exception as e:
        logger.error("Erro inesperado durante a ingestão de dados", error=str(e))
        raise e

def answer_question(question: str) -> None:
    """
    Chama o processo de geração de resposta a partir de uma pergunta do usuário.

    Args:
        question (str): A pergunta a ser respondida.
    """
    logger.info("Iniciando o processo de geração de resposta", question=question)
    query_orchestrator = QueryOrchestrator()
    try:
        answer = query_orchestrator.query_llm(question)
        logger.info("Pergunta respondida com sucesso", question=question, answer=answer)
        print(f"\nResposta: {answer}")

    except Exception as e:
        logger.error("Erro durante a geração de resposta", question=question, error=str(e))
        raise e

def main():
    logger.info("Iniciando a aplicação", args=sys.argv[1:])
    
    if len(sys.argv) == 2 and sys.argv[1] == "--help":
        logger.info("Exibindo a mensagem de ajuda")
        print("""   Para ingestão de dados, use o comando:
              python main.py -i caminho/para/diretorio
    Para fazer uma pergunta ao modelo, use o comando:
              python main.py -q "sua pergunta"
        """)
        return
    if len(sys.argv) != 3:
        logger.error("Argumentos de linha de comando inválidos", args=sys.argv[1:])
        print("Uso incorreto. Use --help para ver a lista de comandos disponíveis.")
        return
    
    if sys.argv[1] == "-i":
        # Ingestão de dados
        directory_path = sys.argv[2]
        ingest_data(directory_path)
    elif sys.argv[1] == "-q":
        # Processamento de query
        question = sys.argv[2]
        answer_question(question)
    else:
        logger.error("Comando inválido", command=sys.argv[1])
        print("Uso incorreto. Use --help para ver a lista de comandos disponíveis.")
        return

if __name__ == "__main__":
    main() 