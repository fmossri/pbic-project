import sys
import time
import os
from typing import Dict, List
from src.data_ingestion import DataIngestionOrchestrator
from src.query_processing import QueryOrchestrator
from src.utils.logger import setup_logging, get_logger
from langchain.schema import Document

print("Iniciando a aplicação")
logger = None

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
    logger.info("Iniciando o processo de ingestao de dados")
    
    if not os.path.exists(directory_path):
        logger.error(f"O diretório {directory_path} não existe")
        raise FileNotFoundError(f"O diretório {directory_path} não existe")
    
    ingestion = DataIngestionOrchestrator()
    
    try:      
        # Processa os documentos e mede o tempo
        start_time = time.time()
        ingestion.process_directory(directory_path)
        total_time = time.time() - start_time
        logger.info("Ingestao de dados concluida", processing_time=total_time)
        
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        logger.error("Erro durante a ingestao de dados", error=str(e))
        raise e
    except KeyboardInterrupt as e:
        logger.warning("Ingestao de dados interrompida pelo usuario")
        raise e
    except Exception as e:
        logger.error("Erro inesperado durante a ingestao de dados", error=str(e))
        raise e

def answer_question(question: str) -> None:
    """
    Chama o processo de geração de resposta a partir de uma pergunta do usuário.

    Args:
        question (str): A pergunta a ser respondida.
    """
    logger.info("Iniciando o processo de geracao de resposta", question=question)
    query_orchestrator = QueryOrchestrator()
    try:
        answer = query_orchestrator.query_llm(question)
        print(f"\nPergunta: {question}\n")
        print(f"{answer}\n")
        logger.info("Pergunta respondida com sucesso", question=question, answer=answer)
        logger.info("Encerrando a aplicacao")
        sys.exit(0)
    except Exception as e:
        logger.error("Erro durante a geracao de resposta", question=question, error=str(e))
        raise e

def main():
    global logger
    

    invalid_use_message = "Uso incorreto. Use main.py --help para ver a lista de comandos disponíveis." 
    help_message = """
    Para ingestão de dados, use o comando:
              python main.py -i caminho/para/diretorio [--debug]
    Para fazer uma pergunta ao modelo, use o comando:
              python main.py -q "sua pergunta" [--debug]
    ** O argumento --debug é opcional. Exibe mais informações no console e gera logs mais detalhados.
    """

    debug = False
    if "--debug" in sys.argv:
        debug = True

    setup_logging(log_dir="logs", debug=debug)
    logger = get_logger("main")


    match (len(sys.argv)):
        case 2 if sys.argv[1] == "--help":
            logger.debug("Exibindo a mensagem de ajuda")
            print(help_message)
            return
        case 3:
            if sys.argv[1] == "-i":
                directory_path = sys.argv[2]
                logger.info("Iniciando a aplicacao - Ingestao de dados", args=sys.argv[1:])
                ingest_data(directory_path)
            elif sys.argv[1] == "-q":
                question = sys.argv[2]
                logger.info("Iniciando a aplicacao - Geracao de resposta", args=sys.argv[1:])
                answer_question(question)
        case 4:
            if sys.argv[1] == "-i" and sys.argv[3] == "--debug":
                directory_path = sys.argv[2]
                logger.info("Iniciando a aplicacao em modo debug - Ingestao de dados", args=sys.argv[1:])
                ingest_data(directory_path, debug=True)
            elif sys.argv[1] == "-q" and sys.argv[3] == "--debug":
                question = sys.argv[2]
                logger.info("Iniciando a aplicacao em modo debug - Geracao de resposta", args=sys.argv[1:])
                answer_question(question, debug=True)

        case _:
            logger.error("Argumentos de linha de comando invalidos", args=sys.argv[1:])
            print(invalid_use_message)
            return

if __name__ == "__main__":
    main() 