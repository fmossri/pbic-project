import sys
import time
from typing import Dict, List, Tuple
from components.data_ingestion import DataIngestionComponent
from components.embedding_generator import EmbeddingGenerator

def print_metrics(
    directory_path: str, 
    pdf_files: List[str], 
    results: Dict[str, List[Dict[str, str]]], 
    embeddings_info: Dict[str, Tuple[List[float], int]],
    processing_time: float,
    embedding_time: float
) -> None:
    """Imprime métricas do processamento.
    
    Args:
        directory_path (str): Caminho para o diretório processado
        pdf_files (List[str]): Lista de arquivos PDF encontrados
        results (Dict[str, List[Dict[str, str]]]): Resultados do processamento
        embeddings_info (Dict[str, Tuple[List[float], int]]): Informações dos embeddings gerados
        processing_time (float): Tempo total de processamento em segundos
        embedding_time (float): Tempo de geração dos embeddings em segundos
    """
    total_chunks = sum(len(chunks) for chunks in results.values())
    processed_files = len(results)
    total_files = len(pdf_files)
    avg_chunks_per_doc = total_chunks / processed_files if processed_files > 0 else 0
    
    # Calcula métricas adicionais
    total_pages = sum(
        max(chunk["metadata"]["page"] for chunk in chunks)
        for chunks in results.values()
    )
    avg_chunks_per_page = total_chunks / total_pages if total_pages > 0 else 0
    
    # Calcula tamanho médio dos chunks
    chunk_sizes = [
        len(chunk["content"])
        for chunks in results.values()
        for chunk in chunks
    ]
    avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    
    # Métricas de embeddings
    total_embeddings = len(embeddings_info)
    embedding_dimension = next(iter(embeddings_info.values()))[1] if embeddings_info else 0
    
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
    print(f"Total de embeddings gerados: {total_embeddings:,}")
    print(f"Dimensão dos embeddings: {embedding_dimension}")
    print(f"Tempo de geração dos embeddings: {embedding_time:.1f} segundos")
    print(f"Velocidade de embeddings: {total_embeddings/embedding_time:.1f} embeddings/segundo")
    print("-" * 80)
    print(f"Tempo total de processamento: {processing_time:.1f} segundos")
    print(f"Velocidade de processamento: {processed_files/processing_time:.1f} docs/segundo")
    print("=" * 80)

def main():
    """Ponto de entrada principal do script."""
    if len(sys.argv) != 2:
        print("Uso: python main.py caminho/para/diretorio")
        print("Exemplo: python main.py ./documentos")
        sys.exit(1)

    directory_path = sys.argv[1]
    
    try:
        # Inicializa os componentes
        ingestion = DataIngestionComponent()
        embedding_generator = EmbeddingGenerator()
        
        # Lista os arquivos PDF
        pdf_files = ingestion.list_pdf_files(directory_path)
        print(f"\nIniciando processamento de {len(pdf_files):,} arquivos PDF...")
        
        # Processa os documentos e mede o tempo
        start_time = time.time()
        results = ingestion.process_directory(directory_path)
        
        # Gera embeddings para todos os chunks
        print("\nGerando embeddings para os chunks...")
        embedding_start = time.time()
        embeddings_info = {}
        
        for filename, chunks in results.items():
            texts = [chunk["content"] for chunk in chunks]
            embeddings = embedding_generator.calculate_embeddings(texts)
            
            # Armazena os embeddings e sua dimensão
            for i, embedding in enumerate(embeddings):
                chunk_id = f"{filename}_{i}"
                embeddings_info[chunk_id] = (embedding, len(embedding))
        
        embedding_time = time.time() - embedding_start
        total_time = time.time() - start_time
        
        # Imprime as métricas
        print_metrics(directory_path, pdf_files, results, embeddings_info, total_time, embedding_time)
        
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        print(f"\nErro: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nProcessamento interrompido pelo usuário.")
        sys.exit(1)
    except Exception as e:
        print(f"\nErro inesperado: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 