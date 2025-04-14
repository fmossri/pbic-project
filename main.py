import sys
import os
from time import time
from typing import Dict
from src.data_ingestion import DataIngestionOrchestrator
from src.query_processing import QueryOrchestrator
from src.utils.logger import setup_logging, get_logger

print("Iniciando a aplicação")

def log_metrics(metrics_data: Dict[str, any], debug: bool = False, process: str = None) -> None:
    """
    Registra métricas do processamento.
    
    Args:
        metrics_data (Dict[str, any]): Dicionário contendo as métricas do processamento
        debug (bool): Se True, exibe métricas detalhadas
        process (str): Nome do processo sendo registrado
    """
    if not process:
        logger.error("Processo nao especificado")
        raise TypeError("Processo nao especificado")

    logger.info(f"Processo: {process}")
    for field, value in metrics_data.items():
        if type(value) == dict and debug:
            for key, value in value.items():
                logger.debug(f"{field} - {key}: {value}")

        else:
            logger.info(f"{field}: {value}")

def ingest_data(directory_path: str) -> None:
    """
    Chama o processo de ingestão de dados a partir de um diretório de arquivos PDF.

    Args:
        directory_path (str): Caminho para o diretório contendo os arquivos PDF.
    """
    logger.info("Iniciando o processo de ingestao de dados")
    
    if not os.path.exists(directory_path):
        logger.error(f"O diretorio {directory_path} nao existe")
        raise FileNotFoundError(f"O diretório {directory_path} não existe")
    
    ingestion = DataIngestionOrchestrator()
    
    try:      
        # Processa os documentos e mede o tempo
        ingestion.process_directory(directory_path)
        logger.info("Encerrando a aplicacao")
        
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        logger.error("Erro durante a ingestao de dados", error=str(e))
        raise e
    except KeyboardInterrupt as e:
        logger.warning("Ingestao de dados interrompida pelo usuario")
        raise e
    except Exception as e:
        logger.error("Erro inesperado durante a ingestao de dados", error=str(e))
        raise e

def answer_question(question: str) -> Dict[str, any]:
    """
    Chama o processo de geração de resposta a partir de uma pergunta do usuário.

    Args:
        question (str): A pergunta a ser respondida.
    """
    logger.info("Iniciando o processo de geracao de resposta", question=question)
    query_orchestrator = QueryOrchestrator()
    try:
        metrics_data = query_orchestrator.query_llm(question)
        print(f"\nPergunta: {question}\n")
        print(f"{metrics_data['answer']}\n")
        logger.info("Pergunta respondida com sucesso", question=question, answer=metrics_data["answer"])
        return metrics_data
    except Exception as e:
        logger.error("Erro durante a geracao de resposta", question=question, error=str(e))
        raise e

def main():
    start_time = time()
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
                metrics_data = ingest_data(directory_path)

                log_metrics(metrics_data, debug, process="data_ingestion")
                total_time = time() - start_time
                logger.info(f"Tempo total de execucao do programa: {total_time:.2f} segundos")
                logger.info("Encerrando a aplicacao")
            elif sys.argv[1] == "-q":
                question = sys.argv[2]
                logger.info("Iniciando a aplicacao - Geracao de resposta", args=sys.argv[1:])
                metrics_data = answer_question(question)

                log_metrics(metrics_data, debug, process="query_processing")
                total_time = time() - start_time
                logger.info(f"Tempo total de execucao do programa: {total_time:.2f} segundos")
                logger.info("Encerrando a aplicacao")
        case 4:
            if sys.argv[1] == "-i" and sys.argv[3] == "--debug":
                directory_path = sys.argv[2]
                logger.info("Iniciando a aplicacao em modo debug - Ingestao de dados", args=sys.argv[1:])
                metrics_data = ingest_data(directory_path)

                log_metrics(metrics_data, debug, process="data_ingestion")
                total_time = time() - start_time
                logger.info(f"Tempo total de execucao do programa: {total_time:.2f} segundos")
                logger.info("Encerrando a aplicacao")
            elif sys.argv[1] == "-q" and sys.argv[3] == "--debug":
                question = sys.argv[2]
                logger.info("Iniciando a aplicacao em modo debug - Geracao de resposta", args=sys.argv[1:])
                metrics_data = answer_question(question)

                log_metrics(metrics_data, debug, process="query_processing")
                total_time = time() - start_time
                logger.info(f"Tempo total de execucao do programa: {total_time:.2f} segundos")
                logger.info("Encerrando a aplicacao")
        case _:

            logger.error("Argumentos de linha de comando invalidos", args=sys.argv[1:])
            print(invalid_use_message)
            return

if __name__ == "__main__":
    main() 