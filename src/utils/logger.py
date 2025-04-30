"""
Logger para o sistema RAG.

Este módulo fornece uma configuração de registro de logs centralizada para todo o sistema,
suportando saída tanto no console quanto em arquivos com capacidades de registro estruturado.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any
from logging.handlers import RotatingFileHandler

class Logger:
    """Um logger que suporta formatação JSON e informações de contexto."""
    
    def __init__(self, name: str, log_domain: str = "default"):
        """
        Inicializa o logger.
        
        Args:
            name: O nome do logger (geralmente o nome do módulo)
            log_domain: O contexto do domínio para o logger; em geral "Ingestão de Dados" ou "Processamento de Queries"
        """
        self.logger = logging.getLogger(name)
        self.log_domain = log_domain
        self.context: Dict[str, Any] = {}
        
    def _format_message(self, message: str, level: str, **kwargs) -> str:
        """Formata a mensagem de log com contexto e campos adicionais."""
        # Obtém o nome da função chamada a partir do stack
        import inspect
        frame = inspect.currentframe()
        try:
            # Subir 3 frames para obter a função real que chamou
            # 1 para esta função
            # 1 para o método do logger (info, error, etc)
            # 1 para o código real que chamou
            caller_frame = frame.f_back.f_back
            function_name = caller_frame.f_code.co_name
        except (AttributeError, TypeError):
            function_name = "unknown"
        finally:
            del frame  # Deleta a referência do frame

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "log_domain": self.log_domain,
            "function": function_name,
            "message": message,
            "caller": self.logger.name,
            **self.context,
            **kwargs
        }
        return json.dumps(log_data)
    
    def set_context(self, **kwargs) -> None:
        """Define contexto adicional para todas as mensagens de log subsequentes."""
        self.context.update(kwargs)
    
    def clear_context(self) -> None:
        """Limpa todas as informações de contexto."""
        self.context.clear()
    
    def info(self, message: str, **kwargs) -> None:
        """Registra uma mensagem de informação com contexto."""
        self.logger.info(
            self._format_message(message, "INFO", **kwargs),
            stacklevel=2
        )
    
    def error(self, message: str, **kwargs) -> None:
        """Registra uma mensagem de erro com contexto."""
        self.logger.error(
            self._format_message(message, "ERROR", **kwargs),
            exc_info=True,
            stack_info=True,
            stacklevel=2
        )
    
    def warning(self, message: str, **kwargs) -> None:
        """Registra uma mensagem de aviso com contexto."""
        self.logger.warning(
            self._format_message(message, "WARNING", **kwargs),
            stacklevel=2
        )
    
    def debug(self, message: str, **kwargs) -> None:
        """Registra uma mensagem de debug com contexto."""
        self.logger.debug(
            self._format_message(message, "DEBUG", **kwargs),
            stacklevel=2
        )
    
    def critical(self, message: str, **kwargs) -> None:
        """Registra uma mensagem crítica com contexto."""
        self.logger.critical(
            self._format_message(message, "CRITICAL", **kwargs),
            exc_info=True,
            stack_info=True,
            stacklevel=2
        )

def setup_logging(
    log_dir: str = "logs",
    debug: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5  # Keep 5 backup files by default
) -> None:
    """
    Configura o sistema de registro de logs da aplicação.
    Cria um novo arquivo de log para cada execução da aplicação.
    Se o arquivo de log exceder o tamanho máximo, cria um novo com o mesmo identificador de execução.
    
    Args:
        log_dir: Diretório onde os arquivos de log serão armazenados
        debug: Se deve mostrar logs no console
        max_file_size: Tamanho máximo de cada arquivo de log em bytes
        backup_count: Número de arquivos de backup a serem mantidos quando rotacionados
    """
    # Cria o diretório de logs se ele não existir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Gera um identificador único para a execução
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Cria o nome do arquivo de log base
    base_log_file = os.path.join(log_dir, f"rag_system_{run_id}.log")
    
    # Limpa qualquer handler existente
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configura o logger raiz
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Cria formatadores
    json_formatter = logging.Formatter('%(message)s')  # Para saída de arquivo
    
    class JsonConsoleFormatter(logging.Formatter):
        def format(self, record):
            try:
                # Tenta fazer o parsing da mensagem como JSON
                log_data = json.loads(record.getMessage())
                # Formata usando os campos do JSON
                info_format = f"{log_data.get('timestamp', '')} - {log_data.get('message', '')}"
                debug_format = f"{log_data.get('timestamp', '')} - {log_data.get('caller', '')} - {log_data.get('level', '')} - {log_data.get('message', '')}"
                return debug_format if debug else info_format
            except json.JSONDecodeError:
                # Se não for JSON, use a mensagem original
                return record.getMessage()
    
    # Cria handlers
    file_handler = RotatingFileHandler(
        base_log_file,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(json_formatter)
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JsonConsoleFormatter())
    console_handler.setLevel(logging.INFO if debug else logging.WARNING) 
    
    # Adiciona handlers ao logger raiz
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # --- Silencia loggers de dependências (como file watcher) --- 
    noisy_loggers = ['watchdog', 'streamlit.watcher.local_sources_watcher', 'asyncio']
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING) 

    # Configura loggers para bibliotecas - apenas erros
    for lib in ['torch', 'transformers', 'sentence_transformers']:
        lib_logger = logging.getLogger(lib)
        lib_logger.setLevel(logging.ERROR)  # Apenas erros
        lib_logger.propagate = True  # Propaga para o root logger
    
    # Registra a configuração
    root_logger.info(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "message": "Sistema de registro de logs configurado",
        "run_id": run_id,
        "log_file": base_log_file,
        "debug": debug,
        "max_file_size": max_file_size,
        "backup_count": backup_count
    }))

def get_logger(name: str, log_domain: str = "default") -> Logger:
    """
    Obtém uma instância do logger.
    
    Args:
        name: O nome do logger (geralmente o nome do módulo)
        log_domain: O contexto do domínio do logger; em geral "Ingestão de dados" ou "Processamento de queries"
        
    Returns:
        Uma instância do Logger
    """
    return Logger(name, log_domain) 