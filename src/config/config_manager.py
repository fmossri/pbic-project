import sys
from pathlib import Path
from typing import Optional
import shutil 
import toml 
import tomli

from pydantic import ValidationError

from .models import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__, log_domain="config_manager")

_config_path: Path = Path("config.toml").resolve()

class ConfigurationError(Exception):
    """Exceção personalizada para erros de carregamento de configuração."""
    pass

def load_config(config_path: Path = _config_path) -> AppConfig:
    """
    Carrega a configuração do arquivo TOML especificado, valida e retorna.

    Args:
        config_path: O caminho para o arquivo de configuração.
                       Padrão é `config.toml` na raiz do projeto.

    Returns:
        O objeto AppConfig validado.

    Raises:
        ConfigurationError: Se o arquivo não for encontrado, não puder ser analisado,
                            ou falhar na validação.
    """
    if not config_path.is_file():
        msg = f"Arquivo de configuração não encontrado: {config_path}"
        logger.error(msg)
        raise ConfigurationError(msg)

    try:
        logger.info(f"Carregando configuração de: {config_path}")
        with open(config_path, "rb") as f:
            config_data = tomli.load(f)

        logger.debug("Dados brutos de configuração carregados, validando...")
        validated_config = AppConfig(**config_data)
        logger.info("Configuração carregada e validada com sucesso.")
        return validated_config

    except tomli.TOMLDecodeError as e:
        msg = f"Erro ao analisar o arquivo de configuração {config_path}: {e}"
        logger.error(msg)
        raise ConfigurationError(msg) from e

    except ValidationError as e:
        logger.error(f"Falha na validação da configuração para {config_path}:\n{e}")
        raise ConfigurationError(f"Falha na validação da configuração para {config_path}:\n{e}") from e

    except Exception as e:
        msg = f"Ocorreu um erro inesperado ao carregar a configuração de {config_path}: {e}"
        logger.error(msg, exc_info=True)
        raise ConfigurationError(msg) from e

def get_config() -> AppConfig:
    """
    Obtém a configuração da aplicação carregando-a do arquivo.
    (Cache removido; o cache agora é gerenciado pelo chamador, ex: Streamlit).

    Returns:
        O objeto AppConfig validado.

    Raises:
        ConfigurationError: Se a configuração não puder ser carregada.
    """
    logger.debug("Chamando load_config para obter a configuração (cache interno removido).")
    return load_config()

def save_config(config: AppConfig, config_path: Path = _config_path) -> None:
    """
    Salva o objeto AppConfig fornecido no arquivo TOML especificado.

    Args:
        config: O objeto AppConfig a ser salvo.
        config_path: O caminho para o arquivo de configuração onde salvar.
                       Padrão é `config.toml` na raiz do projeto.

    Raises:
        ConfigurationError: Se ocorrer um erro durante a escrita do arquivo.
        TypeError: Se o objeto config não for uma instância de AppConfig.
    """
    if not isinstance(config, AppConfig):
        msg = "Objeto fornecido para save_config não é uma instância de AppConfig."
        logger.error(msg)
        raise TypeError(msg)

    try:
        config_dict = config.model_dump(mode='python')

        logger.info(f"Salvando configuração em: {config_path} usando a biblioteca toml")
        logger.debug(f"Dados de configuração a serem salvos: {config_dict}")

        backup_path = config_path.with_suffix('.bak')
        if config_path.exists():
            try:
                shutil.copy2(config_path, backup_path)
                logger.info(f"Backup do arquivo de configuração criado em: {backup_path}")
            except Exception as backup_e:
                logger.warning(f"Falha ao criar backup do arquivo de configuração: {backup_e}")

        with open(config_path, "w", encoding="utf-8") as f: 
            toml.dump(config_dict, f)

        logger.info("Configuração salva com sucesso.")

    except IOError as e:
        msg = f"Erro de I/O ao salvar o arquivo de configuração {config_path}: {e}"
        logger.error(msg, exc_info=True)
        raise ConfigurationError(msg) from e
    except Exception as e:
        msg = f"Ocorreu um erro inesperado ao salvar a configuração em {config_path}: {e}"
        logger.error(msg, exc_info=True)
        raise ConfigurationError(msg) from e

# --- Reset Section Function ---
# Mapping section names to their Pydantic model classes
_section_name_to_model = {
    "system": AppConfig.__fields__['system'].annotation,
    "ingestion": AppConfig.__fields__['ingestion'].annotation,
    "embedding": AppConfig.__fields__['embedding'].annotation,
    "vector_store": AppConfig.__fields__['vector_store'].annotation,
    "query": AppConfig.__fields__['query'].annotation,
    "llm": AppConfig.__fields__['llm'].annotation,
    "text_normalizer": AppConfig.__fields__['text_normalizer'].annotation,
}

def reset_config(config: AppConfig, section_name: str) -> None:
    """
    Resets a specific section of the AppConfig object to its default values
    AND saves the resulting configuration to the default config file.

    Args:
        config: The current AppConfig object (used as base for reset).
        section_name: The name of the attribute in AppConfig to reset (e.g., 'llm', 'embedding').

    Raises:
        ValueError: If the section_name is invalid.
        TypeError: If the config object is not an AppConfig instance.
        ConfigurationError: If saving the reset configuration fails.
    """
    if not isinstance(config, AppConfig):
        msg = "Objeto fornecido para reset não é uma instância de AppConfig."
        logger.error(msg)
        raise TypeError(msg)

    if section_name not in _section_name_to_model:
        msg = f"Nome de seção inválido para reset: '{section_name}'. Válidos: {list(_section_name_to_model.keys())}"
        logger.error(msg)
        raise ValueError(msg)

    try:
        logger.info(f"Resetando a seção '{section_name}' da configuração para os padrões.")
        ModelClass = _section_name_to_model[section_name]
        default_section_instance = ModelClass()
        updated_config = config.model_copy(update={section_name: default_section_instance})
        
        logger.info(f"Salvando configuração com a seção '{section_name}' resetada...")
        # Call save_config internally
        save_config(updated_config) 
        # No return value needed as the action includes saving
        
    except ConfigurationError: # Re-raise config errors from save_config
        raise
    except Exception as e:
        msg = f"Erro inesperado ao resetar e salvar a seção '{section_name}' da configuração: {e}"
        logger.error(msg, exc_info=True)
        raise ConfigurationError(msg) from e
