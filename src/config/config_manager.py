import sys
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

# Usa tomli para Python < 3.11
if sys.version_info < (3, 11):
    import tomli
else:
    import tomllib as tomli

from .models import AppConfig
# Assumindo que o logger pode ser usado depois
# from src.utils.logger import get_logger

# Variável global para cachear a configuração carregada
_cached_config: Optional[AppConfig] = None
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
    # logger = get_logger(__name__) # Adicionar logger se necessário

    if not config_path.is_file():
        msg = f"Arquivo de configuração não encontrado: {config_path}"
        # logger.error(msg)
        print(f"ERRO: {msg}") # Print simples por enquanto
        raise ConfigurationError(msg)

    try:
        # logger.info(f"Carregando configuração de: {config_path}")
        print(f"INFO: Carregando configuração de: {config_path}")
        with open(config_path, "rb") as f:
            config_data = tomli.load(f)

        # logger.debug("Dados brutos de configuração carregados, validando...")
        validated_config = AppConfig(**config_data)
        # logger.info("Configuração carregada e validada com sucesso.")
        print("INFO: Configuração carregada e validada com sucesso.")
        return validated_config

    except tomli.TOMLDecodeError as e:
        msg = f"Erro ao analisar o arquivo de configuração {config_path}: {e}"
        # logger.error(msg)
        print(f"ERRO: {msg}")
        raise ConfigurationError(msg) from e

    except ValidationError as e:
        msg = f"Falha na validação da configuração para {config_path}:\n{e}"
        # logger.error(msg)
        print(f"ERRO: {msg}")
        raise ConfigurationError(msg) from e

    except Exception as e:
        msg = f"Ocorreu um erro inesperado ao carregar a configuração de {config_path}: {e}"
        # logger.error(msg, exc_info=True)
        print(f"ERRO: {msg}")
        raise ConfigurationError(msg) from e

def get_config(force_reload: bool = False) -> AppConfig:
    """
    Obtém a configuração da aplicação, carregando-a se necessário.

    Usa uma versão em cache a menos que force_reload seja True.

    Args:
        force_reload: Se True, recarrega a configuração do arquivo
                      mesmo que já esteja em cache.

    Returns:
        O objeto AppConfig validado.

    Raises:
        ConfigurationError: Se a configuração não puder ser carregada.
    """
    global _cached_config
    if _cached_config is None or force_reload:
        # logger = get_logger(__name__) # Adicionar logger se necessário
        if force_reload:
            # logger.info("Forçando recarregamento da configuração.")
            print("INFO: Forçando recarregamento da configuração.")
        _cached_config = load_config()

    return _cached_config

# Opcional: Adicionar função save_config depois se necessário para a GUI
# def save_config(config: AppConfig, config_path: Path = _config_path):
#     ... 