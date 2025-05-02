import sys
from pathlib import Path
import shutil 

import tomlkit
from tomlkit.items import Table, Item

from pydantic import ValidationError, BaseModel 

from .models import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__, log_domain="config_manager")

_default_config_path: Path = Path("config.toml").resolve()

# Faz o Mapping entre os nomes das seções e as classes Pydantic (pode ficar no nível do módulo)
_section_name_to_model = {
    field_name: model_info.annotation
    for field_name, model_info in AppConfig.model_fields.items()
    if isinstance(model_info.annotation, type) and issubclass(model_info.annotation, BaseModel)
}

class ConfigurationError(Exception):
    """Exceção personalizada para erros de carregamento/salvamento de configuração."""
    pass

class ConfigManager:
    """Gerencia o carregamento, salvamento, redefinição e restauração de configurações TOML."""
    
    def __init__(self, config_path: Path = _default_config_path):
        """Initializes the ConfigManager.

        Args:
            config_path: caminho para o arquivo de configuração.
                        Padrão é 'config.toml' no diretório raiz do projeto.
        """
        self.config_path = config_path.resolve()
        logger.info(f"ConfigManager inicializado para o caminho: {self.config_path}")

    def load_config(self) -> AppConfig:
        """
        Carrega a configuração do arquivo TOML (preservando comentários via tomlkit),
        valida usando Pydantic e retorna o objeto AppConfig.
        Uses the config_path defined during manager initialization.

        Returns:
            O objeto AppConfig validado.

        Raises:
            ConfigurationError: Se o arquivo não for encontrado, não puder ser analisado,
                                ou falhar na validação.
        """
        if not self.config_path.is_file():
            msg = f"Arquivo de configuração não encontrado: {self.config_path}"
            logger.error(msg)
            raise ConfigurationError(msg)

        try:
            logger.info(f"Carregando configuração de: {self.config_path} (com tomlkit)")
            with open(self.config_path, "rt", encoding="utf-8") as f:
                tomlkit_doc = tomlkit.load(f)
            
            config_data_dict = dict(tomlkit_doc) 
            
            logger.debug("Dados de configuração convertidos para dict, validando com Pydantic...")
            validated_config = AppConfig(**config_data_dict)
            logger.info("Configuração carregada e validada com sucesso.")
            return validated_config

        except tomlkit.exceptions.ParseError as e:
            msg = f"Erro ao analisar o arquivo de configuração {self.config_path} com tomlkit: {e}"
            logger.error(msg)
            raise ConfigurationError(msg) from e

        except ValidationError as e:
            logger.error(f"Falha na validação da configuração Pydantic para {self.config_path}:\n{e}")
            raise ConfigurationError(f"Falha na validação da configuração Pydantic para {self.config_path}:\n{e}") from e

        except Exception as e:
            msg = f"Ocorreu um erro inesperado ao carregar a configuração de {self.config_path}: {e}"
            logger.error(msg, exc_info=True)
            raise ConfigurationError(msg) from e

    def get_config(self) -> AppConfig:
        """
        Obtém a configuração da aplicação carregando-a do arquivo gerenciado.
        Returns:
            O objeto AppConfig validado.
        Raises:
            ConfigurationError: Se a configuração não puder ser carregada.
        """
        logger.debug(f"Chamando self.load_config() para obter a configuração de {self.config_path}")
        return self.load_config()

    def save_config(self, config: AppConfig) -> None:
        """
        Salva o objeto AppConfig fornecido no arquivo TOML gerenciado,
        preservando comentários e formatação existentes tanto quanto possível
        usando uma abordagem de atualização seção por seção.

        Args:
            config: O objeto AppConfig a ser salvo.

        Raises:
            ConfigurationError: Se ocorrer um erro durante a leitura ou escrita do arquivo.
            TypeError: Se o objeto config não for uma instância de AppConfig.
        """
        if not isinstance(config, AppConfig):
            msg = "Objeto fornecido para save_config não é uma instância de AppConfig."
            logger.error(msg)
            raise TypeError(msg)

        try:
            if not self.config_path.exists():
                msg = f"Arquivo de configuração não encontrado em {self.config_path}. Não é possível salvar."
                logger.error(msg)
                raise ConfigurationError(msg)

            # --- Backup Logic --- 
            backup_path = self.get_backup_config_path()
            try:
                shutil.copy2(self.config_path, backup_path)
                logger.info(f"Backup do arquivo de configuração criado em: {backup_path}")
            except Exception as backup_e:
                logger.warning(f"Falha ao criar backup do arquivo de configuração: {backup_e}")

            # --- Load, Update, Save Logic ---
            toml_doc: tomlkit.TOMLDocument
            try:
                with open(self.config_path, "rt", encoding="utf-8") as f:
                    toml_doc = tomlkit.load(f)
            except (tomlkit.exceptions.ParseError, IOError) as e:
                msg = f"Erro ao carregar o arquivo TOML existente {self.config_path} para salvamento: {e}. Salvamento abortado"
                logger.error(msg)
                raise ConfigurationError(msg) from e

            # Itera através das seções do AppConfig
            config_data = config.model_dump(mode='python', exclude_none=True)
            
            for section_name, section_data in config_data.items():
                if not isinstance(section_data, dict):
                    logger.warning(f"Esperava um dict para a seção '{section_name}', mas encontrei {type(section_data)}. Pulando atualização para esta seção.")
                    continue


                if section_name not in toml_doc:
                    logger.debug(f"Criando nova tabela para a seção '{section_name}' no documento TOML.")
                    toml_doc[section_name] = tomlkit.table()
                elif not isinstance(toml_doc[section_name], Table):
                     logger.warning(f"Item existente para a seção '{section_name}' não é uma tabela TOML ({type(toml_doc[section_name])}). Substituindo por uma tabela.")
                     old_trivia = None
                     if isinstance(toml_doc[section_name], Item):
                          old_trivia = toml_doc[section_name].trivia
                     toml_doc[section_name] = tomlkit.table()
                     if old_trivia:
                          toml_doc[section_name].trivia = old_trivia
                          
                toml_section_table = toml_doc[section_name]

                processed_keys_in_section = set()
                for key, new_value in section_data.items():
                    processed_keys_in_section.add(key)
                    existing_item = toml_section_table.get(key)
                    
                    update_needed = True
                    if isinstance(existing_item, Item) and not isinstance(existing_item, Table):
                         if type(existing_item.value) == type(new_value) and existing_item.value == new_value:
                             update_needed = False # Value and type same, skip update
                             logger.debug(f"Valor para '{section_name}.{key}' não alterado ('{new_value}'). Pulando.")
                    
                    if update_needed:
                        logger.debug(f"Atualizando valor para '{section_name}.{key}' para '{new_value}'.")
                        try:
                            new_item = tomlkit.item(new_value)
                            if isinstance(existing_item, Item):
                                if existing_item.trivia.comment:
                                    new_item.comment(existing_item.trivia.comment.strip())
                                new_item.trivia.indent = existing_item.trivia.indent
                                new_item.trivia.comment_ws = existing_item.trivia.comment_ws
                                
                            toml_section_table[key] = new_item
                        except Exception as item_e:
                             logger.warning(f"Não foi possível criar item TOML para '{section_name}.{key}' = {new_value}. Erro: {item_e}. Usando valor bruto (comentários perdidos).", exc_info=True)
                             toml_section_table[key] = new_value # Fallback
                
                keys_to_remove = set(toml_section_table.keys()) - processed_keys_in_section
                if keys_to_remove:
                    logger.debug(f"Removendo chaves da seção TOML '{section_name}' que não estão nos dados Pydantic: {keys_to_remove}")
                    for key_to_remove in keys_to_remove:
                        del toml_section_table[key_to_remove]
            
            logger.info(f"Salvando configuração em: {self.config_path} usando tomlkit (abordagem seção por seção)")
            with open(self.config_path, "wt", encoding="utf-8") as f:
                tomlkit.dump(toml_doc, f)

            logger.info("Configuração salva com sucesso.")

        except (IOError, tomlkit.exceptions.TOMLKitError) as e:
            msg = f"Erro de I/O ou TOMLKit ao salvar o arquivo de configuração {self.config_path}: {e}"
            logger.error(msg, exc_info=True)
            raise ConfigurationError(msg) from e
        except ConfigurationError:
            raise
        except Exception as e:
            msg = f"Ocorreu um erro inesperado ao salvar a configuração em {self.config_path}: {e}"
            logger.error(msg, exc_info=True)
            raise ConfigurationError(msg) from e

    def reset_config(self, config: AppConfig, section_name: str) -> None:
        """
        Reseta uma seção específica do objeto AppConfig para seus valores padrão
        e salva a configuração resultante no arquivo de configuração gerenciado.

        Args:
            config: O objeto AppConfig atual (usado como base para o reset).
            section_name: O nome do atributo em AppConfig a ser resetado (e.g., 'llm', 'embedding').

        Raises:
            ValueError: Se o nome da seção for inválido.
            TypeError: Se o objeto config não for uma instância de AppConfig.
            ConfigurationError: Se falhar ao salvar a configuração resetada.
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
            current_config_copy = config.model_copy(deep=True)
            updated_config = current_config_copy.model_copy(update={section_name: default_section_instance})
            
            logger.info(f"Salvando configuração com a seção '{section_name}' resetada para {self.config_path}...")
            self.save_config(updated_config) 
            
        except ConfigurationError:
            raise
        except Exception as e:
            msg = f"Erro inesperado ao resetar e salvar a seção '{section_name}' da configuração: {e}"
            logger.error(msg, exc_info=True)
            raise ConfigurationError(msg) from e

    def get_default_config_path(self) -> Path:
        """Retorna o caminho para o arquivo de configuração gerenciado por esta instância."""
        return self.config_path

    def get_backup_config_path(self) -> Path:
        """Retorna o caminho padrão para o arquivo de backup da configuração gerenciada."""
        return self.config_path.with_suffix('.bak')

    def restore_config_from_backup(self) -> bool:
        """
        Restaura o arquivo de configuração gerenciado a partir de seu backup.
        Uses the paths associated with this ConfigManager instance.

        Returns:
            True se a restauração foi bem-sucedida, False caso contrário.
        """
        backup_path = self.get_backup_config_path() 
        config_path_to_restore = self.config_path

        if not backup_path.exists():
            logger.error(f"Arquivo de backup não encontrado em {backup_path}. Não é possível restaurar {config_path_to_restore}.")
            return False

        try:
            shutil.copy2(backup_path, config_path_to_restore)
            logger.info(f"Configuração restaurada com sucesso de {backup_path} para {config_path_to_restore}.")
            return True
        except Exception as e:
            logger.error(f"Falha ao restaurar configuração de {backup_path} para {config_path_to_restore}: {e}", exc_info=True)
            return False
