import pytest
import tomlkit

from typing import Optional, Dict
from pydantic import BaseModel

from src.config.config_manager import ConfigManager, ConfigurationError, _section_name_to_model
from src.config.models import AppConfig

# --- Dados Auxiliares ---

def get_valid_config_dict():
    """Retorna um dicionário representando uma estrutura AppConfig válida."""
    return {
        "system": {
            "storage_base_path": "data/test_storage",
            "control_db_filename": "control.db"
        },
        "ingestion": {
            "chunk_strategy": "recursive",
            "chunk_size": 1000,
            "chunk_overlap": 200
        },
        "embedding": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 32,
            "normalize_embeddings": True
        },
        "vector_store": {
            "index_type": "IndexFlatL2"
        },
        "query": {
            "retrieval_k": 5
        },
        "llm": {
            "model_repo_id": "test-llm-repo",
            "prompt_template": "Context: {context}\\nQuestion: {query}\\nAnswer:",
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "max_retries": 3,
            "retry_delay_seconds": 2
        },
        "text_normalizer": {
            "use_unicode_normalization": True,
            "use_lowercase": True,
            "use_remove_extra_whitespace": True
        }
    }

# --- Classe de Teste para Métodos de Instância ConfigManager ---

class TestConfigManager:

    @pytest.fixture
    def manager(self, tmp_path) -> ConfigManager:
        """Fornece uma instância ConfigManager isolada para um diretório temporário."""
        config_file = tmp_path / "test_config.toml"
        return ConfigManager(config_path=config_file)
    

    @pytest.fixture
    def tomlkit_manager(self, tmp_path) -> ConfigManager:
        """Fornece uma instância ConfigManager isolada para um diretório temporário
           com conteúdo inicial tipo AppConfig, incluindo comentários."""
        config_file = tmp_path / "update_test.toml"
        initial_content = '''\
# Comentário principal config
[system] # Comentário seção system
storage_base_path = "data/original" # Comentário path
control_db_filename = "control.db"

[embedding]
model_name = "sentence-transformers/all-MiniLM-L6-v2" 
# Comentário device
device = "cuda"

[llm]
temperature = 0.7 # Temp inicial
# top_p está inicialmente ausente/None
'''
        config_file.write_text(initial_content, encoding="utf-8")
        return ConfigManager(config_path=config_file)


    # --- Testes para load_config ---

    def test_load_config_success(self, manager: ConfigManager):
        """Testa carregar com sucesso um arquivo de configuração válido."""
        config_file = manager.config_path # Usa o path do manager
        valid_data = get_valid_config_dict()
        with open(config_file, "w", encoding="utf-8") as f:
            tomlkit.dump(valid_data, f)

        loaded_config = manager.load_config()
        assert isinstance(loaded_config, AppConfig)
        assert loaded_config.system.storage_base_path == "data/test_storage"
        assert loaded_config.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert loaded_config.text_normalizer.use_lowercase is True

        # Verifica o conteúdo do arquivo
        assert config_file.is_file()

        reloaded_config = manager.load_config()
        assert reloaded_config == loaded_config
        assert reloaded_config.llm.temperature == 0.7

    def test_load_config_file_not_found(self, manager: ConfigManager):
        """Testa carregar quando o arquivo de configuração não existe."""
        # O manager é inicializado com um path inexistente do tmp_path
        assert not manager.config_path.exists()
        with pytest.raises(ConfigurationError, match="Arquivo de configuração não encontrado"):
            manager.load_config()

    def test_load_config_invalid_toml(self, manager: ConfigManager):
        """Testa carregar um arquivo com sintaxe TOML inválida."""
        config_file = manager.config_path
        config_file.write_text("isso não é uma sintaxe toml válida [") # TOML Inválido

        with pytest.raises(ConfigurationError, match="Erro ao analisar o arquivo"):
            manager.load_config()

    def test_load_config_validation_error_invalid_type_in_subsection(self, manager: ConfigManager):
        """Testa carregar uma configuração com um tipo inválido em uma subseção."""
        config_file = manager.config_path
        invalid_data = get_valid_config_dict()
        invalid_data["system"]["storage_base_path"] = 12345 # Tipo inválido
        
        with open(config_file, "w", encoding="utf-8") as f:
            tomlkit.dump(invalid_data, f)

        with pytest.raises(ConfigurationError, match="Falha na validação"):
            manager.load_config()

    def test_load_config_validation_error_wrong_type(self, manager: ConfigManager):
        """Testa carregar uma configuração com um campo do tipo errado."""
        config_file = manager.config_path
        invalid_data = get_valid_config_dict()
        invalid_data["ingestion"]["chunk_size"] = "não-é-inteiro" # Tipo errado
        with open(config_file, "w", encoding="utf-8") as f:
            tomlkit.dump(invalid_data, f)

        with pytest.raises(ConfigurationError, match="Falha na validação"):
            manager.load_config()

    # --- Testes para get_config ---

    def test_get_config_calls_load_config(self, manager: ConfigManager, mocker):
        """Testa que manager.get_config chama manager.load_config."""
        # Espiona o método load_config da instância
        mock_load = mocker.spy(manager, "load_config")
        
        try:
            # Garante que o arquivo existe para que load_config não levante FileNotFoundError
            config_file = manager.config_path
            valid_data = get_valid_config_dict()
            with open(config_file, "w", encoding="utf-8") as f:
                tomlkit.dump(valid_data, f)
                
            manager.get_config() 
            mock_load.assert_called_once_with() 
        except Exception as e:
            pytest.fail(f"get_config ou asserção do mock falhou inesperadamente: {e}")

    # --- Testes para save_config ---

    def test_save_config_success(self, manager: ConfigManager):
        """Testa salvar com sucesso um objeto AppConfig válido e recarregá-lo."""
        config_file = manager.config_path
        valid_dict = get_valid_config_dict()
        config_to_save = AppConfig(**valid_dict)

        # Salva usando a instância do manager (opera em manager.config_path)
        # Garante que o arquivo existe antes de salvar, já que save_config agora exige isso.
        config_file.touch() 
        manager.save_config(config_to_save)

        # Verifica o conteúdo do arquivo
        assert config_file.is_file()

        saved_content_raw = config_file.read_text(encoding="utf-8")
        expected_line = 'storage_base_path = "data/test_storage"'
        assert expected_line in saved_content_raw, f"Conteúdo do arquivo não contém '{expected_line}'"

        # Recarrega usando a mesma instância do manager
        reloaded_config = manager.load_config()
        
        assert isinstance(reloaded_config, AppConfig)
        assert reloaded_config.llm.temperature == config_to_save.llm.temperature
        assert reloaded_config.system.storage_base_path == config_to_save.system.storage_base_path

    def test_save_config_preserves_comments_and_updates_values(self, manager: ConfigManager):
        """Testa que salvar a config atualiza valores enquanto preserva comentários."""
        config_file = manager.config_path
        initial_content = '''\
[system]
storage_base_path = "data/original_storage" # Mantenha este comentário
control_db_filename = "control_orig.db" 

[embedding] # Comentário seção
# Comentário modelo
model_name = "sentence-transformers/all-MiniLM-L6-v2"
device = "cuda"
'''
        config_file.write_text(initial_content, encoding="utf-8")

        loaded_config = manager.load_config()

        loaded_config.system.storage_base_path = "data/MODIFIED_storage"
        loaded_config.embedding.device = "cpu"
        loaded_config.embedding.batch_size = 64

        manager.save_config(loaded_config)

        saved_content = config_file.read_text(encoding="utf-8")

        assert "# Mantenha este comentário" in saved_content
        assert "# Comentário seção" in saved_content
        assert "# Comentário modelo" in saved_content
        assert 'storage_base_path = "data/MODIFIED_storage"' in saved_content 
        assert 'control_db_filename = "control_orig.db"' in saved_content 
        assert 'model_name = "sentence-transformers/all-MiniLM-L6-v2"' in saved_content 
        assert 'device = "cpu"' in saved_content 
        assert 'batch_size = 64' in saved_content 

    def test_save_config_type_error(self, manager: ConfigManager):
        """Testa chamar save_config com um objeto que não é AppConfig."""
        not_an_appconfig = {"system": {"log_level": "INFO"}} 
        with pytest.raises(TypeError, match="não é uma instância de AppConfig"):
            manager.save_config(not_an_appconfig)

    def test_save_config_io_error(self, manager: ConfigManager, mocker):
        """Testa save_config quando ocorre um IOError durante a escrita/dump final."""
        # Garante que o arquivo existe primeiro, para que a verificação inicial em save_config passe.
        manager.config_path.touch()
        
        config_to_save = AppConfig(**get_valid_config_dict())

        mock_dump = mocker.patch("src.config.config_manager.tomlkit.dump", side_effect=IOError("Disco cheio"))
        
        # O match espera o erro levantado durante a tentativa de escrita.
        with pytest.raises(ConfigurationError, match="Erro de I/O ou TOMLKit ao salvar"):
            manager.save_config(config_to_save)
        
        # Verifica que o mock foi chamado
        mock_dump.assert_called_once()

    # --- Testes para reset_config ---

    def test_reset_config_section_success_and_preserves_comments(self, manager: ConfigManager):
        """Testa que resetar uma seção a atualiza para os padrões e preserva comentários."""
        config_file = manager.config_path
        initial_content = '''\
[system]
# Comentário system
storage_base_path = "data/custom_storage" 
control_db_filename = "custom.db"

[llm] # Comentário seção LLM
# Prompt customizado
prompt_template = "My custom prompt: {context}"
temperature = 0.99 # Temp customizada
'''
        config_file.write_text(initial_content, encoding="utf-8")

        initial_config = manager.load_config()
        assert initial_config.llm.temperature == 0.99

        # Reseta a seção 'llm' usando a instância do manager
        manager.reset_config(initial_config, ['llm'])
        
        # Recarrega e verifica o conteúdo do arquivo
        saved_content = config_file.read_text(encoding="utf-8")
        reloaded_config_after_reset = manager.load_config()
        
        assert "# Comentário system" in saved_content
        assert "# Comentário seção LLM" in saved_content
        assert "# Prompt customizado" in saved_content
        assert 'storage_base_path = "data/custom_storage"' in saved_content
        
        default_llm_config = _section_name_to_model['llm']() # Obtém instância padrão
        assert reloaded_config_after_reset.llm.temperature == default_llm_config.temperature
        # Verifica a representação string para robustez
        assert f'temperature = {default_llm_config.temperature}' in saved_content or f'temperature = {default_llm_config.temperature:.1f}' in saved_content
        # Modifica asserção para comparar com novas linhas escapadas como salvas pelo tomlkit
        expected_prompt_line = f'prompt_template = "{default_llm_config.prompt_template}"'.replace('\n', '\\n')
        assert expected_prompt_line in saved_content

    def test_reset_config_invalid_section(self, manager: ConfigManager):
        """Testa reset_config com um nome de seção inválido."""
        config_file = manager.config_path
        valid_dict = get_valid_config_dict()
        config_obj = AppConfig(**valid_dict)
        config_file.write_text(tomlkit.dumps(valid_dict)) # Precisa de um arquivo para a chamada potencial de save

        with pytest.raises(ValueError, match="Nome de seção inválido para reset"):
            manager.reset_config(config_obj, 'invalid_section_name')

    def test_reset_config_invalid_config_type(self, manager: ConfigManager):
        """Testa reset_config com um tipo de objeto de configuração inválido."""
        not_an_appconfig = {"test": 1}
        # Nenhum arquivo necessário, deve falhar na verificação de tipo antes de salvar
        with pytest.raises(TypeError, match="não é uma instância de AppConfig"):
            manager.reset_config(not_an_appconfig, 'llm')


    def test_get_default_config_path_method(self, manager: ConfigManager):
        """Testa o método de instância get_default_config_path."""
        assert manager.get_default_config_path() == manager.config_path

    def test_get_backup_config_path_method(self, manager: ConfigManager):
        """Testa o método de instância get_backup_config_path."""
        expected_path = manager.config_path.with_suffix('.bak')
        assert manager.get_backup_config_path() == expected_path

    def test_restore_config_from_backup_success(self, manager: ConfigManager):
        """Testa restaurar com sucesso de um arquivo de backup usando o manager."""
        config_file = manager.config_path
        backup_file = manager.get_backup_config_path()

        config_content = "value = 1"
        backup_content = "value = 0 # Conteúdo backup" 

        config_file.write_text(config_content)
        backup_file.write_text(backup_content)

        assert config_file.read_text() == config_content

        # Realiza a restauração usando a instância do manager
        restored = manager.restore_config_from_backup()

        assert restored is True
        assert config_file.read_text() == backup_content

    def test_restore_config_from_backup_no_backup_file(self, manager: ConfigManager):
        """Testa restaurar quando o arquivo de backup não existe usando o manager."""
        config_file = manager.config_path
        backup_file = manager.get_backup_config_path()

        config_content = "value = 1"
        config_file.write_text(config_content)

        assert not backup_file.exists()

        restored = manager.restore_config_from_backup()

        assert restored is False
        assert config_file.read_text() == config_content

    def test_update_scalar_value_preserves_comments(self, tomlkit_manager: ConfigManager):
        """Testa que atualizar um escalar existente preserva comentários ao redor."""
        config = tomlkit_manager.load_config()
        config.system.storage_base_path = "data/UPDATED" 
        tomlkit_manager.save_config(config)
        
        doc_str = tomlkit_manager.config_path.read_text()
        assert 'storage_base_path = "data/UPDATED"' in doc_str
        assert "# Comentário path" in doc_str
        assert "# Comentário seção system" in doc_str
        assert 'control_db_filename = "control.db"' in doc_str

    def test_add_optional_scalar_key(self, tomlkit_manager: ConfigManager):
        """Testa adicionar/atualizar uma chave escalar que tem um padrão Pydantic
           mas pode estar ausente no arquivo TOML inicial."""
        config = tomlkit_manager.load_config()
        
        # Verifica que Pydantic carregou o valor padrão mesmo se ausente no arquivo inicial
        assert config.llm.top_p == 0.9 
        
        # Agora muda o valor do padrão
        config.llm.top_p = 0.88 
        tomlkit_manager.save_config(config)
        
        doc_str = tomlkit_manager.config_path.read_text()
        assert "top_p = 0.88" in doc_str
        assert 'temperature = 0.7' in doc_str

    def test_update_nested_scalar_value_preserves_comments(self, tomlkit_manager: ConfigManager):
        """Testa que atualizar um escalar em uma seção aninhada preserva comentários."""
        config = tomlkit_manager.load_config()
        config.embedding.device = "cpu"
        tomlkit_manager.save_config(config)
        
        doc_str = tomlkit_manager.config_path.read_text()
        assert 'device = "cpu"' in doc_str
        assert "# Comentário device" in doc_str
        assert 'model_name = "sentence-transformers/all-MiniLM-L6-v2"' in doc_str