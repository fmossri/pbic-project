import pytest
from pathlib import Path
import toml # For reading back saved files in tests
import sys

# Import the module itself to modify its global variable
import src.config.config_manager as config_manager_module

# Conditionally import the correct reader based on Python version
if sys.version_info < (3, 11):
    import tomli
else:
    import tomllib as tomli

from src.config.models import AppConfig
from src.config.config_manager import load_config, get_config, save_config, ConfigurationError, _config_path

# --- Helper Data ---

def get_valid_config_dict():
    """Returns a dictionary representing a valid AppConfig structure."""
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
            "model_name": "test-model",
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

# --- Test Class ---

class TestConfigManager:

    # Reset cache before each test
    @pytest.fixture(autouse=True)
    def reset_cache(self):
        global _cached_config
        _cached_config = None

    # --- Tests for load_config ---

    def test_load_config_success(self, tmp_path):
        """Test successfully loading a valid config file."""
        config_file = tmp_path / "valid_config.toml"
        valid_data = get_valid_config_dict()
        with open(config_file, "w", encoding="utf-8") as f:
            toml.dump(valid_data, f)

        loaded_config = load_config(config_file)
        assert isinstance(loaded_config, AppConfig)
        assert loaded_config.system.storage_base_path == "data/test_storage"
        assert loaded_config.embedding.model_name == "test-model"
        assert loaded_config.text_normalizer.use_lowercase is True

    def test_load_config_file_not_found(self, tmp_path):
        """Test loading when the config file doesn't exist."""
        non_existent_file = tmp_path / "not_a_config.toml"
        with pytest.raises(ConfigurationError, match="Arquivo de configuração não encontrado"):
            load_config(non_existent_file)

    def test_load_config_invalid_toml(self, tmp_path):
        """Test loading a file with invalid TOML syntax."""
        config_file = tmp_path / "invalid_syntax.toml"
        config_file.write_text("this is not valid toml syntax [") # Invalid TOML

        with pytest.raises(ConfigurationError, match="Erro ao analisar o arquivo"):
            load_config(config_file)

    def test_load_config_validation_error_invalid_type_in_subsection(self, tmp_path):
        """Test loading a config with an invalid type in a subsection."""
        config_file = tmp_path / "invalid_type_sub.toml"
        invalid_data = get_valid_config_dict()
        invalid_data["system"]["storage_base_path"] = 12345 # Invalid type (int instead of str)
        
        with open(config_file, "w", encoding="utf-8") as f:
            toml.dump(invalid_data, f)

        with pytest.raises(ConfigurationError, match="Falha na validação"):
            load_config(config_file)

    def test_load_config_validation_error_wrong_type(self, tmp_path):
        """Test loading a config with a field of the wrong type."""
        config_file = tmp_path / "wrong_type.toml"
        invalid_data = get_valid_config_dict()
        invalid_data["ingestion"]["chunk_size"] = "not-an-integer" # Wrong type
        with open(config_file, "w", encoding="utf-8") as f:
            toml.dump(invalid_data, f)

        with pytest.raises(ConfigurationError, match="Falha na validação"):
            load_config(config_file)

    # --- Tests for get_config ---

    def test_get_config_calls_load_config(self, tmp_path, mocker):
        """Test that get_config simply calls load_config."""
        config_file = tmp_path / "get_config_test.toml"
        valid_data = get_valid_config_dict()
        with open(config_file, "w", encoding="utf-8") as f:
            toml.dump(valid_data, f)
        
        mock_load = mocker.patch("src.config.config_manager.load_config", wraps=load_config)
        
        # Override the default path to use our temp file
        original_path = config_manager_module._config_path
        try:
            config_manager_module._config_path = config_file
            config_obj = get_config()
            mock_load.assert_called_once()
            assert isinstance(config_obj, AppConfig)
        finally:
             config_manager_module._config_path = original_path

    # --- Tests for save_config ---

    def test_save_config_success(self, tmp_path):
        """Test successfully saving a valid AppConfig object."""
        config_file = tmp_path / "save_config.toml"
        valid_dict = get_valid_config_dict()
        config_to_save = AppConfig(**valid_dict)

        save_config(config_to_save, config_file)

        # Verify file content
        assert config_file.is_file()
        # with open(config_file, \"rb\") as f: # Use reading mode consistent with loader
        #     saved_data = tomli.load(f) 
        
        # Load the saved file back into an AppConfig object
        reloaded_config = load_config(config_file)
        
        # Compare the AppConfig objects directly (uses Pydantic's __eq__)
        # assert saved_data == config_to_save.model_dump(mode='python')
        assert reloaded_config == config_to_save
        
        # Spot check a nested value on the reloaded object
        # assert saved_data['llm']['temperature'] == 0.7
        assert reloaded_config.llm.temperature == 0.7

    def test_save_config_type_error(self, tmp_path):
        """Test calling save_config with a non-AppConfig object."""
        config_file = tmp_path / "save_type_error.toml"
        not_an_appconfig = {"system": {"log_level": "INFO"}} # Just a dict

        with pytest.raises(TypeError, match="não é uma instância de AppConfig"):
            save_config(not_an_appconfig, config_file)

    def test_save_config_io_error(self, tmp_path, mocker):
        """Test save_config when an IOError occurs during file writing."""
        config_file = tmp_path / "save_io_error.toml"
        valid_dict = get_valid_config_dict()
        config_to_save = AppConfig(**valid_dict)

        # Mock the built-in open function to raise IOError
        mock_open = mocker.patch("builtins.open", side_effect=IOError("Disk full"))

        with pytest.raises(ConfigurationError, match="Erro de I/O ao salvar"):
            save_config(config_to_save, config_file)

        # Ensure mock_open was actually called with the correct path (optional check)
        # Note: Need to handle how builtins.open is used (binary vs text)
        # For toml.dump (text mode):
        # mock_open.assert_called_once_with(config_file, "w", encoding="utf-8")

    # --- (Tests for get_config and save_config will be added next) --- 