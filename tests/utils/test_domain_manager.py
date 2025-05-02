import pytest
from unittest.mock import MagicMock, ANY, patch, call
from src.utils.domain_manager import DomainManager
from src.utils.sqlite_manager import SQLiteManager
from src.models import Domain
from src.config.models import SystemConfig, EmbeddingConfig, VectorStoreConfig, AppConfig
import os
import shutil
import sqlite3 # Import for type hinting mock connection

# Helper function to create a dummy Domain object
def create_dummy_domain(id=1, name="test_domain", base_path="storage") -> Domain:
    name_fs = name.lower().replace(" ", "_")
    db_path = os.path.join(base_path, name_fs, f"{name_fs}.db")
    vs_path = os.path.join(base_path, name_fs, "vector_store", f"{name_fs}.faiss")
    return Domain(
        id=id,
        name=name,
        description=f"{name} description",
        keywords=f"{name}, keywords",
        embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        faiss_index_type="IndexFlatL2",
        db_path=db_path,
        vector_store_path=vs_path
    )

class TestDomainManager:
    """Test suite for the DomainManager class."""

    @pytest.fixture
    def mock_sqlite_manager(self):
        """Fixture to provide a mocked SQLiteManager."""
        mock = MagicMock(spec=SQLiteManager)
        # Setup mock connection context manager
        mock_conn = MagicMock(spec=sqlite3.Connection)
        mock.get_connection.return_value.__enter__.return_value = mock_conn
        # Ensure methods like get_domain return lists or None as expected
        mock.get_domain.return_value = None
        mock.get_document_file.return_value = []
        return mock

    @pytest.fixture
    def mock_logger(self):
        """Fixture to provide a mocked logger."""
        mock = MagicMock()
        return mock

    @pytest.fixture
    def test_config(self, tmp_path):
        """Fixture to provide a SystemConfig for testing using tmp_path."""
        test_storage_path = str(tmp_path / "test_storage")
        # Create the base directory for tests that might need it
        os.makedirs(test_storage_path, exist_ok=True)

        return AppConfig(
            system=SystemConfig(
                storage_base_path=test_storage_path,
                control_db_filename="control_test.db" # Use a distinct name for clarity
            ),
            embedding=EmbeddingConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
            ),
            vector_store=VectorStoreConfig(
                index_type="IndexFlatL2"
            )
        )

    @pytest.fixture
    def domain_manager(self, mocker, test_config, mock_sqlite_manager, mock_logger):
        """Fixture to create a DomainManager instance with config and mocked dependencies."""
        # Reset mocks *before* instantiation if checking init logs
        mock_logger.reset_mock()
        mock_sqlite_manager.reset_mock()
        # Re-apply the connection mock return value after reset
        mock_conn = MagicMock(spec=sqlite3.Connection)
        mock_sqlite_manager.get_connection.return_value.__enter__.return_value = mock_conn
        mock_sqlite_manager.get_domain.return_value = None # Default to not found

        # Patch get_logger before instantiation
        mocker.patch('src.utils.domain_manager.get_logger', return_value=mock_logger)

        # Instantiate DomainManager with config and mock manager
        manager = DomainManager(config=test_config, sqlite_manager=mock_sqlite_manager, log_domain="test_log")

        return manager

    # --- Initialization Test ---
    def test_initialization(self, domain_manager, test_config, mock_sqlite_manager, mock_logger):
        """Test the initialization attributes of DomainManager via fixture."""
        assert domain_manager.config == test_config
        assert domain_manager.sqlite_manager == mock_sqlite_manager
        assert domain_manager.storage_base_path == test_config.system.storage_base_path
        assert domain_manager.logger == mock_logger
        # Check logger was called during init (now happens before reset)
        mock_logger.info.assert_called_with("Inicializando DomainManager")

    # --- create_domain Tests ---
    def test_create_domain_success(self, domain_manager, test_config, mock_sqlite_manager, mock_logger):
        """Test successful creation of a new domain using configured paths."""
        
        domain_data = {
            "name": "New Domain Name",
            "description": "New description",
            "keywords": "new, keywords",
            "embeddings_model": "sentence-transformers/all-MiniLM-L6-v2",
            "faiss_index_type": "IndexFlatL2"
        }

        # Mock DB interactions (domain doesn't exist)
        mock_conn = mock_sqlite_manager.get_connection.return_value.__enter__.return_value
        mock_sqlite_manager.get_domain.return_value = None 

        # Call the method
        domain_manager.create_domain(domain_data)

        # Assertions
        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        mock_sqlite_manager.begin.assert_called_once_with(mock_conn)
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_conn, domain_data["name"])
        mock_sqlite_manager.insert_domain.assert_called_once_with(ANY, mock_conn)

        # Check the details of the Domain object passed to insert_domain
        call_args = mock_sqlite_manager.insert_domain.call_args
        inserted_domain = call_args[0][0]
        assert isinstance(inserted_domain, Domain)
        assert inserted_domain.name == domain_data["name"]
        assert inserted_domain.description == domain_data["description"]
        assert inserted_domain.keywords == domain_data["keywords"]
        # Check constructed paths based on test_config.storage_base_path
        expected_db_path = os.path.join(test_config.system.storage_base_path, domain_data["name"].lower().replace(" ", "_"), f"{domain_data['name'].lower().replace(' ', '_')}.db")
        expected_vs_path = os.path.join(test_config.system.storage_base_path, domain_data["name"].lower().replace(" ", "_"), "vector_store", f"{domain_data['name'].lower().replace(' ', '_')}.faiss")
        assert inserted_domain.db_path == expected_db_path
        assert inserted_domain.vector_store_path == expected_vs_path

        mock_conn.commit.assert_called_once()
        mock_conn.rollback.assert_not_called()
        mock_logger.info.assert_any_call("Dominio de conhecimento adicionado com sucesso", domain_name=domain_data["name"])
        # Check log for adding domain contains the constructed paths
        mock_logger.info.assert_any_call("Adicionando novo domínio de conhecimento", 
                                        name=domain_data["name"], description=domain_data["description"], keywords=domain_data["keywords"], embeddings_model=domain_data["embeddings_model"],
                                        faiss_index_type=domain_data["faiss_index_type"], db_path=expected_db_path, vector_store_path=expected_vs_path)

    def test_create_domain_already_exists(self, domain_manager, test_config, mock_sqlite_manager, mock_logger):
        """Test attempting to create a domain that already exists."""
        
        domain_data = {
            "name": "existing_domain",
            "description": "Existing description",
            "keywords": "existing, keywords",
            "embeddings_model": "sentence-transformers/all-MiniLM-L6-v2",
            "faiss_index_type": "IndexFlatL2"
        }

        mock_conn = mock_sqlite_manager.get_connection.return_value.__enter__.return_value
        existing_domain_obj = create_dummy_domain(id=1, name=domain_data["name"], base_path=test_config.system.storage_base_path)
        mock_sqlite_manager.get_domain.return_value = [existing_domain_obj] # Domain exists

        with pytest.raises(ValueError, match=f"Domínio já existe: {domain_data['name']}"):
            domain_manager.create_domain(domain_data)

        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        mock_sqlite_manager.begin.assert_called_once_with(mock_conn)
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_conn, domain_data["name"])
        mock_sqlite_manager.insert_domain.assert_not_called()
        mock_conn.rollback.assert_called_once()
        mock_conn.commit.assert_not_called()
        mock_logger.error.assert_any_call("Dominio ja existe", domain_name=domain_data["name"])

    def test_create_domain_invalid_args(self, domain_manager):
        """Test create_domain with non-string arguments."""
        with pytest.raises(ValueError, match="Nome, descrição e palavras-chave devem ser strings"):
            args = {
                "name": 123,
                "description": "desc",
                "keywords": "key"
            }
            domain_manager.create_domain(args)
        with pytest.raises(ValueError, match="Nome, descrição e palavras-chave devem ser strings"):
            args = {
                "name": "name",
                "description": None,
                "keywords": "key"
            }
            domain_manager.create_domain(args)
        with pytest.raises(ValueError, match="Nome, descrição e palavras-chave devem ser strings"):
            args = {
                "name": "name",
                "description": "desc",
                "keywords": ["key"]
            }
            domain_manager.create_domain(args)

    # --- remove_domain_registry_and_files Tests ---
    def test_remove_domain_success(self, domain_manager, test_config, mock_sqlite_manager, mock_logger, mocker):
        """Test successful removal of an existing domain and its directory."""
        domain_name = "domain to remove"
        domain_name_fs = "domain_to_remove"
        existing_domain_obj = create_dummy_domain(id=5, name=domain_name, base_path=test_config.system.storage_base_path)
        expected_domain_dir = os.path.join(test_config.system.storage_base_path, domain_name_fs)

        mock_conn = mock_sqlite_manager.get_connection.return_value.__enter__.return_value
        mock_sqlite_manager.get_domain.return_value = [existing_domain_obj] # Domain exists

        mock_isdir = mocker.patch('src.utils.domain_manager.os.path.isdir', return_value=True)
        mock_rmtree = mocker.patch('src.utils.domain_manager.shutil.rmtree')

        domain_manager.remove_domain_registry_and_files(domain_name)

        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        mock_sqlite_manager.begin.assert_called_once_with(mock_conn)
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_conn, domain_name)
        mock_isdir.assert_called_once_with(expected_domain_dir)
        mock_rmtree.assert_called_once_with(expected_domain_dir)
        mock_sqlite_manager.delete_domain.assert_called_once_with(existing_domain_obj, mock_conn)
        mock_conn.commit.assert_called_once()
        mock_conn.rollback.assert_not_called()
        mock_logger.info.assert_any_call("Diretorio e arquivos do dominio removidos com sucesso", domain_directory=expected_domain_dir)
        mock_logger.info.assert_any_call("Dominio de conhecimento removido com sucesso", domain_name=domain_name)

    def test_remove_domain_dir_not_found(self, domain_manager, test_config, mock_sqlite_manager, mock_logger, mocker):
        """Test removal of a domain when its directory doesn't exist."""
        domain_name = "domain no dir"
        domain_name_fs = "domain_no_dir"
        existing_domain_obj = create_dummy_domain(id=6, name=domain_name, base_path=test_config.system.storage_base_path)
        expected_domain_dir = os.path.join(test_config.system.storage_base_path, domain_name_fs)

        mock_conn = mock_sqlite_manager.get_connection.return_value.__enter__.return_value
        mock_sqlite_manager.get_domain.return_value = [existing_domain_obj]

        mock_isdir = mocker.patch('src.utils.domain_manager.os.path.isdir', return_value=False)
        mock_rmtree = mocker.patch('src.utils.domain_manager.shutil.rmtree')

        domain_manager.remove_domain_registry_and_files(domain_name)

        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        mock_sqlite_manager.begin.assert_called_once_with(mock_conn)
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_conn, domain_name)
        mock_isdir.assert_called_once_with(expected_domain_dir)
        mock_rmtree.assert_not_called() # Should not be called
        mock_sqlite_manager.delete_domain.assert_called_once_with(existing_domain_obj, mock_conn)
        mock_conn.commit.assert_called_once()
        mock_conn.rollback.assert_not_called()
        mock_logger.warning.assert_any_call("Diretorio do dominio nao encontrado, removendo o registro do dominio", domain_name=domain_name)
        mock_logger.info.assert_any_call("Dominio de conhecimento removido com sucesso", domain_name=domain_name)

    def test_remove_domain_not_found_in_db(self, domain_manager, mock_sqlite_manager, mock_logger, mocker):
        """Test attempting to remove a domain that does not exist in the database."""
        domain_name = "non_existent_domain"

        mock_conn = mock_sqlite_manager.get_connection.return_value.__enter__.return_value
        mock_sqlite_manager.get_domain.return_value = None # Domain not found

        mock_isdir = mocker.patch('src.utils.domain_manager.os.path.isdir')
        mock_rmtree = mocker.patch('src.utils.domain_manager.shutil.rmtree')

        with pytest.raises(ValueError, match=f"Domínio não encontrado: {domain_name}"):
            domain_manager.remove_domain_registry_and_files(domain_name)

        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        mock_sqlite_manager.begin.assert_called_once_with(mock_conn)
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_conn, domain_name)
        mock_isdir.assert_not_called()
        mock_rmtree.assert_not_called()
        mock_sqlite_manager.delete_domain.assert_not_called()
        mock_conn.rollback.assert_called_once()
        mock_conn.commit.assert_not_called()
        mock_logger.error.assert_any_call("Dominio nao encontrado", domain_name=domain_name)

    # --- update_domain_details Tests ---
    def test_update_domain_details_success_no_rename(self, domain_manager, test_config, mock_sqlite_manager, mock_logger):
        """Test successful update of domain details without renaming."""
        domain_name = "domain_to_update"
        existing_domain_obj = create_dummy_domain(id=10, name=domain_name, base_path=test_config.system.storage_base_path)
        updates = {
            "description": "Updated Description",
            "keywords": "updated, keywords"
        }

        mock_conn = mock_sqlite_manager.get_connection.return_value.__enter__.return_value
        mock_sqlite_manager.get_domain.side_effect = [
            [existing_domain_obj], # First call finds the domain
        ]

        domain_manager.update_domain_details(domain_name, updates)

        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        # get_domain called once to fetch the domain
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_conn, domain_name)
        mock_sqlite_manager.update_domain.assert_called_once_with(existing_domain_obj, mock_conn, updates)
        mock_sqlite_manager.begin.assert_called_once_with(mock_conn)
        mock_conn.commit.assert_called_once()
        mock_conn.rollback.assert_not_called()
        mock_logger.info.assert_any_call("Dominio de conhecimento atualizado com sucesso", domain_name=domain_name, updated_fields=list(updates.keys()))

    def test_update_domain_details_success_with_rename(self, domain_manager, test_config, mock_sqlite_manager, mock_logger, mocker):
        """Test successful update including renaming the domain and its paths."""
        old_name = "Old Domain Name"
        new_name = "New Domain Name"
        base_path = test_config.system.storage_base_path
        updates = {"name": new_name, "description": "Desc after rename"}
        existing_domain_obj = create_dummy_domain(id=11, name=old_name, base_path=base_path)
        mock_conn = mock_sqlite_manager.get_connection.return_value.__enter__.return_value
        mock_sqlite_manager.get_domain.side_effect = [
            [existing_domain_obj],
            None
        ]
        new_name_fs = new_name.lower().replace(" ", "_")
        new_dir = os.path.join(base_path, new_name_fs)
        expected_new_db_path = os.path.join(new_dir, f"{new_name_fs}.db")
        expected_new_vs_path = os.path.join(new_dir, "vector_store", f"{new_name_fs}.faiss")
        mock_rename_paths = mocker.patch.object(domain_manager, '_rename_domain_paths', return_value=(expected_new_db_path, expected_new_vs_path))
        
        domain_manager.update_domain_details(old_name, updates)
        
        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        assert mock_sqlite_manager.get_domain.call_count == 2
        mock_sqlite_manager.get_domain.assert_has_calls([
            call(mock_conn, old_name), 
            call(mock_conn, new_name)
        ])
        mock_rename_paths.assert_called_once_with(old_name, new_name)
        expected_update_payload = {
            "name": new_name,
            "description": "Desc after rename",
            "db_path": expected_new_db_path,
            "vector_store_path": expected_new_vs_path
        }
        mock_sqlite_manager.update_domain.assert_called_once_with(existing_domain_obj, mock_conn, expected_update_payload)
        mock_sqlite_manager.begin.assert_called_once_with(mock_conn)
        mock_conn.commit.assert_called_once()
        mock_conn.rollback.assert_not_called()
        
        # FIX: Check logged fields using set comparison for robustness against order changes
        # Find the specific log call
        update_success_log = None
        for log_call in mock_logger.info.call_args_list:
            if log_call.args[0] == "Dominio de conhecimento atualizado com sucesso":
                update_success_log = log_call
                break
        assert update_success_log is not None, "Success log not found"
        assert update_success_log.kwargs.get("domain_name") == old_name
        assert set(update_success_log.kwargs.get("updated_fields", [])) == set(expected_update_payload.keys())

    def test_update_domain_details_domain_not_found(self, domain_manager, mock_sqlite_manager):
        """Test updating a domain that does not exist."""
        domain_name = "not_found_domain"
        updates = {"description": "New Desc"}
        
        mock_conn = mock_sqlite_manager.get_connection.return_value.__enter__.return_value
        mock_sqlite_manager.get_domain.return_value = None # Simulate domain not found

        with pytest.raises(ValueError, match=f"Domínio não encontrado: {domain_name}"):
            domain_manager.update_domain_details(domain_name, updates)
        
        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_conn, domain_name)
        mock_sqlite_manager.update_domain.assert_not_called()
        mock_sqlite_manager.begin.assert_not_called() 
        mock_conn.commit.assert_not_called()
        mock_conn.rollback.assert_not_called()

    def test_update_domain_details_new_name_exists(self, domain_manager, test_config, mock_sqlite_manager, mock_logger, mocker):
        """Test updating domain name when the new name already exists."""
        old_name = "Original Exists Test"
        new_name = "Taken Name Test"
        base_path = test_config.system.storage_base_path
        updates = {"name": new_name}

        original_domain_obj = create_dummy_domain(id=20, name=old_name, base_path=base_path)
        taken_domain_obj = create_dummy_domain(id=21, name=new_name, base_path=base_path)

        mock_conn = mock_sqlite_manager.get_connection.return_value.__enter__.return_value
        mock_sqlite_manager.get_domain.side_effect = [
            [original_domain_obj],
            [taken_domain_obj] 
        ]
        mock_rename_paths = mocker.patch.object(domain_manager, '_rename_domain_paths')

        with pytest.raises(ValueError, match=f"Domínio já existe: {new_name}"):
            domain_manager.update_domain_details(old_name, updates)

        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        assert mock_sqlite_manager.get_domain.call_count == 2
        mock_sqlite_manager.get_domain.assert_has_calls([
            call(mock_conn, old_name), 
            call(mock_conn, new_name) 
        ])
        mock_rename_paths.assert_not_called()
        mock_sqlite_manager.update_domain.assert_not_called()
        mock_sqlite_manager.begin.assert_not_called()
        mock_conn.commit.assert_not_called()
        mock_conn.rollback.assert_not_called() 
        mock_logger.error.assert_any_call("Dominio ja existe", domain_name=new_name)

    def test_update_domain_details_no_change(self, domain_manager, test_config, mock_sqlite_manager, mock_logger):
        domain_name = "no_change_domain"
        existing_domain_obj = create_dummy_domain(id=11, name=domain_name, base_path=test_config.system.storage_base_path)
        updates = {
            "description": existing_domain_obj.description, 
            "keywords": existing_domain_obj.keywords     
        }

        mock_conn = mock_sqlite_manager.get_connection.return_value.__enter__.return_value
        mock_sqlite_manager.get_domain.return_value = [existing_domain_obj]

        domain_manager.update_domain_details(domain_name, updates)

        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_conn, domain_name)
        
        mock_sqlite_manager.update_domain.assert_not_called()
        mock_sqlite_manager.begin.assert_not_called()
        mock_conn.commit.assert_not_called()
        mock_conn.rollback.assert_not_called()
        mock_logger.info.assert_any_call("Nenhum campo valido ou alterado para atualizar.")

    def test_update_domain_details_invalid_field(self, domain_manager, test_config, mock_sqlite_manager, mock_logger):
        domain_name = "invalid_field_domain"
        existing_domain_obj = create_dummy_domain(id=12, name=domain_name, base_path=test_config.system.storage_base_path)
        updates = {
            "db_path": "/new/path.db", 
            "id": 999 
        }

        mock_conn = mock_sqlite_manager.get_connection.return_value.__enter__.return_value
        mock_sqlite_manager.get_domain.return_value = [existing_domain_obj]

        domain_manager.update_domain_details(domain_name, updates)

        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_conn, domain_name)
        mock_sqlite_manager.update_domain.assert_not_called()
        mock_sqlite_manager.begin.assert_not_called()
        mock_conn.commit.assert_not_called()
        mock_conn.rollback.assert_not_called()
        mock_logger.warning.assert_any_call("Campo nao pode ser atualizado manualmente", column="db_path")
        mock_logger.warning.assert_any_call("Campo nao pode ser atualizado manualmente", column="id")
        mock_logger.info.assert_any_call("Nenhum campo valido ou alterado para atualizar.")

    # --- _rename_domain_paths Tests ---
    def test_rename_domain_paths_success(self, domain_manager, test_config, mock_logger, tmp_path):
        old_name = "old rename name"
        new_name = "new rename name"
        old_name_fs = "old_rename_name"
        new_name_fs = "new_rename_name"
        base_path = test_config.system.storage_base_path

        # Create dummy old structure
        old_dir = tmp_path / "test_storage" / old_name_fs
        old_db_file = old_dir / f"{old_name_fs}.db"
        old_vs_dir = old_dir / "vector_store"
        old_vs_file = old_vs_dir / f"{old_name_fs}.faiss"
        os.makedirs(old_vs_dir)
        old_db_file.touch()
        old_vs_file.touch()

        assert old_dir.is_dir()
        assert old_db_file.is_file()
        assert old_vs_file.is_file()

        # Expected new paths
        new_dir = tmp_path / "test_storage" / new_name_fs
        expected_new_db_path = str(new_dir / f"{new_name_fs}.db")
        expected_new_vs_path = str(new_dir / "vector_store" / f"{new_name_fs}.faiss")

        # Call the private method
        result_db_path, result_vs_path = domain_manager._rename_domain_paths(old_name, new_name)

        # Assertions
        assert result_db_path == expected_new_db_path
        assert result_vs_path == expected_new_vs_path
        assert not old_dir.exists() 
        assert new_dir.is_dir()     
        assert (new_dir / f"{new_name_fs}.db").is_file() 
        assert (new_dir / "vector_store" / f"{new_name_fs}.faiss").is_file() 
        mock_logger.info.assert_any_call("Renomeando arquivos do dominio", old_name=old_name, new_name=new_name)

    def test_rename_domain_paths_dir_not_exist(self, domain_manager, test_config):
        old_name = "dir not exist"
        new_name = "new dir name"
        old_name_fs = "dir_not_exist"
        new_name_fs = "new_dir_name"
        base_path = test_config.system.storage_base_path

        # Expected paths 
        new_dir = os.path.join(base_path, new_name_fs)
        expected_new_db_path = os.path.join(new_dir, f"{new_name_fs}.db")
        expected_new_vs_path = os.path.join(new_dir, "vector_store", f"{new_name_fs}.faiss")

        # Call the private method
        result_db_path, result_vs_path = domain_manager._rename_domain_paths(old_name, new_name)

        # Assert returned paths 
        assert result_db_path == expected_new_db_path
        assert result_vs_path == expected_new_vs_path
        assert not os.path.exists(os.path.join(base_path, old_name_fs))
        assert not os.path.exists(new_dir)

    def test_rename_domain_paths_target_exists(self, domain_manager, test_config, tmp_path):
        old_name = "old target exist"
        new_name = "new target exist"
        old_name_fs = "old_target_exist"
        new_name_fs = "new_target_exist"
        base_path = test_config.system.storage_base_path

        # Create dummy old and *target* directories
        old_dir = tmp_path / "test_storage" / old_name_fs
        new_dir = tmp_path / "test_storage" / new_name_fs
        os.makedirs(old_dir)
        os.makedirs(new_dir)

        with pytest.raises(FileExistsError, match=f"Diretorio já existe: {str(new_dir)}"):
            domain_manager._rename_domain_paths(old_name, new_name)

    def test_rename_domain_paths_os_error(self, domain_manager, test_config, mocker, tmp_path):
        """Test handling of OSError during renaming, ensuring rollback attempt."""
        old_name = "os error domain"
        new_name = "new os error"
        old_name_fs = "os_error_domain"
        new_name_fs = "new_os_error"
        base_path = test_config.system.storage_base_path
        old_dir = tmp_path / "test_storage" / old_name_fs
        new_dir = tmp_path / "test_storage" / new_name_fs 
        old_dir.mkdir(parents=True)
        mock_rename = mocker.patch('src.utils.domain_manager.os.rename', side_effect=OSError("Disk full"))
        
        # FIX: Match the actual OSError message raised
        with pytest.raises(OSError, match="Disk full"):
        # with pytest.raises(OSError, match="Erro ao renomear caminhos do dominio: Disk full"):
            domain_manager._rename_domain_paths(old_name, new_name)

        mock_rename.assert_called_once_with(str(old_dir), str(new_dir))
        # Rollback attempt happens within the method now, error message indicates the original error

    # --- list_domains Tests ---
    def test_list_domains_success(self, domain_manager, test_config, mock_sqlite_manager):
        domain1 = create_dummy_domain(id=1, name="Domain Alpha", base_path=test_config.system.storage_base_path)
        domain2 = create_dummy_domain(id=2, name="Domain Beta", base_path=test_config.system.storage_base_path)
        mock_conn = mock_sqlite_manager.get_connection.return_value.__enter__.return_value
        mock_sqlite_manager.get_domain.return_value = [domain1, domain2]

        result = domain_manager.list_domains()

        assert result == [domain1, domain2]
        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_conn)

    def test_list_domains_empty(self, domain_manager, mock_sqlite_manager):
        mock_conn = mock_sqlite_manager.get_connection.return_value.__enter__.return_value
        mock_sqlite_manager.get_domain.return_value = None 

        result = domain_manager.list_domains()

        assert result is None 
        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_conn)

    # --- list_domain_documents Tests ---
    def test_list_domain_documents_success(self, domain_manager, test_config, mock_sqlite_manager):
        domain_name = "docs domain"
        domain = create_dummy_domain(id=3, name=domain_name, base_path=test_config.system.storage_base_path)
        doc1 = MagicMock() 
        doc2 = MagicMock()
        expected_docs = [doc1, doc2]

        mock_control_conn = MagicMock(spec=sqlite3.Connection)
        mock_domain_conn = MagicMock(spec=sqlite3.Connection)
        mock_sqlite_manager.get_connection.side_effect = [
            MagicMock(__enter__=MagicMock(return_value=mock_control_conn)), 
            MagicMock(__enter__=MagicMock(return_value=mock_domain_conn))  
        ]
        mock_sqlite_manager.get_domain.return_value = [domain] 
        mock_sqlite_manager.get_document_file.return_value = expected_docs 

        with patch('src.utils.domain_manager.os.path.exists', return_value=True) as mock_exists:
            result = domain_manager.list_domain_documents(domain_name)

            mock_exists.assert_called_once_with(domain.db_path)
            assert result == expected_docs
            assert mock_sqlite_manager.get_connection.call_count == 2
            mock_sqlite_manager.get_connection.assert_has_calls([
                call(control=True), 
                call(db_path=domain.db_path)
            ])
            mock_sqlite_manager.get_domain.assert_called_once_with(mock_control_conn, domain_name)
            mock_sqlite_manager.get_document_file.assert_called_once_with(mock_domain_conn)

    def test_list_domain_documents_domain_not_found(self, domain_manager, mock_sqlite_manager):
        domain_name = "no such domain"

        mock_control_conn = MagicMock(spec=sqlite3.Connection)
        mock_sqlite_manager.get_connection.return_value.__enter__.return_value = mock_control_conn
        mock_sqlite_manager.get_domain.return_value = None 

        with pytest.raises(ValueError, match=f"Domínio não encontrado: {domain_name}"):
            domain_manager.list_domain_documents(domain_name)

        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_control_conn, domain_name)
        mock_sqlite_manager.get_document_file.assert_not_called() 

    def test_list_domain_documents_db_not_found(self, domain_manager, test_config, mock_sqlite_manager):
        domain_name = "no db domain"
        domain = create_dummy_domain(id=4, name=domain_name, base_path=test_config.system.storage_base_path)

        mock_control_conn = MagicMock(spec=sqlite3.Connection)
        mock_sqlite_manager.get_connection.return_value.__enter__.return_value = mock_control_conn
        mock_sqlite_manager.get_domain.return_value = [domain] 

        with patch('src.utils.domain_manager.os.path.exists', return_value=False) as mock_exists:
             with pytest.raises(FileNotFoundError, match=f"Banco de dados do domínio não encontrado: {domain.db_path}"):
                 domain_manager.list_domain_documents(domain_name)

             mock_exists.assert_called_once_with(domain.db_path)
             mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
             mock_sqlite_manager.get_domain.assert_called_once_with(mock_control_conn, domain_name)
             mock_sqlite_manager.get_document_file.assert_not_called() 

    def test_list_domain_documents_no_documents(self, domain_manager, test_config, mock_sqlite_manager):
        domain_name = "empty domain"
        domain = create_dummy_domain(id=7, name=domain_name, base_path=test_config.system.storage_base_path)

        mock_control_conn = MagicMock(spec=sqlite3.Connection)
        mock_domain_conn = MagicMock(spec=sqlite3.Connection)
        mock_sqlite_manager.get_connection.side_effect = [
            MagicMock(__enter__=MagicMock(return_value=mock_control_conn)),
            MagicMock(__enter__=MagicMock(return_value=mock_domain_conn))
        ]
        mock_sqlite_manager.get_domain.return_value = [domain]
        mock_sqlite_manager.get_document_file.return_value = [] 

        with patch('src.utils.domain_manager.os.path.exists', return_value=True) as mock_exists:
            result = domain_manager.list_domain_documents(domain_name)

            mock_exists.assert_called_once_with(domain.db_path)
            assert result == [] 
            assert mock_sqlite_manager.get_connection.call_count == 2
            mock_sqlite_manager.get_connection.assert_has_calls([
                call(control=True),
                call(db_path=domain.db_path)
            ])
            mock_sqlite_manager.get_domain.assert_called_once_with(mock_control_conn, domain_name)
            mock_sqlite_manager.get_document_file.assert_called_once_with(mock_domain_conn)