import pytest
from unittest.mock import MagicMock, ANY, patch
from src.utils.domain_manager import DomainManager
from src.utils.sqlite_manager import SQLiteManager
from src.models import Domain
import os
import shutil

class TestDomainManager:
    """Test suite for the DomainManager class."""

    @pytest.fixture
    def mock_sqlite_manager(self):
        """Fixture to provide a mocked SQLiteManager."""
        mock = MagicMock(spec=SQLiteManager)
        return mock

    @pytest.fixture
    def mock_logger(self):
        """Fixture to provide a mocked logger."""
        mock = MagicMock()
        # Make logger methods chainable or return None if needed
        mock.info.return_value = None
        mock.debug.return_value = None
        mock.warning.return_value = None
        mock.error.return_value = None
        return mock

    @pytest.fixture
    def domain_manager(self, mocker, mock_sqlite_manager, mock_logger):
        """Fixture to create a DomainManager instance with mocked dependencies."""
        mocker.patch('src.utils.domain_manager.get_logger', return_value=mock_logger)
        mocker.patch('src.utils.domain_manager.SQLiteManager', return_value=mock_sqlite_manager)
        manager = DomainManager()
        # Reset mocks for clean state in each test using this fixture
        mock_logger.reset_mock()
        mock_sqlite_manager.reset_mock()
        return manager

    def test_initialization(self, mocker, mock_sqlite_manager, mock_logger):
        """Test the initialization of DomainManager."""
        
        # Patch the dependencies (SQLiteManager and get_logger)
        mock_get_logger = mocker.patch('src.utils.domain_manager.get_logger', return_value=mock_logger)
        mock_sqlite_manager_class = mocker.patch('src.utils.domain_manager.SQLiteManager', return_value=mock_sqlite_manager)
        
        # Instantiate the DomainManager
        domain_manager = DomainManager(log_domain="test_log")
        
        # Assertions
        assert domain_manager is not None
        # Check if get_logger was called correctly
        mock_get_logger.assert_called_once_with('src.utils.domain_manager', 'test_log')
        # Check if the logger instance was assigned
        assert domain_manager.logger == mock_logger
        # Check if SQLiteManager was instantiated
        mock_sqlite_manager_class.assert_called_once_with()
        # Check if the sqlite_manager instance was assigned
        assert domain_manager.sqlite_manager == mock_sqlite_manager
        # Check logger initialization message (optional but good)
        mock_logger.info.assert_called_with("Inicializando DomainManager")

    def test_create_domain_success(self, domain_manager, mock_sqlite_manager, mock_logger):
        """Test successful creation of a new domain."""
        domain_name = "new_domain"
        description = "New description"
        keywords = "new, keywords"

        # Mock the connection context manager
        mock_conn = MagicMock()
        mock_sqlite_manager.get_connection.return_value.__enter__.return_value = mock_conn

        # Mock DB interactions
        mock_sqlite_manager.get_domain.return_value = None # Simulate domain doesn't exist

        # Call the method
        domain_manager.create_domain(domain_name, description, keywords)

        # Assertions
        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        mock_sqlite_manager.begin.assert_called_once_with(mock_conn)
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_conn, domain_name)
        
        # Assert insert_domain was called with a Domain object
        # We use ANY from unittest.mock because the actual Domain object is created inside the method
        mock_sqlite_manager.insert_domain.assert_called_once_with(ANY, mock_conn)

        # Check the details of the Domain object passed to insert_domain
        call_args = mock_sqlite_manager.insert_domain.call_args
        inserted_domain = call_args[0][0] # First positional argument of the first call
        assert isinstance(inserted_domain, Domain)
        assert inserted_domain.name == domain_name
        assert inserted_domain.description == description
        assert inserted_domain.keywords == keywords
        # Check constructed paths
        expected_db_path = os.path.join("storage", "domains", domain_name, f"{domain_name}.db")
        expected_vs_path = os.path.join("storage", "domains", domain_name, "vector_store", f"{domain_name}.faiss")
        assert inserted_domain.db_path == expected_db_path
        assert inserted_domain.vector_store_path == expected_vs_path

        mock_conn.commit.assert_called_once()
        mock_conn.rollback.assert_not_called()
        mock_logger.info.assert_any_call("Domínio de conhecimento adicionado com sucesso", domain_name=domain_name)

    def test_create_domain_already_exists(self, domain_manager, mock_sqlite_manager, mock_logger):
        """Test attempting to create a domain that already exists."""
        domain_name = "existing_domain"
        description = "Existing description"
        keywords = "existing, keywords"

        # Mock the connection context manager
        mock_conn = MagicMock()
        mock_sqlite_manager.get_connection.return_value.__enter__.return_value = mock_conn

        # Mock get_domain to return an existing domain
        # Create a dummy Domain object to return
        existing_domain_obj = Domain(
            id=1, name=domain_name, description="old desc", keywords="old keys",
            db_path="old/path.db", vector_store_path="old/vector.faiss"
        )
        mock_sqlite_manager.get_domain.return_value = [existing_domain_obj] # get_domain returns a list

        # Call the method and assert ValueError
        with pytest.raises(ValueError, match=f"Domínio já existe: {domain_name}"):
            domain_manager.create_domain(domain_name, description, keywords)

        # Assertions
        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        mock_sqlite_manager.begin.assert_called_once_with(mock_conn)
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_conn, domain_name)
        
        # Assert insert_domain was NOT called
        mock_sqlite_manager.insert_domain.assert_not_called()

        # Assert rollback WAS called
        mock_conn.rollback.assert_called_once()
        # Assert commit was NOT called
        mock_conn.commit.assert_not_called()
        mock_logger.error.assert_any_call("Domínio já existe", domain_name=domain_name)

    def test_remove_domain_success(self, domain_manager, mock_sqlite_manager, mock_logger, mocker):
        """Test successful removal of an existing domain and its directory."""
        domain_name = "domain_to_remove"
        # Create a dummy Domain object to be returned by get_domain
        existing_domain_obj = Domain(
            id=5, name=domain_name, description="desc", keywords="keys",
            db_path=os.path.join("storage", "domains", domain_name, f"{domain_name}.db"), 
            vector_store_path=os.path.join("storage", "domains", domain_name, "vector_store", f"{domain_name}.faiss")
        )
        expected_domain_dir = os.path.join("storage", "domains", domain_name)

        # Mock the connection context manager
        mock_conn = MagicMock()
        mock_sqlite_manager.get_connection.return_value.__enter__.return_value = mock_conn

        # Mock DB interactions
        mock_sqlite_manager.get_domain.return_value = [existing_domain_obj] # Domain exists

        # Mock filesystem interactions
        mock_isdir = mocker.patch('src.utils.domain_manager.os.path.isdir', return_value=True)
        mock_rmtree = mocker.patch('src.utils.domain_manager.shutil.rmtree')

        # Call the method
        domain_manager.remove_domain_registry_and_files(domain_name)

        # Assertions
        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        mock_sqlite_manager.begin.assert_called_once_with(mock_conn)
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_conn, domain_name)
        
        # Check filesystem mocks
        mock_isdir.assert_called_once_with(expected_domain_dir)
        mock_rmtree.assert_called_once_with(expected_domain_dir)

        # Check DB delete was called with the correct domain object
        mock_sqlite_manager.delete_domain.assert_called_once_with(existing_domain_obj, mock_conn)
        
        mock_conn.commit.assert_called_once()
        mock_conn.rollback.assert_not_called()
        mock_logger.info.assert_any_call("Diretório e arquivos do domínio removidos com sucesso", domain_name=domain_name)
        mock_logger.info.assert_any_call("Domínio de conhecimento removido com sucesso", domain_name=domain_name)

    def test_remove_domain_dir_not_found(self, domain_manager, mock_sqlite_manager, mock_logger, mocker):
        """Test successful removal of a domain when its directory doesn't exist."""
        domain_name = "domain_no_dir"
        existing_domain_obj = Domain(
            id=6, name=domain_name, description="desc", keywords="keys",
            db_path=os.path.join("storage", "domains", domain_name, f"{domain_name}.db"), 
            vector_store_path=os.path.join("storage", "domains", domain_name, "vector_store", f"{domain_name}.faiss")
        )
        expected_domain_dir = os.path.join("storage", "domains", domain_name)

        mock_conn = MagicMock()
        mock_sqlite_manager.get_connection.return_value.__enter__.return_value = mock_conn
        mock_sqlite_manager.get_domain.return_value = [existing_domain_obj]

        # Mock filesystem checks - directory does NOT exist
        mock_isdir = mocker.patch('src.utils.domain_manager.os.path.isdir', return_value=False)
        mock_rmtree = mocker.patch('src.utils.domain_manager.shutil.rmtree')

        domain_manager.remove_domain_registry_and_files(domain_name)

        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        mock_sqlite_manager.begin.assert_called_once_with(mock_conn)
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_conn, domain_name)
        
        # Check filesystem mocks
        mock_isdir.assert_called_once_with(expected_domain_dir)
        mock_rmtree.assert_not_called() # Should not be called

        # Check DB delete was called
        mock_sqlite_manager.delete_domain.assert_called_once_with(existing_domain_obj, mock_conn)
        
        mock_conn.commit.assert_called_once()
        mock_conn.rollback.assert_not_called()
        # Check for the specific warning log
        mock_logger.warning.assert_any_call("Diretório do domínio não encontrado, removendo o registro do domínio", domain_name=domain_name)
        mock_logger.info.assert_any_call("Domínio de conhecimento removido com sucesso", domain_name=domain_name)

    def test_remove_domain_not_found_in_db(self, domain_manager, mock_sqlite_manager, mock_logger, mocker):
        """Test attempting to remove a domain that does not exist in the database."""
        domain_name = "non_existent_domain"

        mock_conn = MagicMock()
        mock_sqlite_manager.get_connection.return_value.__enter__.return_value = mock_conn

        # Mock DB interactions - domain not found
        mock_sqlite_manager.get_domain.return_value = None 

        # Mock filesystem checks (shouldn't be called)
        mock_isdir = mocker.patch('src.utils.domain_manager.os.path.isdir')
        mock_rmtree = mocker.patch('src.utils.domain_manager.shutil.rmtree')

        # Call the method and assert ValueError
        with pytest.raises(ValueError, match=f"Domínio não encontrado: {domain_name}"):
            domain_manager.remove_domain_registry_and_files(domain_name)

        # Assertions
        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        mock_sqlite_manager.begin.assert_called_once_with(mock_conn)
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_conn, domain_name)
        
        # Filesystem and DB delete should NOT be called
        mock_isdir.assert_not_called()
        mock_rmtree.assert_not_called()
        mock_sqlite_manager.delete_domain.assert_not_called()
        
        # Rollback should be called, commit should not
        mock_conn.rollback.assert_called_once()
        mock_conn.commit.assert_not_called()
        mock_logger.error.assert_any_call("Domínio não encontrado", domain_name=domain_name) 

    def test_update_domain_details_success_no_rename(self, domain_manager, mock_sqlite_manager, mock_logger):
        """Test successfully updating domain details without renaming."""
        domain_name = "existing_domain"
        updates = {
            "description": "Updated Description",
            "keywords": "updated, key, words",
            "total_documents": 50 # This should be ignored as it's not directly updatable this way
        }
        
        # Existing domain data
        existing_domain_obj = Domain(
            id=10, name=domain_name, description="Old Desc", keywords="old, keys", 
            db_path="path/to/old.db", vector_store_path="path/to/old.faiss", total_documents=10
        )
        
        mock_conn = MagicMock()
        mock_sqlite_manager.get_connection.return_value.__enter__.return_value = mock_conn
        mock_sqlite_manager.get_domain.return_value = [existing_domain_obj] # Domain exists

        # Call the method
        domain_manager.update_domain_details(domain_name, updates)

        # Assertions
        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_conn, domain_name)
        
        # Check that update_domain was called with the correct filtered updates
        expected_update_payload = {
            "description": "Updated Description",
            "keywords": "updated, key, words",
            "total_documents": 50 # Corrected: total_documents IS updatable and different
        }
        mock_sqlite_manager.update_domain.assert_called_once_with(existing_domain_obj, mock_conn, expected_update_payload)
        
        mock_sqlite_manager.begin.assert_called_once_with(mock_conn)
        mock_conn.commit.assert_called_once()
        mock_conn.rollback.assert_not_called()
        mock_logger.info.assert_any_call("Domínio de conhecimento atualizado com sucesso", domain_name=domain_name)

    def test_update_domain_details_success_with_rename(self, domain_manager, mock_sqlite_manager, mock_logger, mocker):
        """Test successfully updating domain name, triggering path rename."""
        old_name = "old_domain_name"
        new_name = "new_domain_name"
        updates = {"name": new_name, "description": "Desc after rename"}

        existing_domain_obj = Domain(
            id=11, name=old_name, description="Old Desc", keywords="keys", 
            db_path=os.path.join("storage", "domains", old_name, f"{old_name}.db"), 
            vector_store_path=os.path.join("storage", "domains", old_name, "vector_store", f"{old_name}.faiss")
        )
        
        # Mock connection and initial get_domain for the old name
        mock_conn = MagicMock()
        mock_sqlite_manager.get_connection.return_value.__enter__.return_value = mock_conn
        # Use side_effect to handle multiple calls to get_domain
        # 1st call (check old_name exists): return [existing_domain_obj]
        # 2nd call (check new_name doesn't exist): return None
        mock_sqlite_manager.get_domain.side_effect = [
            [existing_domain_obj], # First call finds the old domain
            None                 # Second call confirms new name is free
        ]

        # Mock the internal rename_domain_paths method
        new_db_path = os.path.join("storage", "domains", new_name, f"{new_name}.db")
        new_vs_path = os.path.join("storage", "domains", new_name, "vector_store", f"{new_name}.faiss")
        mock_rename = mocker.patch.object(domain_manager, 'rename_domain_paths', return_value=(new_db_path, new_vs_path))

        # Call the method
        domain_manager.update_domain_details(old_name, updates)

        # Assertions
        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        # Check get_domain calls
        assert mock_sqlite_manager.get_domain.call_count == 2
        mock_sqlite_manager.get_domain.assert_any_call(mock_conn, old_name)
        mock_sqlite_manager.get_domain.assert_any_call(mock_conn, new_name)
        
        # Check rename was called
        mock_rename.assert_called_once_with(old_name, new_name)

        # Check update_domain call payload includes new name and paths
        expected_update_payload = {
            "name": new_name,
            "description": "Desc after rename",
            "db_path": new_db_path,
            "vector_store_path": new_vs_path
        }
        mock_sqlite_manager.update_domain.assert_called_once_with(existing_domain_obj, mock_conn, expected_update_payload)

        mock_sqlite_manager.begin.assert_called_once_with(mock_conn)
        mock_conn.commit.assert_called_once()
        mock_conn.rollback.assert_not_called()
        mock_logger.info.assert_any_call("Domínio de conhecimento atualizado com sucesso", domain_name=old_name)

    def test_update_domain_details_domain_not_found(self, domain_manager, mock_sqlite_manager, mock_logger):
        """Test updating a domain that does not exist."""
        domain_name = "ghost_domain"
        updates = {"description": "New desc"}
        
        mock_conn = MagicMock()
        mock_sqlite_manager.get_connection.return_value.__enter__.return_value = mock_conn
        mock_sqlite_manager.get_domain.return_value = None # Domain doesn't exist

        # Call the method and assert ValueError
        with pytest.raises(ValueError, match=f"Domínio não encontrado: {domain_name}"):
            domain_manager.update_domain_details(domain_name, updates)
        
        # Assertions
        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_conn, domain_name)
        mock_sqlite_manager.begin.assert_not_called()
        mock_sqlite_manager.update_domain.assert_not_called()
        mock_conn.commit.assert_not_called()
        mock_conn.rollback.assert_not_called() # No transaction started
        mock_logger.error.assert_any_call("Domínio não encontrado", domain_name=domain_name)

    def test_update_domain_details_new_name_exists(self, domain_manager, mock_sqlite_manager, mock_logger, mocker):
        """Test renaming a domain to a name that already exists."""
        old_name = "original_domain"
        new_name = "taken_domain_name"
        updates = {"name": new_name}

        # Add required path fields
        original_domain_obj = Domain(
            id=20, name=old_name, description="Original", keywords="orig",
            db_path=f"path/{old_name}.db", vector_store_path=f"path/{old_name}.faiss"
        )
        taken_domain_obj = Domain(
            id=21, name=new_name, description="Taken", keywords="taken",
            db_path=f"path/{new_name}.db", vector_store_path=f"path/{new_name}.faiss"
        )

        mock_conn = MagicMock()
        mock_sqlite_manager.get_connection.return_value.__enter__.return_value = mock_conn
        # 1st call (find original): return [original_domain_obj]
        # 2nd call (find new name): return [taken_domain_obj]
        mock_sqlite_manager.get_domain.side_effect = [
            [original_domain_obj],
            [taken_domain_obj]
        ]

        mock_rename = mocker.patch.object(domain_manager, 'rename_domain_paths')

        # Call the method and assert ValueError
        with pytest.raises(ValueError, match=f"Domínio já existe: {new_name}"):
            domain_manager.update_domain_details(old_name, updates)

        # Assertions
        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        assert mock_sqlite_manager.get_domain.call_count == 2
        mock_sqlite_manager.get_domain.assert_any_call(mock_conn, old_name)
        mock_sqlite_manager.get_domain.assert_any_call(mock_conn, new_name)
        
        # Rename and update should not happen
        mock_rename.assert_not_called()
        mock_sqlite_manager.begin.assert_not_called()
        mock_sqlite_manager.update_domain.assert_not_called()
        mock_conn.commit.assert_not_called()
        mock_conn.rollback.assert_not_called()
        mock_logger.error.assert_any_call("Domínio já existe", domain_name=new_name)

    def test_update_domain_details_no_change(self, domain_manager, mock_sqlite_manager, mock_logger):
        """Test updating with no actual changes or only non-updatable fields."""
        domain_name = "stable_domain"
        updates = {
            "description": "Same Description", # Same as existing
            "keywords": "same, keys",         # Same as existing
            "db_path": "new/path.db"         # Not updatable
        }
        
        existing_domain_obj = Domain(
            id=30, name=domain_name, description="Same Description", keywords="same, keys",
            db_path="original/path.db", vector_store_path="original/path.faiss"
        )
        
        mock_conn = MagicMock()
        mock_sqlite_manager.get_connection.return_value.__enter__.return_value = mock_conn
        mock_sqlite_manager.get_domain.return_value = [existing_domain_obj]

        # Call the method
        domain_manager.update_domain_details(domain_name, updates)

        # Assertions
        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_conn, domain_name)
        
        # Begin and Update should NOT be called as no valid changes were detected
        mock_sqlite_manager.begin.assert_not_called()
        mock_sqlite_manager.update_domain.assert_not_called()
        mock_conn.commit.assert_not_called()
        mock_conn.rollback.assert_not_called()
        mock_logger.warning.assert_any_call("Campo não pode ser atualizado manualmente", column="db_path")
        # Should not log success if no update happened
        # Let's check that the success log was NOT called
        success_log_call = mock_logger.info.call_args_list
        assert all(
            call.args[0] != "Domínio de conhecimento atualizado com sucesso" 
            for call in success_log_call
        ), "Success log should not be called when no update occurs" 

    def test_rename_domain_paths_success(self, domain_manager, mock_logger, mocker):
        """Test successful renaming of all domain paths."""
        old_name = "old_fragrant_domain"
        new_name = "new_shiny_domain"
        
        # Expected paths
        old_dir = os.path.join("storage", "domains", old_name)
        new_dir = os.path.join("storage", "domains", new_name)
        old_db = os.path.join(new_dir, f"{old_name}.db") # Note: Uses new_dir after dir rename
        new_db = os.path.join(new_dir, f"{new_name}.db")
        old_faiss = os.path.join(new_dir, "vector_store", f"{old_name}.faiss")
        new_faiss = os.path.join(new_dir, "vector_store", f"{new_name}.faiss")

        # Mock filesystem operations
        mock_exists = mocker.patch('src.utils.domain_manager.os.path.exists', return_value=True)
        mock_rename = mocker.patch('src.utils.domain_manager.os.rename')

        # Call the method
        result_db_path, result_faiss_path = domain_manager.rename_domain_paths(old_name, new_name)

        # Assertions
        # Check existence checks were performed
        assert mock_exists.call_count == 3
        mock_exists.assert_any_call(old_dir)
        mock_exists.assert_any_call(old_db)
        mock_exists.assert_any_call(old_faiss)

        # Check rename calls were performed correctly
        assert mock_rename.call_count == 3
        mock_rename.assert_any_call(old_dir, new_dir)
        mock_rename.assert_any_call(old_db, new_db)
        mock_rename.assert_any_call(old_faiss, new_faiss)

        # Check returned paths
        assert result_db_path == new_db
        assert result_faiss_path == new_faiss
        mock_logger.error.assert_not_called() # No errors expected

    def test_rename_domain_paths_some_not_found(self, domain_manager, mock_logger, mocker):
        """Test renaming when exactly one file (faiss) doesn't exist."""
        old_name = "old_partial_domain"
        new_name = "new_partial_domain"

        old_dir = os.path.join("storage", "domains", old_name)
        new_dir = os.path.join("storage", "domains", new_name)
        old_db = os.path.join(new_dir, f"{old_name}.db") 
        new_db = os.path.join(new_dir, f"{new_name}.db")
        old_faiss = os.path.join(new_dir, "vector_store", f"{old_name}.faiss")
        new_faiss = os.path.join(new_dir, "vector_store", f"{new_name}.faiss")

        # Mock filesystem - Dir and DB exist, Faiss does not
        def exists_side_effect(path):
            if path == old_dir: return True
            if path == old_db: return True
            if path == old_faiss: return False
            # Need to mock exists for the potential rollback check on new_dir
            if path == new_dir: return True 
            return False # Default
        mock_exists = mocker.patch('src.utils.domain_manager.os.path.exists', side_effect=exists_side_effect)
        mock_rename = mocker.patch('src.utils.domain_manager.os.rename')
        # Mock isdir for the rollback check
        mock_isdir = mocker.patch('src.utils.domain_manager.os.path.isdir', return_value=True)

        # Call the method
        result_db_path, result_faiss_path = domain_manager.rename_domain_paths(old_name, new_name)

        # Assertions
        # os.path.exists calls: old_dir, old_db, old_faiss
        assert mock_exists.call_count == 3 
        mock_exists.assert_any_call(old_dir)
        mock_exists.assert_any_call(old_db)
        mock_exists.assert_any_call(old_faiss)

        # Check only dir and db rename were initially attempted
        # +1 for the rollback attempt triggered by missing_files == 1
        assert mock_rename.call_count == 3 
        mock_rename.assert_any_call(old_dir, new_dir)       # Initial dir rename
        mock_rename.assert_any_call(old_db, new_db)         # Initial db rename
        mock_rename.assert_any_call(new_dir, old_dir)       # Rollback dir rename
        
        # Check isdir was called for rollback
        mock_isdir.assert_called_once_with(new_dir)

        # Check returned paths are still the target new paths
        assert result_db_path == new_db
        assert result_faiss_path == new_faiss

        # Check for logs
        # Initial warning for missing faiss
        mock_logger.warning.assert_any_call(f"Arquivo .faiss {old_faiss} não encontrado, pulando renomeação do vectorstore.")
        # New critical log for exactly one missing file
        mock_logger.critical.assert_any_call(
            f"Alerta! Inconsistência encontrada no sistema de arquivos. {old_faiss} não existe. Remova o domínio e seus arquivos.",
            domain_name=old_name,
            missing_files=[old_faiss]
        )
        # Info log for rollback attempt
        mock_logger.info.assert_any_call("Tentativa de reverter renomeação do diretório.")
        mock_logger.error.assert_not_called()

    def test_rename_domain_paths_os_error_with_rollback(self, domain_manager, mock_logger, mocker):
        """Test handling of OSError during rename, triggering rollback attempt."""
        old_name = "old_fail_domain"
        new_name = "new_fail_domain"

        old_dir = os.path.join("storage", "domains", old_name)
        new_dir = os.path.join("storage", "domains", new_name)
        # Define specific paths for side_effect checking
        db_path_old_in_new = os.path.join(new_dir, f"{old_name}.db")
        db_path_new_in_new = os.path.join(new_dir, f"{new_name}.db")

        # Mock filesystem
        mock_exists = mocker.patch('src.utils.domain_manager.os.path.exists', return_value=True)
        # Refine side_effect to only fail on the specific db rename
        def rename_side_effect(src, dst):
            if src == old_dir and dst == new_dir:
                print(f"Mock rename: {src} -> {dst} (SUCCESS)") # Debug print
                return # Allow directory rename
            elif src == db_path_old_in_new and dst == db_path_new_in_new:
                print(f"Mock rename: {src} -> {dst} (FAILING)") # Debug print
                raise OSError("Permission denied") # Fail DB rename
            elif src == new_dir and dst == old_dir:
                 print(f"Mock rename: {src} -> {dst} (ROLLBACK SUCCESS)") # Debug print
                 return # Allow rollback rename
            else:
                 # Optional: Raise error for unexpected calls
                 raise ValueError(f"Unexpected call to os.rename: {src} -> {dst}")
                 
        mock_rename = mocker.patch('src.utils.domain_manager.os.rename', side_effect=rename_side_effect)
        mock_isdir = mocker.patch('src.utils.domain_manager.os.path.isdir', return_value=True)

        # Call the method and assert OSError
        with pytest.raises(OSError, match="Permission denied"):
            domain_manager.rename_domain_paths(old_name, new_name)

        # Assertions
        # Check rename calls: dir success, db fail, rollback success
        assert mock_rename.call_count == 3
        mock_rename.assert_any_call(old_dir, new_dir)
        mock_rename.assert_any_call(db_path_old_in_new, db_path_new_in_new)
        mock_rename.assert_any_call(new_dir, old_dir) # Rollback attempt
        
        mock_isdir.assert_called_once_with(new_dir)
        mock_logger.error.assert_any_call(f"Erro ao renomear caminhos do domínio: Permission denied")
        # Now this info log should be called
        mock_logger.info.assert_any_call("Tentativa de reverter renomeação do diretório.") 
        # Ensure the inner exception log was NOT called
        assert not any(
            call.args[0].startswith("Falha ao reverter renomeação do diretório") 
            for call in mock_logger.error.call_args_list
        ) 

    def test_list_domains(self, domain_manager, mock_sqlite_manager, mock_logger):
        """Test listing all domains successfully."""
        # Create dummy domains to be returned
        domain1 = Domain(id=1, name="domain1", description="Desc 1", keywords="k1", db_path="p1", vector_store_path="v1")
        domain2 = Domain(id=2, name="domain2", description="Desc 2", keywords="k2", db_path="p2", vector_store_path="v2")
        expected_domains = [domain1, domain2]

        # Mock connection and DB call
        mock_conn = MagicMock()
        mock_sqlite_manager.get_connection.return_value.__enter__.return_value = mock_conn
        mock_sqlite_manager.get_domain.return_value = expected_domains

        # Call the method
        result = domain_manager.list_domains()

        # Assertions
        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        # get_domain should be called to list all
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_conn)
        assert result == expected_domains
        mock_logger.info.assert_any_call("Listando domínios de conhecimento")

    def test_list_domains_empty(self, domain_manager, mock_sqlite_manager, mock_logger):
        """Test listing domains when none exist."""
        # Mock connection and DB call returning None
        mock_conn = MagicMock()
        mock_sqlite_manager.get_connection.return_value.__enter__.return_value = mock_conn
        mock_sqlite_manager.get_domain.return_value = None

        # Call the method
        result = domain_manager.list_domains()

        # Assertions
        mock_sqlite_manager.get_connection.assert_called_once_with(control=True)
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_conn)
        assert result is None # Should return None if get_domain returns None
        mock_logger.info.assert_any_call("Listando domínios de conhecimento")

    def test_list_domain_documents_success(self, domain_manager, mock_sqlite_manager, mock_logger):
        """Test listing documents for an existing domain successfully."""
        domain_name = "doc_domain"
        domain_db_path = f"path/to/{domain_name}.db"
        existing_domain = Domain(id=40, name=domain_name, description="Desc", keywords="k", db_path=domain_db_path, vector_store_path="vpath")
        expected_documents = [
            (1, 'doc1.txt', 'Processed', '2023-01-01', 100, 10),
            (2, 'doc2.pdf', 'Pending', '2023-01-02', 200, 20)
        ]

        # Mock control connection and get_domain
        mock_control_conn = MagicMock()
        mock_domain_conn_success = MagicMock() # Explicit mock for the domain connection
        mock_sqlite_manager.get_connection.return_value.__enter__.side_effect = [mock_control_conn, mock_domain_conn_success] # Use the explicit mock
        mock_sqlite_manager.get_domain.return_value = [existing_domain]

        # Mock domain-specific get_document_file
        mock_sqlite_manager.get_document_file.return_value = expected_documents

        # Call the method
        result = domain_manager.list_domain_documents(domain_name)

        # Assertions
        # Check control connection calls
        mock_sqlite_manager.get_connection.assert_any_call(control=True)
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_control_conn, domain_name)

        # Check domain connection call
        mock_sqlite_manager.get_connection.assert_any_call(db_path=domain_db_path, control=False)
        # Check get_document_file call (uses the second mock connection object)
        mock_sqlite_manager.get_document_file.assert_called_once_with(mock_domain_conn_success)

        assert result == expected_documents
        mock_logger.info.assert_any_call("Listando documentos do domínio de conhecimento", domain_name=domain_name)

    def test_list_domain_documents_no_documents(self, domain_manager, mock_sqlite_manager, mock_logger):
        """Test listing documents when the domain exists but has no documents."""
        domain_name = "empty_doc_domain"
        domain_db_path = f"path/to/{domain_name}.db"
        existing_domain = Domain(id=41, name=domain_name, description="Desc", keywords="k", db_path=domain_db_path, vector_store_path="vpath")

        # Mock control connection and get_domain
        mock_control_conn = MagicMock()
        mock_domain_conn = MagicMock()
        mock_sqlite_manager.get_connection.return_value.__enter__.side_effect = [mock_control_conn, mock_domain_conn]
        mock_sqlite_manager.get_domain.return_value = [existing_domain]

        # Mock get_document_file returning None (or empty list, adjust if needed based on SQLiteManager)
        mock_sqlite_manager.get_document_file.return_value = None 

        # Call the method
        result = domain_manager.list_domain_documents(domain_name)

        # Assertions
        mock_sqlite_manager.get_connection.assert_any_call(control=True)
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_control_conn, domain_name)
        mock_sqlite_manager.get_connection.assert_any_call(db_path=domain_db_path, control=False)
        mock_sqlite_manager.get_document_file.assert_called_once_with(mock_domain_conn)

        assert result is None # Expecting None if get_document_file returns None
        mock_logger.info.assert_any_call("Listando documentos do domínio de conhecimento", domain_name=domain_name)
        # Optionally check for a specific log if no docs are found, e.g.,
        # mock_logger.info.assert_any_call("Nenhum documento encontrado para o domínio", domain_name=domain_name)

    def test_list_domain_documents_domain_not_found(self, domain_manager, mock_sqlite_manager, mock_logger):
        """Test listing documents when the specified domain does not exist."""
        domain_name = "non_existent_domain"

        # Mock control connection and get_domain returning None
        mock_control_conn = MagicMock()
        mock_sqlite_manager.get_connection.return_value.__enter__.return_value = mock_control_conn
        mock_sqlite_manager.get_domain.return_value = None # Domain not found

        # Call the method and assert ValueError
        with pytest.raises(ValueError, match=f"Domínio não encontrado: {domain_name}"):
            domain_manager.list_domain_documents(domain_name)

        # Assertions
        mock_sqlite_manager.get_connection.assert_called_once_with(control=True) # Only control connection
        mock_sqlite_manager.get_domain.assert_called_once_with(mock_control_conn, domain_name)

        # Domain-specific calls should NOT happen
        mock_sqlite_manager.get_document_file.assert_not_called()
        mock_logger.error.assert_any_call(f"Erro ao listar documentos do domínio de conhecimento: Domínio não encontrado: {domain_name}")