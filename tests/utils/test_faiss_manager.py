import os
import pytest
import numpy as np
import faiss
import time
from pathlib import Path
from unittest.mock import patch

from src.utils import FaissManager
from src.config.models import AppConfig, VectorStoreConfig, QueryConfig

# Define standard embedding dimension for tests
TEST_DIMENSION = 64 

class TestFaissManager:
    """Test suite for FaissManager class."""
    
    @pytest.fixture(scope="class")
    def test_dir(self):
        test_run_dir = Path(f"tests/temp_faiss_{int(time.time() * 1000)}")
        test_run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nCreated test directory: {test_run_dir}")
        yield test_run_dir
        print(f"\nCleaning up test directory: {test_run_dir}")
        for item in test_run_dir.iterdir():
            if item.is_file():
                try: item.unlink()
                except OSError as e: print(f"Error removing file {item}: {e}")
        try: test_run_dir.rmdir()
        except OSError as e: print(f"Error removing directory {test_run_dir}: {e}")

    @pytest.fixture(scope="function") 
    def index_path(self, test_dir, request):
        test_name = request.node.name
        return str(test_dir / f"{test_name}_{os.getpid()}.faiss")

    @pytest.fixture(scope="class")
    def app_config(self):
        return AppConfig(
            vector_store=VectorStoreConfig(index_type="IndexFlatL2"),
            query=QueryConfig(retrieval_k=5)
        )

    @pytest.fixture(scope="function")
    def faiss_manager(self, app_config):
        """Fixture to create a FaissManager instance."""
        return FaissManager(
            config=app_config,
            log_domain="test_faiss"
        )
    
    @pytest.fixture
    def sample_embeddings(self):
        count = 5
        embeddings = np.random.random((count, TEST_DIMENSION)).astype(np.float32)
        faiss.normalize_L2(embeddings)
        return embeddings

    @pytest.fixture
    def sample_ids(self, sample_embeddings):
        return [1000 + i*10 for i in range(sample_embeddings.shape[0])]

    def test_initialize_index_creates_new(self, faiss_manager, index_path):
        """Test that _initialize_index creates a new file if none exists."""
        index_file = Path(index_path)
        if index_file.exists(): index_file.unlink()
        assert not index_file.exists()
        
        index = faiss_manager._initialize_index(index_path, TEST_DIMENSION)
        
        assert index is not None
        assert isinstance(index, faiss.IndexIDMap) 
        assert index.d == TEST_DIMENSION
        assert index.ntotal == 0
        assert index_file.exists(), "Index file was not created"

    def test_initialize_index_loads_existing(self, faiss_manager, index_path, sample_embeddings, sample_ids):
        """Test that _initialize_index loads an existing index file."""
        # 1. Create and save an index manually first (using TEST_DIMENSION)
        base_index = faiss.IndexFlatL2(TEST_DIMENSION)
        initial_index = faiss.IndexIDMap(base_index)
        initial_index.add_with_ids(sample_embeddings, np.array(sample_ids, dtype=np.int64))
        faiss.write_index(initial_index, index_path)
        assert Path(index_path).exists()
        
        # 2. Initialize using the manager - should load (pass correct dimension)
        loaded_index = faiss_manager._initialize_index(index_path, TEST_DIMENSION)
        
        assert loaded_index is not None
        assert isinstance(loaded_index, faiss.IndexIDMap)
        assert loaded_index.d == TEST_DIMENSION
        assert loaded_index.ntotal == len(sample_ids)

    def test_initialize_index_dimension_mismatch(self, faiss_manager, index_path):
        """Test error handling when loaded index dimension mismatches."""
        wrong_dim = TEST_DIMENSION + 1
        # 1. Create and save index with wrong dimension
        base_index_wrong = faiss.IndexFlatL2(wrong_dim)
        initial_index_wrong = faiss.IndexIDMap(base_index_wrong)
        dummy_embed = np.random.random((1, wrong_dim)).astype(np.float32)
        dummy_id = np.array([1], dtype=np.int64)
        initial_index_wrong.add_with_ids(dummy_embed, dummy_id)
        faiss.write_index(initial_index_wrong, index_path)

        # 2. Attempt to initialize with the *correct* dimension - should raise error
        with pytest.raises(ValueError, match=f"Dimensão do índice carregado \({wrong_dim}\) diferente da esperada \({TEST_DIMENSION}\)"):
            faiss_manager._initialize_index(index_path, TEST_DIMENSION)

    def test_add_embeddings(self, faiss_manager, index_path, sample_embeddings, sample_ids):
        """Test adding embeddings with IDs and verify by searching."""
        index_file = Path(index_path)
        if index_file.exists(): index_file.unlink()
        assert not index_file.exists()

        result = faiss_manager.add_embeddings(sample_embeddings, sample_ids, index_path, TEST_DIMENSION)
        
        assert result is None 
        assert index_file.exists()

        query_vector = sample_embeddings[1] 
        expected_id = sample_ids[1]

        distances, ids_result = faiss_manager.search_faiss_index(query_vector, index_path, TEST_DIMENSION, k=1)

        assert ids_result.shape == (1, 1)
        assert distances.shape == (1, 1)
        assert ids_result[0, 0] == expected_id
        assert distances[0, 0] < 1e-6 
        
        reloaded_index = faiss_manager._initialize_index(index_path, TEST_DIMENSION)
        assert reloaded_index.ntotal == len(sample_ids)
        assert reloaded_index.d == TEST_DIMENSION

    def test_add_embeddings_invalid_ids_type(self, faiss_manager, index_path, sample_embeddings):
         """Test adding embeddings with invalid ID types."""
         ids_float = [float(i) for i in range(len(sample_embeddings))]
         with pytest.raises(TypeError, match="Todos os IDs na lista devem ser inteiros"):
             faiss_manager.add_embeddings(sample_embeddings, ids_float, index_path, TEST_DIMENSION)

    def test_add_embeddings_invalid_ids_length(self, faiss_manager, index_path, sample_embeddings, sample_ids):
        """Test adding embeddings with mismatching ID list length."""
        wrong_ids = sample_ids[:-1] 
        with pytest.raises(ValueError, match="Número de IDs .* não corresponde"):
             faiss_manager.add_embeddings(sample_embeddings, wrong_ids, index_path, TEST_DIMENSION)
             
    def test_add_embeddings_invalid_embedding_dim(self, faiss_manager, index_path, sample_ids):
         """Test adding embeddings with wrong dimension."""
         wrong_embeddings = np.random.random((len(sample_ids), TEST_DIMENSION + 1)).astype(np.float32)
         with pytest.raises(ValueError, match="Embeddings inválidos"):
             faiss_manager.add_embeddings(wrong_embeddings, sample_ids, index_path, TEST_DIMENSION)

    def test_search_faiss_index(self, faiss_manager, index_path, sample_embeddings, sample_ids):
        """Test searching the index and getting correct IDs."""
        faiss_manager.add_embeddings(sample_embeddings, sample_ids, index_path, TEST_DIMENSION)
        
        query_vector = sample_embeddings[2].reshape(1, -1)
        k = 3
        distances, ids_result = faiss_manager.search_faiss_index(query_vector, index_path, TEST_DIMENSION, k=k)
        
        assert distances.shape == (1, k)
        assert ids_result.shape == (1, k)
        assert ids_result.dtype == np.int64
        assert ids_result[0, 0] == sample_ids[2] 
        assert distances[0, 0] < 1e-6 
        assert all(found_id in sample_ids for found_id in ids_result[0])

    def test_search_empty_index(self, faiss_manager, index_path):
        """Test searching an empty index."""
        index = faiss_manager._initialize_index(index_path, TEST_DIMENSION)
        assert index.ntotal == 0
        
        query_vector = np.random.random((1, TEST_DIMENSION)).astype(np.float32)
        distances, ids_result = faiss_manager.search_faiss_index(query_vector, index_path, TEST_DIMENSION, k=5)
        
        assert distances.shape == (1, 0) 
        assert ids_result.shape == (1, 0)

    def test_search_k_greater_than_ntotal(self, faiss_manager, index_path, sample_embeddings, sample_ids):
        """Test searching with k larger than the number of items in the index."""
        num_to_add = 3
        assert num_to_add < faiss_manager.config.query.retrieval_k
        
        faiss_manager.add_embeddings(sample_embeddings[:num_to_add], sample_ids[:num_to_add], index_path, TEST_DIMENSION)
        
        query_vector = sample_embeddings[0].reshape(1, -1)
        distances, ids_result = faiss_manager.search_faiss_index(query_vector, index_path, TEST_DIMENSION)
        
        assert distances.shape == (1, num_to_add)
        assert ids_result.shape == (1, num_to_add)
        assert ids_result[0, 0] == sample_ids[0] 
        assert all(found_id in sample_ids[:num_to_add] for found_id in ids_result[0])

    def test_persistence_with_idmap(self, faiss_manager, index_path, sample_embeddings, sample_ids):
         """Test saving and loading an IndexIDMap correctly."""
         # 1. Add data - pass dimension
         faiss_manager.add_embeddings(sample_embeddings, sample_ids, index_path, TEST_DIMENSION)
         
         # 2. Create new manager (no system_config needed)
         manager2 = FaissManager(
             config=faiss_manager.config,
             log_domain="test_faiss_load"
         )
         
         # 3. Search - pass dimension
         query_vector = sample_embeddings[1].reshape(1, -1)
         distances, ids_result = manager2.search_faiss_index(query_vector, index_path, TEST_DIMENSION, k=1)
         
         assert ids_result.shape == (1, 1)
         assert ids_result[0, 0] == sample_ids[1] 
         assert distances[0, 0] < 1e-6

    # --- update_config Tests ---

    def test_update_config_no_change(self, faiss_manager, app_config):
        """Test update_config when the new config object is identical."""
        initial_config_ref = faiss_manager.config
        new_config = app_config.model_copy() # Create identical copy

        faiss_manager.update_config(new_config)

        # Config reference should be updated to the new object
        assert faiss_manager.config is new_config
        # Values should still be equal
        assert faiss_manager.config == initial_config_ref 

    def test_update_config_retrieval_k_change(self, faiss_manager, app_config, index_path, sample_embeddings, sample_ids):
        """Test that updating config (specifically retrieval_k) affects search."""
        # Add data first
        faiss_manager.add_embeddings(sample_embeddings, sample_ids, index_path, TEST_DIMENSION)
        initial_k = faiss_manager.config.query.retrieval_k
        assert initial_k == 5 # Based on fixture

        # Create new config with different k
        new_config = app_config.model_copy(deep=True)
        new_k = 3
        new_config.query.retrieval_k = new_k
        assert new_k != initial_k

        # Update the manager's config
        faiss_manager.update_config(new_config)

        # Verify config object was updated
        assert faiss_manager.config is new_config
        assert faiss_manager.config.query.retrieval_k == new_k

        # Perform a search and verify the *new* k is used (implicitly by search method)
        query_vector = sample_embeddings[0].reshape(1, -1)
        # Mock _initialize_index to avoid file interaction and focus on search logic
        with patch.object(faiss_manager, '_initialize_index', return_value=faiss.read_index(index_path)):
             distances, ids_result = faiss_manager.search_faiss_index(
                 query_embedding=query_vector,
                 index_path=index_path, 
                 dimension=TEST_DIMENSION
                 # IMPORTANT: Do NOT pass k here, let the method use its internal config
             )
        
        # Check if the search returned the NEW k results
        assert ids_result.shape[1] == new_k 

    def test_update_config_vector_store_change(self, faiss_manager, app_config):
        """Test updating vector_store config (currently just updates reference)."""
        # This test mainly verifies the config object updates.
        # The current update_config doesn't have specific logic for vector_store changes.
        initial_config_ref = faiss_manager.config

        new_config = app_config.model_copy(deep=True)
        new_config.vector_store.index_type = "IndexIVFFlat" # Hypothetical change
        new_config.vector_store.index_params = {"nlist": 100} # Hypothetical change
        
        faiss_manager.update_config(new_config)

        assert faiss_manager.config is new_config
        assert faiss_manager.config != initial_config_ref
        assert faiss_manager.config.vector_store.index_type == "IndexIVFFlat"
        assert faiss_manager.config.vector_store.index_params == {"nlist": 100}
        # NOTE: No assertion here that the *behavior* of index creation changes,
        # because update_config itself doesn't trigger re-initialization based on this.