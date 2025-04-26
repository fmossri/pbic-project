import os
import shutil
import pytest
import tempfile
import numpy as np
import faiss
from pathlib import Path

from src.utils import FaissManager

class TestFaissManager:
    """Test suite for FaissManager class."""
    
    @pytest.fixture
    def test_indices_dir(self):
        """Create a temporary directory for test indices."""
        # Create a subdirectory within tests/storage/domains/test_domain/vector_store
        test_dir = os.path.join("tests", "storage", "domains", "test_domain", "vector_store")
        os.makedirs(test_dir, exist_ok=True)
        
        yield test_dir
        
        # Cleanup: remove all files in the test directory
        for file in os.listdir(test_dir):
            file_path = os.path.join(test_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    @pytest.fixture
    def index_path(self, test_indices_dir):
        test_index_path = os.path.join(test_indices_dir, f"test_index_{os.getpid()}.faiss")
        return test_index_path

    @pytest.fixture
    def faiss_manager(self, test_indices_dir):
        """Create a FaissManager instance configured to use the test directory."""
        # Initialize FaissManager with the test directory
        # Use a different name to avoid conflicts with other tests

        manager = FaissManager(
            log_domain="test_domain"
        )
        manager.index_path = os.path.join(test_indices_dir, f"test_index_{os.getpid()}.faiss")
        manager.dimension = 384
        
        return manager
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        embeddings = np.zeros((5, 384), dtype=np.float32)
        
        # Create 5 test embeddings
        for i in range(5):
            # Create a normalized random vector
            vec = np.random.random(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            embeddings[i] = vec
        
        return embeddings
    
    def test_initialization(self, index_path):
        """Test FaissManager initialization."""
        
        # Make sure there's no index file before the test
        if os.path.exists(index_path):
            os.unlink(index_path)
        
        manager = FaissManager(
            log_domain="test_domain"
        )
        manager.dimension = 512
        manager.index_path = index_path
        manager._initialize_index()
        
        # Verify the manager was initialized correctly
        assert manager.index_path == index_path
        assert manager.dimension == 512
        assert manager.index is not None
        assert isinstance(manager.index, faiss.Index)
        assert manager.index.d == 512  # Dimension matches
        
        # Verify that an index file was created
        assert os.path.exists(index_path)
    
    def test_add_embeddings(self, faiss_manager, sample_embeddings):
        """Test adding embeddings to the index."""
        
        # Get the initial count of vectors in the index
        faiss_manager._initialize_index()
        initial_count = faiss_manager.index.ntotal
        original_count = sample_embeddings.shape[0]
        
        # Add the sample embeddings to the index
        vector_store_path = faiss_manager.index_path
        embedding_dimension = faiss_manager.dimension
        faiss_indices = faiss_manager.add_embeddings(sample_embeddings, vector_store_path, embedding_dimension)
        
        # Verify that the vectors were added to the index
        assert faiss_manager.index.ntotal == initial_count + original_count
        
        # Verify that the returned faiss_indices has the correct length
        assert len(faiss_indices) == original_count
        
        # Verify indices are sequential starting from initial_count
        for i, index in enumerate(faiss_indices):
            assert index == initial_count + i
    
    def test_index_persistence(self, test_indices_dir):
        """Test that the index is persisted to disk."""
        # Create a unique index path
        test_index_path = os.path.join(test_indices_dir, f"persistence_test_{os.getpid()}.faiss")
        
        # Create a manager and add some vectors
        manager1 = FaissManager(
            log_domain="test_domain"
        )
        manager1.index_path = test_index_path
        manager1.dimension = 384
        # Create and add a test embedding
        vec = np.random.random(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        vec_array = np.reshape(vec, (1, 384))
        
        manager1.add_embeddings(vec_array, test_index_path, 384)
        
        # Create a new manager that should load the existing index
        manager2 = FaissManager(
            log_domain="test_domain"
        )
        manager2.index_path = test_index_path
        manager2.dimension = 384
        manager2._initialize_index()
        # Verify that the second manager has the same vector count
        assert manager2.index.ntotal == 1
        
        # Verify the loaded index contains the vector
        # Extract the vector for similarity check
        encoded_vec = vec.reshape(1, -1)
        distances, indices = manager2.index.search(encoded_vec, 1)
        
        # The most similar vector should be itself
        assert indices[0][0] == 0
        assert distances[0][0] < 1e-5  # Should be very close to 0
    
    def test_multiple_adds(self, faiss_manager, sample_embeddings):
        """Test adding embeddings in multiple batches."""
        # Initialize index explicitly
        faiss_manager._initialize_index()
        
        # Split the sample embeddings into two batches
        batch1 = sample_embeddings[:2]
        batch2 = sample_embeddings[2:]
        
        # Add the first batch
        vector_store_path = faiss_manager.index_path
        embedding_dimension = faiss_manager.dimension
        faiss_indices1 = faiss_manager.add_embeddings(batch1, vector_store_path, embedding_dimension)
        
        # Verify vectors were added to the index
        assert faiss_manager.index.ntotal == batch1.shape[0]
        
        # Add the second batch
        faiss_indices2 = faiss_manager.add_embeddings(batch2, vector_store_path, embedding_dimension)
        
        # Verify that all vectors were added
        assert faiss_manager.index.ntotal == batch1.shape[0] + batch2.shape[0]
        
        # Verify the returned indices are correct
        assert len(faiss_indices1) == batch1.shape[0]
        assert len(faiss_indices2) == batch2.shape[0]
        assert faiss_indices2[0] == batch1.shape[0]  # Second batch should start after first batch
    
    def test_search_faiss_index(self, faiss_manager, sample_embeddings):
        """Test searching for similar vectors."""
        # Initialize index explicitly
        faiss_manager._initialize_index()
        
        # Add sample embeddings to the index
        vector_store_path = faiss_manager.index_path
        embedding_dimension = faiss_manager.dimension
        faiss_manager.add_embeddings(sample_embeddings, vector_store_path, embedding_dimension)
        
        # Search for a vector (use the first one from the sample set)
        query_vector = sample_embeddings[0].reshape(1, -1)
        distances, indices = faiss_manager.search_faiss_index(query_vector, k=1)
        
        # The most similar vector should be itself
        assert indices[0][0] == 0
        assert distances[0][0] < 1e-5  # Should be very close to 0