import os
import shutil
import pytest
import tempfile
import numpy as np
import faiss
from pathlib import Path

from components.shared import FaissManager
from components.models import Embedding

class TestFaissManager:
    """Test suite for FaissManager class."""
    
    @pytest.fixture
    def test_indices_dir(self):
        """Create a temporary directory for test indices."""
        # Create a subdirectory within indices/ called 'test'
        test_dir = os.path.join("indices", "test")
        os.makedirs(test_dir, exist_ok=True)
        
        yield test_dir
        
        # Cleanup: remove all files in the test directory
        for file in os.listdir(test_dir):
            file_path = os.path.join(test_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    
    @pytest.fixture
    def faiss_manager(self, test_indices_dir):
        """Create a FaissManager instance configured to use the test directory."""
        # Initialize FaissManager with the test directory
        # Use a different name to avoid conflicts with other tests
        test_index_path = os.path.join(test_indices_dir, f"test_index_{os.getpid()}.faiss")
        manager = FaissManager(
            index_path=test_index_path,
            dimension=384  # Standard dimension for test
        )
        
        return manager
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        embeddings = []
        
        # Create 5 test embeddings
        for i in range(5):
            # Create a normalized random vector
            vec = np.random.random(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            
            embedding = Embedding(
                id=None,
                chunk_id=i + 1,
                faiss_index_path=None,  # Will be set by FaissManager
                chunk_faiss_index=None,  # Will be set by FaissManager
                dimension=384,
                embedding=vec
            )
            
            embeddings.append(embedding)
        
        return embeddings
    
    def test_initialization(self, test_indices_dir):
        """Test FaissManager initialization."""
        # Create a new FaissManager instance
        test_index_path = os.path.join(test_indices_dir, "init_test.faiss")
        
        # Make sure there's no index file before the test
        if os.path.exists(test_index_path):
            os.unlink(test_index_path)
        
        manager = FaissManager(
            index_path=test_index_path,
            dimension=512  # Different dimension for this test
        )
        
        # Verify the manager was initialized correctly
        assert manager.index_path == test_index_path
        assert manager.dimension == 512
        assert manager.index is not None
        assert isinstance(manager.index, faiss.Index)
        assert manager.index.d == 512  # Dimension matches
        
        # Verify that an index file was created
        assert os.path.exists(test_index_path)
    
    def test_add_embeddings(self, faiss_manager, sample_embeddings):
        """Test adding embeddings to the index."""
        # Get the initial count of vectors in the index
        initial_count = faiss_manager.index.ntotal
        original_count = len(sample_embeddings)
        
        # Add the sample embeddings to the index
        faiss_manager.add_embeddings(sample_embeddings)
        
        # ONLY verify that the vectors were added to the index
        assert faiss_manager.index.ntotal == initial_count + original_count
        
        # We no longer expect the embedding objects to be modified
        # No assertions about faiss_index_path or chunk_faiss_index
    
    def test_index_persistence(self, test_indices_dir):
        """Test that the index is persisted to disk."""
        # Create a unique index path
        test_index_path = os.path.join(test_indices_dir, f"persistence_test_{os.getpid()}.faiss")
        
        # Create a manager and add some vectors
        manager1 = FaissManager(
            index_path=test_index_path,
            dimension=384
        )
        
        # Create and add a test embedding
        vec = np.random.random(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        embedding = Embedding(
            id=None,
            chunk_id=1,
            faiss_index_path=None,
            chunk_faiss_index=None,
            dimension=384,
            embedding=vec
        )
        
        manager1.add_embeddings([embedding])
        
        # Create a new manager that should load the existing index
        manager2 = FaissManager(
            index_path=test_index_path,
            dimension=384
        )
        
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
        # Split the sample embeddings into two batches
        batch1 = sample_embeddings[:2]
        batch2 = sample_embeddings[2:]
        
        # Add the first batch
        faiss_manager.add_embeddings(batch1)
        
        # Verify only that vectors were added to the index
        assert faiss_manager.index.ntotal == len(batch1)
        
        # Add the second batch
        faiss_manager.add_embeddings(batch2)
        
        # Verify that all vectors were added
        assert faiss_manager.index.ntotal == len(batch1) + len(batch2)
    
    def test_index_reuse(self, test_indices_dir):
        """Test creating a new manager with the same index file."""
        # Create a unique index name
        test_index_path = os.path.join(test_indices_dir, "reuse_test.faiss")
        
        # Make sure the file doesn't exist initially
        if os.path.exists(test_index_path):
            os.unlink(test_index_path)
        
        # Create the first manager and add vectors
        manager1 = FaissManager(
            index_path=test_index_path,
            dimension=384
        )
        
        # Create and add multiple embeddings
        embedding_count = 5
        embeddings = []
        
        for i in range(embedding_count):
            vec = np.random.random(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            
            embedding = Embedding(
                id=None,
                chunk_id=i + 1,
                faiss_index_path=None,
                chunk_faiss_index=None,
                dimension=384,
                embedding=vec
            )
            
            embeddings.append(embedding)
        
        # Add embeddings to the first manager
        manager1.add_embeddings(embeddings)
        
        # Create a second manager that should load the index
        manager2 = FaissManager(
            index_path=test_index_path,
            dimension=384
        )
        
        # Verify the second manager loaded all vectors
        assert manager2.index.ntotal == embedding_count