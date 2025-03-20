import os
import pytest
import shutil

@pytest.fixture(scope="session")
def test_files_dir(tmp_path_factory):
    """Create a temporary directory for test files."""
    test_dir = tmp_path_factory.mktemp("test_files")
    yield str(test_dir)
    # Cleanup after tests
    shutil.rmtree(str(test_dir), ignore_errors=True) 