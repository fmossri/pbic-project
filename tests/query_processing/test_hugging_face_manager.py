import pytest
import os
from unittest.mock import patch, MagicMock
from components.query_processing.hugging_face_manager import HuggingFaceManager

class TestHuggingFaceManager:
    """Suite de testes para a classe HuggingFaceManager."""
    
    def test_initialization(self):
        """Testa a inicialização do HuggingFaceManager."""
        # Simply verify that a manager can be created and has a client
        manager = HuggingFaceManager()
        assert manager.client is not None
        
        # Verify the client has the expected model
        assert hasattr(manager.client, 'model'), "Client should have a model attribute"
        
        # Check token is obtained from environment
        with patch('os.getenv', return_value='test_token') as mock_getenv:
            another_manager = HuggingFaceManager()
            mock_getenv.assert_called_with("HUGGINGFACE_API_TOKEN")
    
    def test_empty_prompt(self):
        """Testa o comportamento do HuggingFaceManager com um prompt vazio."""
        # This test doesn't need mocking as it doesn't make API calls for empty prompts
        manager = HuggingFaceManager()
        
        # Test with empty string
        result = manager.generate_answer("")
        assert result == "prompt vazio ou inválido"
        
        # Test with None
        result = manager.generate_answer(None)
        assert result == "prompt vazio ou inválido"
    
    def test_generate_answer_success(self):
        """Testa a geração de resposta com sucesso."""
        # Create manager first
        manager = HuggingFaceManager()
        
        # Replace the client with a mock
        mock_client = MagicMock()
        mock_response = "Esta é uma resposta de teste do modelo."
        mock_client.text_generation.return_value = mock_response
        manager.client = mock_client
        
        # Test
        prompt = "Qual é a capital do Brasil?"
        result = manager.generate_answer(prompt)
        
        # Verify results
        assert result == mock_response
        mock_client.text_generation.assert_called_once()
        
        # Verify parameters
        args, kwargs = mock_client.text_generation.call_args
        assert kwargs.get('prompt') == prompt
        assert kwargs.get('details') is True
        assert kwargs.get('max_new_tokens') == 1000
        assert kwargs.get('temperature') == 0.7
        assert kwargs.get('top_p') == 0.9
        assert kwargs.get('top_k') == 50
        assert kwargs.get('repetition_penalty') == 1.0
    
    def test_generate_answer_error(self):
        """Testa o comportamento quando ocorre um erro na geração da resposta."""
        # Create manager
        manager = HuggingFaceManager()
        
        # Replace client with mock that raises an exception
        mock_client = MagicMock()
        error_message = "API error: rate limit exceeded"
        mock_client.text_generation.side_effect = Exception(error_message)
        manager.client = mock_client
        
        # Test exception handling
        prompt = "Qual é a capital do Brasil?"
        with pytest.raises(Exception) as exc_info:
            manager.generate_answer(prompt)
        
        # Verify the right exception was raised
        assert error_message in str(exc_info.value)
        mock_client.text_generation.assert_called_once()
        
        # Verify parameters
        args, kwargs = mock_client.text_generation.call_args
        assert kwargs.get('prompt') == prompt 