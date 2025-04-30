import pytest
import os
from unittest.mock import patch, MagicMock
from src.query_processing.hugging_face_manager import HuggingFaceManager
from src.config.models import LLMConfig
from huggingface_hub.errors import HfHubHTTPError

# Create a default config for testing purposes
@pytest.fixture(scope="class")
def test_llm_config():
    return LLMConfig(
        model_repo_id="mock-model/test-model-v1",
        max_new_tokens=150, 
        temperature=0.6,
        top_p=0.8,
        top_k=40,
        repetition_penalty=1.1,
        max_retries=2,
        retry_delay_seconds=1
    )

class TestHuggingFaceManager:
    """Suite de testes para a classe HuggingFaceManager."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self, test_llm_config):
        """Inject the test_llm_config into the test class instance."""
        self.config = test_llm_config

    def test_initialization(self):
        """Testa a inicialização do HuggingFaceManager."""
        manager = HuggingFaceManager(config=self.config, log_domain="test_domain")
        assert manager.client is not None
        assert manager.config == self.config
        with patch('src.query_processing.hugging_face_manager.InferenceClient') as MockInferenceClient:
            with patch('os.getenv', return_value='test_token'):
                manager = HuggingFaceManager(config=self.config, log_domain="test_domain")
                MockInferenceClient.assert_called_once_with(
                    token='test_token', 
                    model=self.config.model_repo_id
                )

    def test_empty_prompt(self):
        """Testa o comportamento do HuggingFaceManager com um prompt vazio."""
        manager = HuggingFaceManager(config=self.config, log_domain="test_domain")
        dummy_question = "Qualquer pergunta?"
        
        # Test with empty context prompt - renamed back to generate_answer, added question
        with pytest.raises(ValueError, match="Prompt vazio ou inválido"):
            manager.generate_answer(dummy_question, "") 
        
        # Test with None context prompt - renamed back to generate_answer, added question
        with pytest.raises(ValueError, match="Prompt vazio ou inválido"):
            manager.generate_answer(dummy_question, None) # type: ignore
    
    def test_generate_answer_success(self):
        """Testa a geração de resposta com sucesso."""
        manager = HuggingFaceManager(config=self.config, log_domain="test_domain")
        
        mock_client = MagicMock()
        mock_response_text = "Esta é uma resposta de teste do modelo."
        mock_client.text_generation.return_value = mock_response_text
        manager.client = mock_client
        
        # Test - renamed back, added question
        question = "Qual a capital?" # Specific question for context
        context_prompt = "Contexto: Brasil é um país na América do Sul. Pergunta: Qual a capital?"
        result = manager.generate_answer(question, context_prompt)
        
        assert result == mock_response_text
        # Mock assertion remains the same, checking parameters passed to text_generation
        mock_client.text_generation.assert_called_once()
        args, kwargs = mock_client.text_generation.call_args
        assert kwargs.get('prompt') == context_prompt
        assert kwargs.get('details') is False
        assert kwargs.get('return_full_text') is False
        assert kwargs.get('max_new_tokens') == self.config.max_new_tokens
        assert kwargs.get('temperature') == self.config.temperature
        assert kwargs.get('top_p') == self.config.top_p
        assert kwargs.get('top_k') == self.config.top_k
        assert kwargs.get('repetition_penalty') == self.config.repetition_penalty
    
    def test_generate_answer_http_error(self):
        """Testa o comportamento quando ocorre um HfHubHTTPError."""
        manager = HuggingFaceManager(config=self.config, log_domain="test_domain")
        
        mock_client = MagicMock()
        error_message = "API error: rate limit exceeded (429)"
        mock_response = MagicMock()
        mock_response.status_code = 429
        http_error = HfHubHTTPError(error_message, response=mock_response)
        mock_client.text_generation.side_effect = http_error
        manager.client = mock_client
        
        # Test exception handling - renamed back, added question
        question = "Qual a capital?"
        context_prompt = "Contexto: Brasil é um país na América do Sul. Pergunta: Qual a capital?"
        with pytest.raises(HfHubHTTPError) as exc_info:
            manager.generate_answer(question, context_prompt)
        
        assert error_message in str(exc_info.value)
        assert exc_info.value.response.status_code == 429
        # Mock assertion remains the same
        mock_client.text_generation.assert_called_once_with(
            prompt=context_prompt,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            details=False,
            return_full_text=False
        )
        
    def test_generate_answer_generic_error(self):
        """Testa o comportamento quando ocorre um erro genérico na geração da resposta."""
        manager = HuggingFaceManager(config=self.config, log_domain="test_domain")
        
        mock_client = MagicMock()
        error_message = "Some unexpected internal error"
        generic_exception = Exception(error_message)
        mock_client.text_generation.side_effect = generic_exception
        manager.client = mock_client
        
        # Test exception handling - renamed back, added question
        question = "Qual a capital?"
        context_prompt = "Contexto: Brasil é um país na América do Sul. Pergunta: Qual a capital?"
        with pytest.raises(Exception, match=error_message) as exc_info:
             manager.generate_answer(question, context_prompt)

        assert not isinstance(exc_info.value, HfHubHTTPError)
        # Mock assertion remains the same
        mock_client.text_generation.assert_called_once() 