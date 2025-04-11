from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from dotenv import load_dotenv
import os
import time
from typing import Optional

load_dotenv()

class HuggingFaceManager:
    def __init__(self, model_name: str = "HuggingFaceH4/zephyr-7b-beta", max_retries: int = 3, retry_delay: int = 2):
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = self._initialize_client()

        print(f"Inicializando o cliente para o modelo: {self.model_name}")
        
    def _initialize_client(self) -> InferenceClient:
        """Initialize the Hugging Face Inference client."""
        token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not token:
            print("Aviso: HUGGINGFACE_API_TOKEN não encontrado no ambiente. Algumas operações podem falhar.")
        
        return InferenceClient(
            token=token,
            model=self.model_name,
        )

    def generate_answer(self, question: str, context_prompt: str) -> str:
        """
        Gera uma resposta para uma pergunta usando o modelo da Hugging Face.

        Args:
            context_prompt (str): O prompt para a geração de resposta.

        Returns:
            str: A resposta gerada pelo modelo ou uma mensagem de erro.
        """
        print(f"\n\nPergunta: {question}")
        if not context_prompt:
            return "prompt vazio ou inválido"

        try:
            response = self.client.text_generation(
                details=False,
                return_full_text=False,
                prompt=context_prompt,
                max_new_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.0
                )
            
            return response
            
        except HfHubHTTPError as e:
                
            if hasattr(e, 'response') and e.response.status_code == 429:
                # Rate limit error
                print(f"Erro de limite de taxa.")
                raise e

            elif hasattr(e, 'response') and e.response.status_code in (503, 502, 504):
                # Service unavailability
                print(f"Erro de serviço Hugging Face ({e.response.status_code}).")
                raise e
            else:
                # Other HTTP errors
                print(f"Erro na API Hugging Face: {str(e)}")
                raise e
                    
