from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from dotenv import load_dotenv
import os
from src.utils.logger import get_logger
from src.config.models import LLMConfig

load_dotenv()

class HuggingFaceManager:
    def __init__(self, config: LLMConfig, log_domain: str = "Processamento de queries"):
        """Inicializa o gerenciador de interação com a API Hugging Face.

        Args:
            config (LLMConfig): Objeto de configuração contendo os parâmetros do LLM, incluindo retry.
            log_domain (str): Domínio para o logger.
        """
        self.logger = get_logger(__name__, log_domain=log_domain)
        self.config = config.model_copy(deep=True)
        self.logger.info(f"Inicializando o HuggingFaceManager com configuração: {config}")
        self.max_retries = self.config.max_retries
        self.retry_delay = self.config.retry_delay_seconds
        self.client = self._initialize_client()

    def update_config(self, new_config: LLMConfig) -> None:
        """
        Atualiza a configuração do HuggingFaceManager com base na configuração fornecida.

        Args:
            config (LLMConfig): A nova configuração a ser aplicada.
        """ 
        if new_config == self.config:
            self.logger.info("Nenhuma alteracao na configuracao detectada")
            return
        
        self.max_retries = new_config.max_retries
        self.retry_delay = new_config.retry_delay_seconds

        if new_config.model_repo_id != self.config.model_repo_id:
            self.config = new_config.model_copy(deep=True)
            self.client = self._initialize_client()

        else:
            self.config = new_config.model_copy(deep=True)
        
        self.logger.info("Configuracoes do HuggingFaceManager atualizadas com sucesso")

    def _initialize_client(self) -> InferenceClient:
        """Inicializa o cliente Hugging Face Inference."""

        self.logger.debug("Inicializando o cliente Hugging Face")
        
        token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not token:
            self.logger.warning("HUGGINGFACE_API_TOKEN não encontrado no ambiente. Algumas operações podem falhar.")
        
        self.logger.debug(f"Inicializado InferenceClient do Hugging Face para o modelo: {self.config.model_repo_id}")
        return InferenceClient(
            token=token,
            #model=self.config.model_repo_id,
        )



    def generate_answer(self, question: str, context_prompt: str) -> str: 
        """
        Gera uma resposta de texto usando o modelo configurado da Hugging Face.

        Args:
            question (str): A pergunta para a geração de resposta.
            context_prompt (str): O prompt completo para a geração de texto.

        Returns:
            str: A resposta gerada pelo modelo ou uma mensagem de erro.
        
        Raises:
            ValueError: Se o prompt for vazio ou inválido.
            HfHubHTTPError: Se ocorrer um erro na comunicação com a API Hugging Face.
        """
        self.logger.info("Gerando resposta via API Hugging Face", question=question)
        if not context_prompt:
            self.logger.error("Erro ao gerar a resposta: Prompt vazio ou invalido")
            raise ValueError("Prompt vazio ou inválido")

        try:
            response = self.client.text_generation(
                prompt=context_prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                details=False,
                return_full_text=False
                )
            self.logger.debug("Resposta gerada com sucesso")
            return response
            
        except HfHubHTTPError as e:
            self.logger.error(f"Erro HTTP ao gerar resposta da API Hugging Face: {str(e)}", exc_info=True)
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                if status_code == 429:
                    self.logger.error(f"{status_code} - Erro de Rate Limit: {str(e)}")
                elif status_code in (502, 503, 504):
                    self.logger.error(f"{status_code} - Erro de servico Hugging Face indisponível: {str(e)}")
                else:
                    self.logger.error(f"{status_code} - Erro na API Hugging Face: {str(e)}")
            else:
                 self.logger.error(f"Erro HTTP indeterminado ou sem resposta detalhada na API Hugging Face: {str(e)}")
            raise e
        except Exception as e:
            self.logger.error(f"Erro inesperado ao gerar resposta: {e}", exc_info=True)
            raise e
                    
