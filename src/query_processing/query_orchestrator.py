import numpy as np
from typing import List

from src.utils import TextNormalizer, EmbeddingGenerator, FaissManager, SQLiteManager
from src.utils.logger import get_logger
from .hugging_face_manager import HuggingFaceManager

class QueryOrchestrator:
    DEFAULT_LOG_DOMAIN = "Processamento de queries"
    def __init__(self):
        self.logger = get_logger(__name__, log_domain=self.DEFAULT_LOG_DOMAIN)
        self.logger.info("Inicializando o QueryOrchestrator")

        self.text_normalizer = TextNormalizer(log_domain=self.DEFAULT_LOG_DOMAIN)
        self.embedding_generator = EmbeddingGenerator(log_domain=self.DEFAULT_LOG_DOMAIN)
        self.faiss_manager = FaissManager(log_domain=self.DEFAULT_LOG_DOMAIN)
        self.sqlite_manager = SQLiteManager(log_domain=self.DEFAULT_LOG_DOMAIN)
        self.hugging_face_manager = HuggingFaceManager(log_domain=self.DEFAULT_LOG_DOMAIN)

    def _process_query(self, query: str) -> np.ndarray:
        """
        Processa a query e retorna o embedding gerado.

        Args:
            query (str): A query original.

        Returns:
            np.ndarray: O embedding gerado.
        """
        self.logger.info("Iniciando o tratamento da query")
        if not query:
            self.logger.error("Query vazia ou inválida")
            raise ValueError("Query vazia ou inválida")

        try:
            self.logger.info("Normalizando a query")
            normalized_query = self.text_normalizer.normalize(query)
            self.logger.info("Gerando o embedding da query")
            query_embedding = self.embedding_generator.generate_embeddings(normalized_query)
            
            if query_embedding.size == 0:
                self.logger.error("Não foi possível gerar um embedding para a query")
                raise ValueError("Não foi possível gerar um embedding para a query")
            
            return query_embedding
        
        except Exception as e:
            self.logger.error(f"Erro ao processar a query: {str(e)}")
            raise e
    
    def _retrieve_documents(self, query_embedding: np.ndarray) -> List[str]:
        """
        Recupera os chunks de conteúdo relevantes para a query usando o FaissManager.

        Args:
            query_embedding (np.ndarray): O embedding da query.

        Returns:
            List[str]: Uma lista de chunks de conteúdo relevantes.
        """
        self.logger.info("Recuperando os chunks de conteúdo relevantes")
        if query_embedding is None or query_embedding.size == 0:
            self.logger.error("Embedding vazio ou inválido")
            raise ValueError("Embedding vazio ou inválido")
        
        try:
            _, indices = self.faiss_manager.search_faiss_index(query_embedding)
            flat_indices = indices.flatten().tolist()

            with self.sqlite_manager.get_connection() as conn:
                chunks_content = self.sqlite_manager.get_embeddings_chunks(conn, flat_indices)
            self.logger.info("Chunks de conteúdo recuperados com sucesso")
            return chunks_content
        
        except Exception as e:
            self.logger.error(f"Erro ao recuperar chunks de conteúdo: {str(e)}")
            raise e
    
    def _prepare_context_prompt(self, query: str, chunks_content: List[str]) -> str:
        """
        Prepara o prompt para ser enviado ao modelo LLM.

        Args:
            query (str): A query original.
            chunks_content (List[str]): Uma lista de strings contendo os chunks recuperados.

        Returns:
            str: O prompt preparado para a geração de resposta.
        """
        self.logger.info("Preparando o prompt para a geração de resposta")
        if not query:
            self.logger.error("Query vazia ou inválida")
            raise ValueError("Query vazia ou inválida")
        
        if not chunks_content:
            self.logger.error("Lista de chunks vazia ou inválida")
            raise ValueError("Lista de chunks vazia ou inválida")
        
        context_prompt = f"""
        Context:
        {chunks_content}

        Question:
        {query}
        """
        self.logger.info("Prompt preparado com sucesso")
        return context_prompt

    def query_llm(self, query: str) -> str:
        """
        Processa a query e retorna a resposta gerada pelo modelo LLM.

        Args:
            query (str): A query original.

        Returns:
            str: A resposta gerada pelo modelo LLM.
        """
        self.logger.info("Iniciando o pipeline de processamento de queries")
        if not query:
            self.logger.error("Query vazia ou inválida")
            raise ValueError("Query vazia ou inválida")

        try:
            query_embedding = self._process_query(query)
            chunks_content = self._retrieve_documents(query_embedding)
            context_prompt = self._prepare_context_prompt(query, chunks_content)
            answer = self.hugging_face_manager.generate_answer(query, context_prompt)
            return answer
        
        except Exception as e:
            self.logger.error(f"Erro ao processar a query: {str(e)}")
            raise e
        
        