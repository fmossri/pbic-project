import numpy as np
from typing import List, Dict, Any
from datetime import datetime

from src.utils import TextNormalizer, EmbeddingGenerator, FaissManager, SQLiteManager
from src.utils.logger import get_logger
from .hugging_face_manager import HuggingFaceManager

class QueryOrchestrator:
    DEFAULT_LOG_DOMAIN = "Processamento de queries"
    def __init__(self):
        self.logger = get_logger(__name__, log_domain=self.DEFAULT_LOG_DOMAIN)
        self.logger.info("Inicializando o QueryOrchestrator")

        self.metrics_data = {}
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
            self.logger.error("Query vazia ou invalida")
            raise ValueError("Query vazia ou inválida")

        try:
            self.logger.info("Normalizando a query")
            normalized_query = self.text_normalizer.normalize(query)
            self.logger.info("Gerando o embedding da query")
            query_embedding = self.embedding_generator.generate_embeddings(normalized_query)
            self.metrics_data["query_embedding"] = query_embedding
            self.metrics_data["query_embedding_size"] = query_embedding.size
            if query_embedding.size == 0:
                self.logger.error("Erro ao gerar o embedding da query")
                raise ValueError("Erro ao gerar o embedding da query")
            
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
        self.logger.info("Recuperando os chunks de conteudo relevantes")
        if query_embedding is None or query_embedding.size == 0:
            self.logger.error("Vetor de embedding vazio ou invalido")
            raise ValueError("Vetor de embedding vazio ou inválido")
        
        try:
            _, indices = self.faiss_manager.search_faiss_index(query_embedding)
            flat_indices = indices.flatten().tolist()
            self.metrics_data["knn_faiss_indices"] = len(flat_indices)

            with self.sqlite_manager.get_connection() as conn:
                chunks_content = self.sqlite_manager.get_chunks_content(conn, flat_indices)
                self.metrics_data["chunks_content"] = len(chunks_content)
            self.logger.info("Chunks de conteudo recuperados com sucesso")
            return chunks_content
        
        except Exception as e:
            self.logger.error(f"Erro ao recuperar chunks de conteudo: {str(e)}")
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
        self.logger.info("Preparando o prompt de contexto")
        if not query:
            self.logger.error("Erro ao preparar o prompt de contexto: Query vazia ou invalida")
            raise ValueError("Query vazia ou inválida")
        
        if not chunks_content:
            self.logger.error("Erro ao preparar o prompt de contexto: Lista de chunks vazia ou invalida")
            raise ValueError("Lista de chunks vazia ou inválida")
        
        context_prompt = f"""
        Context:
        {chunks_content}

        Question:
        {query}
        """
        self.logger.debug("Prompt de contexto preparado com sucesso")
        return context_prompt
    
    def _setup_metrics_data(self) -> None:
        """
        Obtém os dados de métricas para o processo de processamento de queries.
        
        Returns:
            Dict[str, Any]: Dicionário contendo os dados de métricas
        """
        start_time = datetime.now()
        self.metrics_data["process"] = "Processamento de queries"
        self.metrics_data["start_time"] = start_time
        self.metrics_data["embedding_model"] = self.embedding_generator.model_name
        self.metrics_data["embedding_dimension"] = self.embedding_generator.embedding_dimension
        self.metrics_data["faiss_index_path"] = self.faiss_manager.index_path
        self.metrics_data["faiss_index_type"] = type(self.faiss_manager.index).__name__
        self.metrics_data["faiss_dimension"] = self.faiss_manager.dimension
        self.metrics_data["database_path"] = self.sqlite_manager.db_path

    def query_llm(self, query: str) -> Dict[str, Any]:
        """
        Processa a query e retorna a resposta gerada pelo modelo LLM.

        Args:
            query (str): A query original.

        Returns:
            str: A resposta gerada pelo modelo LLM.
        """
        self._setup_metrics_data()

        self.logger.info("Iniciando o processamento da pergunta")
        if not query:
            self.metrics_data["processing_duration"] = str(datetime.now() - self.metrics_data["start_time"])
            self.metrics_data["success"] = False
            self.logger.error("Erro ao processar a query: Query vazia ou invalida")
            raise ValueError("Query vazia ou inválida")

        try:
            self.metrics_data["question"] = query
            query_embedding = self._process_query(query)
            chunks_content = self._retrieve_documents(query_embedding)
            context_prompt = self._prepare_context_prompt(query, chunks_content)
            answer = self.hugging_face_manager.generate_answer(query, context_prompt)
            self.metrics_data["answer"] = answer
            self.metrics_data["processing_duration"] = str(datetime.now() - self.metrics_data["start_time"])
            self.metrics_data["success"] = True
            return self.metrics_data
        
        except Exception as e:
            self.metrics_data["processing_duration"] = str(datetime.now() - self.metrics_data["start_time"])
            self.metrics_data["success"] = False
            self.logger.error(f"Erro ao processar a query: {str(e)}")
            raise e
        
        