from components.shared import TextNormalizer
from components.shared import EmbeddingGenerator
from components.storage import FaissManager, SQLiteManager
from .hugging_face_manager import HuggingFaceManager
from typing import List
import numpy as np

class QueryOrchestrator:
    def __init__(self):
        print("Inicializando o QueryOrchestrator...")
        self.text_normalizer = TextNormalizer()
        self.embedding_generator = EmbeddingGenerator()
        self.faiss_manager = FaissManager()
        self.sqlite_manager = SQLiteManager()
        self.hugging_face_manager = HuggingFaceManager()

    def _process_query(self, query: str) -> np.ndarray:
        if not query:
            raise ValueError("Query vazia ou inválida")
        
        normalized_query = self.text_normalizer.normalize(query)
        query_embedding = self.embedding_generator.generate_embeddings(normalized_query)
        
        if query_embedding.size == 0:
            raise ValueError("Não foi possível gerar um embedding para a query")
        
        return query_embedding
    
    def _retrieve_documents(self, query_embedding: np.ndarray) -> List[str]:
        """
        Recupera os chunks de conteúdo relevantes para a query usando o FaissManager.

        Args:
            query_embedding (np.ndarray): O embedding da query.

        Returns:
            List[str]: Uma lista de chunks de conteúdo relevantes.
        """
        if query_embedding is None or query_embedding.size == 0:
            raise ValueError("Embedding vazio ou inválido")
        
        try:
            _, indices = self.faiss_manager.search_faiss_index(query_embedding)
            flat_indices = indices.flatten().tolist()

            with self.sqlite_manager.get_connection() as conn:
                chunks_content = self.sqlite_manager.get_embeddings_chunks(conn, flat_indices)
            return chunks_content
        
        except Exception as e:
            print(f"Erro ao recuperar chunks de conteúdo: {str(e)}")
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
        if not query:
            raise ValueError("Query vazia ou inválida")
        
        if not chunks_content:
            raise ValueError("Lista de chunks vazia ou inválida")
        
        context_prompt = f"""
        Context:
        {chunks_content}

        Question:
        {query}
        """
        return context_prompt

    def query_llm(self, query: str) -> str:
        """
        Processa a query e retorna a resposta gerada pelo modelo LLM.

        Args:
            query (str): A query original.

        Returns:
            str: A resposta gerada pelo modelo LLM.
        """

        if not query:
            raise ValueError("Query vazia ou inválida")

        try:
            query_embedding = self._process_query(query)
            chunks_content = self._retrieve_documents(query_embedding)
            context_prompt = self._prepare_context_prompt(query, chunks_content)
            answer = self.hugging_face_manager.generate_answer(query, context_prompt)

            return answer
        
        except Exception as e:
            print(f"Erro ao processar a query: {str(e)}")
            raise e
        
        