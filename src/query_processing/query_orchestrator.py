import os
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.models import Domain
from src.utils import TextNormalizer, EmbeddingGenerator, FaissManager, SQLiteManager
from src.utils.logger import get_logger
from .hugging_face_manager import HuggingFaceManager

class QueryOrchestrator:
    """
    Orquestrador de queries para o sistema de busca.
    """
    DEFAULT_LOG_DOMAIN = "Processamento de queries"
    def __init__(self):
        self.logger = get_logger(__name__, log_domain=self.DEFAULT_LOG_DOMAIN)
        self.logger.info("Inicializando o QueryOrchestrator")

        self.metrics_data = {}
        self.text_normalizer = TextNormalizer(log_domain=self.DEFAULT_LOG_DOMAIN)
        self.embedding_generator = EmbeddingGenerator(log_domain=self.DEFAULT_LOG_DOMAIN)
        self.faiss_manager = FaissManager(log_domain=self.DEFAULT_LOG_DOMAIN)
        self.sqlite_manager = SQLiteManager(log_domain=self.DEFAULT_LOG_DOMAIN)
        self.hugging_face_manager = HuggingFaceManager(log_domain=self.DEFAULT_LOG_DOMAIN, model_name="HuggingFaceH4/zephyr-7b-beta")

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
    
    def _select_domains(self, query: str, selected_domains: Optional[List[str]] = None) -> List[Domain]:
        """
        Seleciona os domínios relevantes para a query.

        Args:
            query (str): A query original.
            :
        """
        self.logger.info("Selecionando os dominios relevantes para a query")
        if not query:
            self.logger.error("Erro ao selecionar os dominios relevantes: Query vazia ou inválida")
            raise ValueError("Query vazia ou inválida")
        
        prepared_prompt = f"""
        Você é um especialista em selecionar domínios relevantes para uma query.
        Sua tarefa é selecionar os domínios que são mais relevantes para a query.
        Sua resposta deve ser sempre um nome de domínio ou uma lista de nomes de domínios separados por pipes "|".
        Você deve selecionar apenas um domínio, a não ser que a query seja muito ampla e possa precisar ser respondida por mais de um domínio.
        Retorne apenas o nome do domínio selecionado. Se mais de um domínio for relevante, retorne uma lista com os nomes dos domínios selecionados separados por pipes "|".
        Você não deve adicionar mais nenhuma informação à sua resposta, não explique ou exponha seu raciocínio. Mantenha sua resposta no formato abaixo:

        ================================================
        Exemplo de resposta para um único domínio:
        "Domínio 1"
        Exemplo de resposta para mais de um domínio:
        "Domínio 1|Domínio 2|Domínio 3"

        ================================================
        Domínios disponíveis:
        """


        try:
            with self.sqlite_manager.get_connection(control=True) as conn:
                domains = self.sqlite_manager.get_domain(conn)
                fetched_domain_names = [d.name for d in domains] if domains else []
                self.logger.debug("Dominios recuperados do banco de controle", fetched_domain_count=len(fetched_domain_names), fetched_domain_names=fetched_domain_names)
                
                # Se não houver domínios selecionados pelo usuário, seleciona automaticamente entre todos domínios populados
                if not selected_domains:
                    if not domains:
                        self.logger.error("Banco de controle nao retornou nenhum dominio. Nao e possivel selecionar automaticamente.")
                        raise ValueError("Nenhum domínio encontrado no banco de controle.")

                    i = 0
                    valid_domains = []
                    for domain in domains:
                        # Adiciona todos os domínios populados ao prompt
                        if domain.db_path and os.path.exists(domain.db_path):
                            prepared_prompt += f"\n\nDomínio {i+1}:\nNome: {domain.name}\nDescrição: {domain.description}\nPalavras-chave: {domain.keywords}\nid: {domain.id}"
                            i += 1
                            valid_domains.append(domain)
                    
                    self.logger.info(f"Dominios disponiveis para selecao: {[domain.name for domain in valid_domains] if valid_domains else 'Nenhum'}")
                    prepared_prompt += f"\n\nPergunta: {query}"

                    self.logger.debug(f"Prompt preparado: {prepared_prompt}")
                    self.logger.info("Enviando prompt para o LLM para selecao de dominio", domain_selection_prompt=prepared_prompt)

                    try:
                        response: str = self.hugging_face_manager.generate_answer(query, prepared_prompt)
                        self.logger.debug("chamada ao LLM para selecao de dominio realizada com sucesso.")
                    except Exception as llm_error:
                        self.logger.error("Erro durante a chamada ao LLM para selecao de dominio", exc_info=True)
                        raise llm_error

                    self.logger.debug("Resposta bruta do LLM para selecao de dominio:", raw_response=response)
                    
                    response_domain_names = response.split("|")
                    response_domain_names = [name.strip() for name in response_domain_names]
                    self.logger.debug(f"Dominios selecionados: {response_domain_names}")
                    self.logger.info(f"Dominios selecionados: {response_domain_names}")
                    selected_domains = [domain for domain in domains if domain.name in response_domain_names]

                # Se o usuario selecionou domínios específicos, simplesmente retorna os objetos Domain correspondentes
                else:
                    selected_domains = [domain for domain in domains if domain.name in selected_domains]
                if selected_domains:
                    self.logger.debug(f"Valor do retorno: Lista final de dominios selecionados: {[domain.name for domain in selected_domains]}")
                    return selected_domains
                else:
                    self.logger.error("Nenhum domínio selecionado")
                    raise ValueError("Nenhum dominio selecionado")
        
        except Exception as e:
            self.logger.error(f"Erro durante a selecao automatica do dominio: {str(e)}", exc_info=True) 
            raise ValueError(f"Falha ao selecionar dominio automaticamente: {str(e)}") from e
        
    def _retrieve_documents(self, query_embedding: np.ndarray, domain: Domain) -> List[str]:
        """
        Recupera os chunks de conteúdo relevantes para a query usando o FaissManager.

        Args:
            query_embedding (np.ndarray): O embedding da query.

        Returns:
            List[str]: Uma lista de chunks de conteúdo relevantes.
        """
        self.logger.info("Recuperando os chunks de conteudo relevantes", domain=domain.name)
        if query_embedding is None or query_embedding.size == 0:
            self.logger.error("Vetor de embedding vazio ou invalido")
            raise ValueError("Vetor de embedding vazio ou inválido")
        
        try:
            self.logger.debug(f"Procurando o indice FAISS em: {domain.vector_store_path}")

            _, indices = self.faiss_manager.search_faiss_index(query_embedding=query_embedding, k=5, vector_store_path=domain.vector_store_path)
            flat_indices = indices.flatten().tolist()
            self.logger.debug(f"Valor de retorno da busca no indice FAISS", flat_indices=flat_indices)
            self.metrics_data["knn_faiss_indices"] = len(flat_indices)
            
            self.logger.debug(f"Procurando chunks no banco de dados: {domain.db_path} para os indices: {flat_indices}")
            with self.sqlite_manager.get_connection(db_path=domain.db_path) as conn:
                chunks_content = self.sqlite_manager.get_chunks_content(conn, flat_indices)
    
                self.logger.debug(f"Valor de retorno da busca no banco de dados: {len(chunks_content)} chunks.", chunks_content=chunks_content)
    
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

    def query_llm(self, query: str, domain_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Processa a query e retorna a resposta gerada pelo modelo LLM.

        Args:
            query (str): A query original.

        Returns:
            str: A resposta gerada pelo modelo de linguagem.
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
            selected_domains = self._select_domains(query, domain_names)

            selected_domain_names_log = [d.name for d in selected_domains] if selected_domains else []
            self.logger.debug(f"Dominios selecionados para recuperacao: {selected_domain_names_log}")

            query_embedding = self._process_query(query)

            chunks_content = []
            for domain in selected_domains:
                domain_chunks = self._retrieve_documents(query_embedding, domain)
                chunks_content.extend(domain_chunks)

            if not chunks_content:
                self.logger.error("Nenhum chunk de conteudo recuperado")

                self.metrics_data["success"] = False
                raise ValueError("Nenhum chunk de conteúdo recuperado")
            
            self.logger.debug(f"Chunks recuperados para contexto ({len(chunks_content)} total):", context_chunks=chunks_content)

            context_prompt = self._prepare_context_prompt(query, chunks_content)
            self.logger.debug("Prompt final sendo enviado ao LLM:", final_prompt=context_prompt)

            answer = self.hugging_face_manager.generate_answer(query, context_prompt)
            self.logger.debug("Resposta do LLM:", answer=answer)

            self.metrics_data["answer"] = answer
            self.metrics_data["success"] = True
            self.metrics_data["processing_duration"] = str(datetime.now() - self.metrics_data["start_time"])

            return self.metrics_data
        
        except Exception as e:
            self.logger.error(f"Erro ao processar a query: {str(e)}")
            
            self.metrics_data["success"] = False
            self.metrics_data["processing_duration"] = str(datetime.now() - self.metrics_data["start_time"])
            raise e
        
        