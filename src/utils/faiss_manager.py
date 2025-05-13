# src/utils/faiss_manager.py

import faiss
import numpy as np
from typing import List, Optional
import os
from pathlib import Path
from src.utils.logger import get_logger
from src.config.models import AppConfig

class FaissManager:
    """Componente para armazenamento e indexação de vetores usando FAISS.
       Utiliza IndexIDMap para permitir o uso de IDs personalizados no IndexFlatL2.
       Configurado com parâmetros gerais, opera em arquivos de índice específicos,
       aceitando a dimensão do embedding por método.
    """

    def __init__(self, config: AppConfig, log_domain: str = "utils"):
        """
        Inicializa o FaissManager com configurações.
        
        Args:
            config (AppConfig): Configuração do sistema.
            log_domain (str): Domínio para o logger.
        """
        self.logger = get_logger(__name__, log_domain=log_domain)
        self.config = config.model_copy(deep=True)
        self.logger.info("FaissManager inicializado.", 
                         vector_store_config=self.config.vector_store.model_dump(),
                         query_config=self.config.query.model_dump()
                        )
        
    def update_config(self, new_config: AppConfig) -> None:
        """
        Atualiza a configuração do DomainManager com base na configuração fornecida.

        Args:
            config (AppConfig): A nova configuração a ser aplicada.
        """
        self.config = new_config.model_copy(deep=True)
        self.logger.info("Configuracoes do DomainManager atualizadas com sucesso")

    def _create_vector_store(self, index_path: str, dimension: int) -> faiss.Index:

        match self.config.vector_store.index_type:
            case "IndexFlatL2":
                base_index = faiss.IndexFlatL2(dimension)
                # IndexIDMap como wrapper para permitir uso de IDs personalizados
                index = faiss.IndexIDMap(base_index)

            #TODO: Implementar outros tipos de índice

        self._save_state(index, index_path)
        self.logger.info(f"Novo indice FAISS (IDMap, vazio) salvo em: {index_path}")

        return index

    def _initialize_index(self, index_path: str, dimension: int) -> faiss.Index:
        """Inicializa ou carrega um índice FAISS (IndexIDMap(IndexFlatL2)) específico.
        
        Args:
            index_path (str): O caminho completo para o arquivo de índice FAISS.
            dimension (int): A dimensão esperada para os vetores do índice.

        Returns:
            faiss.Index: O índice FAISS IndexIDMap carregado ou recém-criado.
            
        Raises:
            FileNotFoundError: Se o diretório não puder ser criado.
            Exception: Para outros erros de FAISS ou I/O.
        """
        self.logger.info(f"Inicializando índice FAISS em {index_path} com dimensão {dimension}")
        index_file = Path(index_path)
        index_dir = index_file.parent
        
        try:
            index_dir.mkdir(parents=True, exist_ok=True)

            if index_file.exists():
                index = faiss.read_index(str(index_file))
                # Valida a dimensão do índice carregado
                if index.d != dimension:
                     self.logger.error(f"Dimensão do indice carregado ({index.d}) diferente da esperada ({dimension}) em {index_path}")
                     raise ValueError(f"Dimensão do índice carregado ({index.d}) diferente da esperada ({dimension}) em {index_path}")
                self.logger.debug(f"Índice FAISS carregado: {index_path}, n_vectors={index.ntotal}, dimension={index.d}")
                return index
            # Se o arquivo de índice não existir, cria um novo índice
            else:
                self.logger.warning(f"Indice FAISS não encontrado em {index_path}. Criando novo indice IndexIDMap(IndexFlatL2) com dimensão {dimension}.")
                index = self._create_vector_store(index_path, dimension)

                return index
        except Exception as e:
            self.logger.error(f"Erro ao inicializar o indice FAISS (IDMap) em {index_path}: {e}", exc_info=True)
            raise e

    def add_embeddings(self, embeddings: np.ndarray, ids: List[int], index_path: str, dimension: int) -> None:
        """
        Adiciona embeddings com IDs específicos a um índice FAISS.
        Converte a lista de IDs fornecida para o formato np.int64 necessário.
        
        Args:
            embeddings (np.ndarray): Array numpy de embeddings (N, D).
            ids (List[int]): Lista de IDs inteiros (N,) correspondentes aos embeddings.
            index_path (str): Caminho completo para o arquivo de índice FAISS.
            dimension (int): A dimensão esperada dos embeddings.
        """
        # Valida os embeddings
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2 or embeddings.shape[1] != dimension:
            msg = f"Embeddings inválidos. Esperado np.ndarray(N, {dimension}), recebido {type(embeddings)} shape {getattr(embeddings, 'shape', 'N/A')}."
            self.logger.error(msg)
            raise ValueError(msg)
        
        # Valida os IDs
        if not isinstance(ids, list):
            msg = f"IDs devem ser uma lista de ints, recebido {type(ids)}."
            self.logger.error(msg)
            raise TypeError(msg)
        if len(ids) != embeddings.shape[0]:
            msg = f"Número de IDs ({len(ids)}) não corresponde ao número de embeddings ({embeddings.shape[0]})."
            self.logger.error(msg)
            raise ValueError(msg)
        
        if not all(isinstance(item, int) for item in ids):
            # Encontra o primeiro não-int para uma mensagem de erro mais informativa
            first_non_int = next((item for item in ids if not isinstance(item, int)), None)
            type_found = type(first_non_int).__name__ if first_non_int is not None else "unknown"
            msg = f"Todos os IDs na lista devem ser inteiros. Encontrado tipo '{type_found}'."
            self.logger.error(msg, first_offending_value=first_non_int)
            raise TypeError(msg)

        
        try:
            # Converte List[int] => np.ndarray(dtype=np.int64) para o uso do FAISS
            np_ids = np.array(ids, dtype=np.int64)
            if np_ids.shape[0] != len(ids):
                 raise ValueError("Falha na conversão de IDs para NumPy array.")
        except (ValueError, TypeError) as e:
            msg = f"Falha ao converter a lista de IDs para np.int64. Verifique se a lista contém apenas inteiros. {e}"
            self.logger.error(msg)
            raise TypeError(msg) from e 
            
        self.logger.debug(f"Adicionando {embeddings.shape[0]} embeddings (dim={dimension}) com IDs ao índice: {index_path}")
        try:
            index = self._initialize_index(index_path, dimension)
            index.add_with_ids(embeddings, np_ids)
            self.logger.debug(f"{embeddings.shape[0]} embeddings com IDs adicionados. Total agora: {index.ntotal}")
                
            self._save_state(index, index_path)
            self.logger.debug(f"Embeddings com IDs adicionados com sucesso ao índice {index_path}.")
        except Exception as e:
            self.logger.error(f"Erro ao adicionar embeddings com IDs ao índice FAISS {index_path}: {e}", exc_info=True)
            raise e

    def search_faiss_index(self, query_embedding: np.ndarray, index_path: str, dimension: int, k: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Realiza uma busca em um índice FAISS (IDMap) específico e retorna os IDs dos k vizinhos mais próximos.
        
        Args:
            query_embedding (np.ndarray): Vetor de embedding da query (1, D) ou (D,).
            index_path (str): Caminho completo para o arquivo de índice FAISS.
            dimension (int): A dimensão esperada do query_embedding e do índice.
            k (Optional[int]): Número de vizinhos a retornar. Usa config.query.retrieval_k se None.

        Returns:
            tuple[np.ndarray, np.ndarray]: Distâncias (1, k) e os IDs (1, k) dos vizinhos (np.int64).
                                           Os IDs retornados são aqueles fornecidos durante add_embeddings.
        """
        k = k if k is not None else self.config.query.retrieval_k
        if k <= 0:
             self.logger.warning(f"Solicitado k={k} para busca. Usando k=1.")
             k = 1

        if not isinstance(query_embedding, np.ndarray):
             msg = f"query_embedding deve ser um np.ndarray, recebido {type(query_embedding)}."
             self.logger.error(msg)
             raise TypeError(msg)
        
        # Garante que a query seja 2D (1, D) para a busca no índice
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Valida a dimensão do embedding da query contra a dimensão do índice
        if query_embedding.shape[1] != dimension:
            msg = f"Dimensão do query_embedding ({query_embedding.shape[1]}) diferente da dimensão esperada ({dimension})."
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.info(f"Realizando busca por similaridade (IDMap, dim={dimension}) no índice {index_path}", top_k=k)
        try:
            index = self._initialize_index(index_path, dimension)
            if index.ntotal == 0:
                 self.logger.warning(f"Busca realizada em índice (IDMap) vazio: {index_path}. Retornando vazio.")
                 return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)
            
            # Ajusta k se for maior que o número de vetores no índice
            actual_k = min(k, index.ntotal)
            if actual_k < k:
                 self.logger.warning(f"Solicitado k={k}, mas índice contém apenas {index.ntotal} vetores. Usando k={actual_k}.")

            distances, ids = index.search(query_embedding, actual_k)

            self.logger.debug("Busca no índice FAISS (IDMap) realizada com sucesso.", 
                              k_requested=k, k_actual=actual_k, 
                              distances_shape=distances.shape, ids_shape=ids.shape,
                              returned_ids=ids.flatten().tolist()
                             )
            
            return distances, ids 
        except FileNotFoundError as e:
             # Lançado por _initialize_index se o diretório não puder ser criado
             self.logger.error(f"Erro de FileNotFoundError ao buscar no índice {index_path}. Caminho pode ser inválido: {e}", exc_info=True)
             raise
        except Exception as e:
            self.logger.error(f"Erro ao realizar busca no indice FAISS {index_path}: {e}", exc_info=True)
            raise e
        
    # Modified signature
    def _save_state(self, index: faiss.Index, index_path: str) -> None:
        """Salva o estado de um índice FAISS específico."""
        self.logger.info(f"Salvando estado do índice FAISS em: {index_path}")
        try:
            faiss.write_index(index, index_path)
            self.logger.info(f"Estado do indice FAISS salvo com sucesso em {index_path}")
        except Exception as e:
            self.logger.error(f"Erro ao salvar o estado do indice FAISS em {index_path}: {e}", exc_info=True)
            raise e
