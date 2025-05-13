import numpy as np

from typing import Optional, Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from sklearn.cluster import AgglomerativeClustering


from langchain_core.documents import Document

from src.models import Chunk, DocumentFile
from src.config.models import AppConfig
from src.utils.logger import get_logger
from .chunking_strategy import ChunkingStrategy

class SemanticClusterStrategy(ChunkingStrategy):
    def __init__(self, config: AppConfig, log_domain: str):
        """
        Inicializa o SemanticClusterStrategy com base na configuração fornecida.

        Args:
            config (AppConfig): Objeto de configuração contendo os parâmetros de configuração.
            log_domain (str): Domínio para o logger.
        """
        super().__init__(config, log_domain)

        self.logger.info("Carregando o modelo SentenceTransformer: sentence-transformers/LaBSE")
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=self.config.ingestion.chunk_size, chunk_overlap=self.config.ingestion.chunk_overlap)
        self.logger.info("Modelo SentenceTransformer carregado com sucesso.")

    def _chunk_text_small(self, pages: List[Document]) -> List[Document]:
        """
        Quebra o texto em chunks pequenos para processamento intermediário.
        Estes chunks serão posteriormente agrupados em chunks semânticos maiores.
        
        Args:
            pages (List[Document]): Lista de objetos langchain.Document representando páginas
            
        Returns:
            List[Document]: Lista de chunks pequenos com metadados de página
        """
        small_chunks: List[Document] = []

        self.logger.debug("Passo-1: Criando chunks pequenos para processamento intermediário")
        for page in pages:
            if not page.page_content:
                self.logger.warning(f"Página vazia encontrada. Pulando processamento da página: {page.metadata.get('page', 'unknown')}")
                continue

            # Cria chunks a partir do conteúdo da página
            page_chunks = self.splitter.create_documents(
                [page.page_content], 
                [{"page_number": page.metadata.get('page')}]
            )
            small_chunks.extend(page_chunks)

        # Adiciona o index_in_doc a cada chunk
        for i, chunk in enumerate(small_chunks):
            chunk.metadata.update({'index_in_doc': i})

        self.logger.info(f"{len(small_chunks)} chunks pequenos criados")
        return small_chunks

    def _enrich_small_chunks(self, chunks: List[Document]) -> List[str]:
        """
        Enriquece os chunks pequenos com metadados, adicionando informações de página.
        
        Args:
            chunks (List[Document]): Lista de chunks pequenos para enriquecer
            
        Returns:
            List[str]: Lista de conteúdos enriquecidos para clustering
        """
        self.logger.debug("Passo 2: Enriquecendo chunks pequenos com metadados: page_number e index_in_doc")
        enriched_contents : List[str] = []

        for chunk in chunks:
            prepend_string = f"[página: {chunk.metadata['page_number']}, índice: {chunk.metadata['index_in_doc']}]"
            enriched_content = f"{prepend_string} {chunk.page_content}"
            enriched_contents.append(enriched_content)

        self.logger.info(f"Total de {len(enriched_contents)} chunks enriquecidos com metadados")
        return enriched_contents

    def _prepare_metadata_strings(self, small_chunks: List[Document]) -> List[str]:
        """
        Prepara as strings dos metadados
        """
        metadata_strings = [
            f"Página: {item['metadata'].get('page_num', '')} | "
            f"Índice: {item['metadata'].get('index_in_doc', '')}"
            for item in small_chunks
        ]
        return metadata_strings

    def _combine_embeddings(self, text_embeddings: np.ndarray, metadata_embeddings: np.ndarray) -> np.ndarray:
        """
        Combina os embeddings de texto e metadados usando pesos.
        
        Args:
            text_embeddings: Array numpy contendo os embeddings do texto
            metadata_embeddings: Array numpy contendo os embeddings dos metadados
            
        Returns:
            Array numpy com os embeddings combinados
        """
        self.logger.info("Passo 5: Combinando embeddings textuais e de metadados com peso: %.2f", self.config.embedding.weight)
        # Define o peso para os metadados (ajuste conforme necessário)
        metadata_weight = self.config.embedding.weight
        
        # Combina os embeddings usando pesos
        combined_embeddings = (
            (1 - metadata_weight) * text_embeddings + 
            metadata_weight * metadata_embeddings
        )
        
        self.logger.info(f"Combined_embeddings shape: {combined_embeddings.shape}")
        return combined_embeddings

    def _cluster_small_chunks(self, combined_embeddings: np.ndarray, enriched_chunks: List[str]) -> Dict:
        """
        Agrupa os chunks em clusters
        """
        self.logger.info("Passo 6: Iniciando o clustering hierárquico com distância limite: %.2f", self.config.clustering.distance_threshold)
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=self.config.clustering.distance_threshold)
        labels = clustering.fit_predict(combined_embeddings)
        self.logger.info(f"Clustering concluído. {len(set(labels))} clusters identificados.")

        self.logger.info("Passo 7: Agrupando chunks pequenos em clusters")
        clusters = {}
        for label, chunk in zip(labels, enriched_chunks):
            clusters.setdefault(label, []).append(chunk)
        self.logger.info(f"Chunks pequenos agrupados em {len(clusters)} clusters.")
        
        return clusters

    def _chunk_clusters(self, clusters: Dict) -> List[str]:
        """
        Cria chunks grandes a partir dos clusters
        
        Args:
            clusters (Dict): Dicionário de clusters
            
        Returns:
            List[str]: Lista de chunks grandes
        """
        self.logger.info("Passo 8: Criando chunks grandes com tamanho otimizado a partir dos clusters")
        big_chunks = []
        max_words = self.config.chunking.max_words
        for cluster_small_chunks in clusters.values():
            chunk = " ".join(cluster_small_chunks)
            words = chunk.split()
            if len(words) > max_words:
                temp_chunk = ""
                temp_word_count = 0
                for small_chunk in cluster_small_chunks:
                    small_chunk_word_count = len(small_chunk.split())
                    if temp_word_count + small_chunk_word_count <= max_words:
                        temp_chunk += " " + small_chunk
                        temp_word_count += small_chunk_word_count
                    else:
                        big_chunks.append(temp_chunk.strip())
                        temp_chunk = small_chunk
                        temp_word_count = small_chunk_word_count
                if temp_chunk:
                    big_chunks.append(temp_chunk.strip())
            else:
                big_chunks.append(chunk.strip())

        self.logger.info(f"Processo de chunking finalizado. {len(big_chunks)} chunks gerados.")
        return big_chunks

    def _enrich_cluster_chunks(self, chunks: List[str], keywords: List[List[str]], file_name: str) -> List[str]:
        """
        Enriquecimento de chunks com keywords e título do arquivo
        
        Args:
            chunks (List[str]): Lista de chunks para enriquecer
            keywords (List[List[str]]): Lista de listas de keywords para cada chunk
            file_name (str): Nome do arquivo
            
        Returns:
            List[str]: Lista de chunks enriquecidos
        """
        self.logger.info("Passo 10: Enriquecendo chunks com keywords e título do arquivo")
        enriched_cluster_chunks = []
        try:
            for chunk_keywords, chunk in zip(keywords, chunks):
                # Formata as keywords como string
                keywords_str = ", ".join(chunk_keywords) if chunk_keywords else ""
                enriched_chunk = f"[documento: {file_name}, keywords: {keywords_str}] {chunk}"
                enriched_cluster_chunks.append(enriched_chunk)

            self.logger.info(f"Total de {len(enriched_cluster_chunks)} chunks enriquecidos com keywords e título do arquivo")
            return enriched_cluster_chunks
        except Exception as e:
            self.logger.error(f"Erro ao enriquecer chunks: {str(e)}")
            # Em caso de erro, retorna os chunks originais sem enriquecimento
            return chunks

    def create_chunks(self, file: DocumentFile) -> List[Chunk]:
        """
        Realiza clustering semântico de segmentos de texto com o uso de metadados, 
        e cria chunks a partir dos clusters.

        Args:
            file (DocumentFile): Arquivo a ser dividido em chunks.

        Returns:
            List[Chunk]: Lista de chunks semanticamente agrupados.
        """
        self.logger.info("Iniciando o processo de cluster chunking semântico com metadados")
        
        # Verificar as principais chaves do JSON
        self.logger.info(f"Arquivo: {file.name}")
        self.logger.info(f"Páginas: {len(file.pages)}")
        pages = file.pages

        if not file or not file.pages:
            self.logger.error("Documento inválido ou nenhuma página encontrada no arquivo.")
            raise ValueError("Documento inválido ou nenhuma página encontrada no arquivo.")

        # Cria chunks pequenos para processamento intermediário
        small_chunks = self._chunk_text_small(pages)

        # Enriquecimento das sentenças com metadados
        enriched_small_chunks = self._enrich_small_chunks(small_chunks)

        # Geração de embeddings textuais
        self.logger.info("Passo 3: Gerando embeddings textuais para os chunks pequenos")
        text_embeddings = self.embedding_model.encode(enriched_small_chunks, self.config.embedding.batch_size)

        # Preparação das strings dos metadados
        metadata_strings = self._prepare_metadata_strings(small_chunks)
        # Geração de embeddings de metadados
        self.logger.info("Passo 4: Gerando embeddings para os metadados dos chunks pequenos")
        metadata_embeddings = self.embedding_model.encode(metadata_strings, self.config.embedding.batch_size)

        # Combinação dos embeddings

        combined_embeddings = self._combine_embeddings(text_embeddings, metadata_embeddings)

        # Clusteriza os chunks pequenos
        clusters = self._cluster_small_chunks(combined_embeddings, enriched_small_chunks)

        # Forma chunks grandes a partir dos clusters
        big_chunks = self._chunk_clusters(clusters)

        # Gerando keywords para cada chunk
        keywords = self._generate_keywords(big_chunks)
        self.logger.info(f"Keywords: {keywords}")

        enriched_cluster_chunks = self._enrich_cluster_chunks(big_chunks, keywords, file.name)

        self.logger.info("Passo 11: Criando objetos Chunk finais")
        final_chunks = []
        for chunk_keywords, chunk in zip(keywords, enriched_cluster_chunks):
            page_list = []
            index_list = []
            # Extrai todas as páginas e índices de cada chunk
            try:
                # Encontra todas as seções de metadados no chunk
                metadata_sections = chunk.split('[')
                for section in metadata_sections:
                    if 'página:' in section and 'índice:' in section:
                        try:
                            page_part = section.split('página:')[1].split(',')[0].strip()
                            index_part = section.split('índice:')[1].split(']')[0].strip()
                            page_list.append(int(page_part))
                            index_list.append(int(index_part))
                        except (IndexError, ValueError) as e:
                            self.logger.warning(f"Falha ao parsear a seção de metadados: {section[:100]}... Error: {e}")
                            continue
            except Exception as e:
                self.logger.error(f"Erro ao processar metadados do chunk: {e}")
                continue

            chunk_obj = Chunk(
                document_id=file.id,
                content=chunk,
                metadata={
                    "page_list": page_list,
                    "index_list": index_list,
                    "keywords": chunk_keywords,
                }
            )
            final_chunks.append(chunk_obj)

        self.logger.info(f"Total de {len(final_chunks)} chunks criados. Estratégia de cluster chunking semântico concluída com sucesso.")
        return final_chunks