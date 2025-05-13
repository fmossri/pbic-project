import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Any

from src.models import DocumentFile, Chunk
from src.utils import TextNormalizer, EmbeddingGenerator, FaissManager, SQLiteManager
from src.utils.logger import get_logger
from .document_processor import DocumentProcessor
from .chunking_strategy.chunking_manager import ChunkingManager
from src.config import AppConfig, check_config_changes
class DataIngestionOrchestrator:
    """Componente principal para gerenciar o processamento de arquivos PDF."""

    DEFAULT_LOG_DOMAIN = "Ingestao de dados"
    
    def __init__(self, config: AppConfig):
        """
        Inicializa o orquestrador com os componentes necessários.
        
        Args:
            db_path (str, optional): Caminho para o arquivo de banco de dados SQLite.
            index_path (str, optional): Caminho para o diretório de índices FAISS.
            chunk_size (int): Tamanho de cada chunk de texto.
            overlap (int): Sobreposição entre chunks.
        """
        self.logger = get_logger(__name__, log_domain=self.DEFAULT_LOG_DOMAIN)
        self.logger.info("Inicializando o DataIngestionOrchestrator")
        
        self.metrics_data = {}
        self.config = config.model_copy(deep=True)
        self.document_processor = DocumentProcessor(log_domain=self.DEFAULT_LOG_DOMAIN)
        self.text_chunker = ChunkingManager(config, log_domain=self.DEFAULT_LOG_DOMAIN)
        self.text_normalizer = TextNormalizer(config.text_normalizer, log_domain=self.DEFAULT_LOG_DOMAIN)
        self.embedding_generator = EmbeddingGenerator(config.embedding, log_domain=self.DEFAULT_LOG_DOMAIN)
        self.sqlite_manager = SQLiteManager(config.system, log_domain=self.DEFAULT_LOG_DOMAIN)
        self.faiss_manager = FaissManager(config, log_domain=self.DEFAULT_LOG_DOMAIN)
        self.document_hashes: Dict[str, str] = {}

    def update_config(self, new_config: AppConfig) -> None:
        """
        Atualiza a configuração do DataIngestionOrchestrator com base na configuração fornecida.

        Args:
            config (AppConfig): A nova configuração a ser aplicada.
        """
        self.logger.info("Iniciando a atualizacao das configuracoes do DataIngestionOrchestrator")
        update_fields = check_config_changes(self.config, new_config)
        
        if not update_fields:
            self.logger.info("Nenhuma alteracao na configuracao detectada")
            return
        
        self.logger.info("Alteracoes na configuracao detectadas", update_fields=update_fields)

        self.logger.info("Atualizando os componentes do DataIngestionOrchestrator")
        for field in update_fields:
            match field:
                case "ingestion":
                    self.text_chunker.update_config(new_config)
                case "embedding":
                    self.embedding_generator.update_config(new_config.embedding)
                case "vector_store":
                    self.faiss_manager.update_config(new_config)
                case "text_normalizer":
                    self.text_normalizer.update_config(new_config.text_normalizer)
                case "system":
                    self.sqlite_manager.update_config(new_config.system)

        self.config = new_config.model_copy(deep=True)
        self.logger.info("Configuracoes do DataIngestionOrchestrator atualizadas com sucesso")

    def _find_original_document(self, duplicate_hash: str, conn: sqlite3.Connection) -> DocumentFile:
        """
        Encontra o documento original do hash duplicado.
        
        Args:
            duplicate_hash (str): Hash do documento duplicado
            conn (sqlite3.Connection): Conexão com o banco de dados
            
        Returns:
            DocumentFile: Documento original encontrado
            
        Raises:
            ValueError: Se o documento original não for encontrado
        """
        self.logger.info(f"Procurando o documento original com o hash: {duplicate_hash}")
        # Primeiro procura no dicionário de hashes
        try:
            for filename, hash_value in self.document_hashes.items():
                if hash_value == duplicate_hash:
                    # Cria um DocumentFile com as informações básicas
                    return DocumentFile(
                        id=None,
                        hash=hash_value,
                        name=filename,
                        path=os.path.join(os.path.dirname(filename), filename),
                        total_pages=0
                    )
            
        # Se não encontrou no dicionário, procura no banco de dados
            cursor = conn.execute("SELECT * FROM document_files WHERE hash = ?", (duplicate_hash,)) 
            result = cursor.fetchone()
            if result:
                return DocumentFile(
                    id=result[0],
                    hash=result[1],
                    name=result[2],
                    path=result[3],
                    total_pages=result[4]
                )
        except Exception as e:
            self.logger.error(f"Erro ao encontrar o documento original: {e}")
            raise ValueError(f"Documento original não encontrado para o hash: {duplicate_hash}")

    def _is_duplicate(self, document_hash: str, conn: sqlite3.Connection) -> bool:
        """
        Verifica se um documento é duplicado.
        
        Args:
            document_hash (str): Hash do documento a ser verificado
            
        Returns:
            bool: True se o documento for duplicado, False caso contrário
        """
        self.logger.debug(f"Verificando se o documento é duplicado: {document_hash}")    
        try:
            # Verifica se já existe um documento com o mesmo hash
            for existing_hash in self.document_hashes.values():
                if existing_hash == document_hash:
                    self.logger.warning(f"Documento duplicado encontrado")
                    return True
            
            cursor = conn.execute("SELECT * FROM document_files WHERE hash = ?", (document_hash,))
            result = cursor.fetchone()
            if result:
                self.logger.warning(f"Documento duplicado encontrado")
                return True
            return False
        
        except Exception as e:
            self.logger.error(f"Erro ao verificar se o documento e duplicado: {e}")
            raise e
        
    def _set_metrics_data(self) -> None:
        """
        Obtém os dados de métricas para o processo de ingestão de dados.
        
        Returns:
            Dict[str, Any]: Dicionário contendo os dados de métricas
        """
        start_time = datetime.now()
        self.metrics_data["process"] = "Ingestão de dados"
        self.metrics_data["start_time"] = start_time
        self.metrics_data["text_chunker"] = type(self.text_chunker.chunker).__name__
        self.metrics_data["chunk_size"] = self.text_chunker.config.ingestion.chunk_size
        self.metrics_data["chunk_overlap"] = self.text_chunker.config.ingestion.chunk_overlap
        self.metrics_data["embedding_model"] = self.embedding_generator.config.model_name
        self.metrics_data["embedding_dimension"] = None
        self.metrics_data["faiss_index_path"] = None
        self.metrics_data["faiss_index_type"] = self.faiss_manager.config.vector_store.index_type
        self.metrics_data["database_path"] = None
        self.metrics_data["processed_files"] = 0
        self.metrics_data["processed_pages"] = 0
        self.metrics_data["processed_chunks"] = 0
        self.metrics_data["processed_embeddings"] = 0
        self.metrics_data["duplicate_files"] = 0
        self.metrics_data["invalid_files"] = 0        
    
    def _list_pdf_files(self, directory_path: str) -> List[DocumentFile]:
        """
        Lista todos os arquivos PDF em um diretório.

        Args:
            directory_path (str): Caminho para o diretório

        Returns:
            List[str]: Lista de nomes de arquivos PDF

        Raises:
            FileNotFoundError: Se o diretório não existir
            NotADirectoryError: Se o caminho não for um diretório
            ValueError: Se nenhum arquivo PDF for encontrado
        """
        self.logger.info("Listando arquivos PDF no diretorio", directory_path=directory_path)

        if not os.path.exists(directory_path):
            self.logger.error("Diretorio nao encontrado", directory_path=directory_path)
            raise FileNotFoundError(f"Diretório não encontrado: {directory_path}")

        if not os.path.isdir(directory_path):
            self.logger.error("Caminho nao e um diretorio", directory_path=directory_path)
            raise NotADirectoryError(f"Caminho não é um diretório: {directory_path}")

        pdf_files = []
        try:
            with os.scandir(directory_path) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.lower().endswith('.pdf'):
                        file = DocumentFile(id = None, hash = None, name=entry.name, path=entry.path, total_pages=0)
                        pdf_files.append(file)
        except Exception as e:
            self.logger.error(f"Erro ao listar arquivos PDF: {e}")
            raise e

        if not pdf_files:
            self.logger.error("Nenhum arquivo PDF encontrado no diretorio", directory_path=directory_path)
            raise ValueError(f"Nenhum arquivo PDF encontrado em: {directory_path}")
        
        self.logger.info(f"{len(pdf_files)} arquivos PDF encontrados no diretorio", directory_path=directory_path)
        return pdf_files
    
    def process_directory(self, directory_path: str, domain_name: str = None) -> Dict[str, Any]:
        """
        Processa todos os arquivos PDF em um diretório.
        
        Args:
            directory_path (str): Caminho para o diretório contendo PDFs
            
        Returns:
            Dict[str, List[Document]]: Dicionário mapeando nomes de arquivos para seus chunks de texto
            
        Raises:
            FileNotFoundError: Se o diretório não existir
            NotADirectoryError: Se o caminho não for um diretório
            ValueError: Se não houver arquivos PDF no diretório
        """
        self._set_metrics_data()
        if not os.path.exists(directory_path):
            self.logger.error("Diretorio nao encontrado", directory_path=directory_path)
            raise FileNotFoundError(f"Diretório não encontrado: {directory_path}")
        if not os.path.isdir(directory_path):
            self.logger.error("Caminho nao e um diretorio", directory_path=directory_path)
            raise NotADirectoryError(f"Caminho não é um diretório: {directory_path}")
        
        self.logger.info("Iniciando o processamento do diretorio", directory_path=directory_path)

        # Busca o domínio de conhecimento no banco de dados de controle
        with self.sqlite_manager.get_connection(control=True) as conn:
            try:
                self.sqlite_manager.begin(conn)
                [domain] = self.sqlite_manager.get_domain(conn, domain_name)

                if not domain:
                    self.logger.error(f"Domínio de conhecimento não encontrado: {domain_name}")
                    raise ValueError(f"Domínio de conhecimento não encontrado: {domain_name}")
                
                #Antes da primeira ingestão, a dimensão dos embeddings é 0. Define o modelo e a dimensão dos embeddings do domínio
                if domain.embeddings_dimension == 0:
                    embedding_dimension = self.embedding_generator.embedding_dimension
                    
                    # Atualizar o objeto domain localmente (se necessário para lógica subsequente)
                    domain.embeddings_dimension = embedding_dimension
                    
                    self.sqlite_manager.update_domain(domain, conn, {"embeddings_dimension": embedding_dimension})
                
                self.metrics_data["database_path"] = domain.db_path
                self.metrics_data["embedding_dimension"] = domain.embeddings_dimension
                self.metrics_data["faiss_index_path"] = domain.vector_store_path
                conn.commit()
            except Exception as e:
                self.logger.error(f"Erro ao obter o domínio de conhecimento: {e}")
                raise e
            
        pdf_files = self._list_pdf_files(directory_path)

        self.metrics_data["total_files"] = len(pdf_files)
        total_chunk_size = 0

        with self.sqlite_manager.get_connection(db_path=domain.db_path) as conn:
            self.logger.info(f"Iniciando o processamento dos arquivos PDF")

            file_counter = 0
            duplicate_counter = 0
            for file in pdf_files:
                self.sqlite_manager.begin(conn)
                file_start_time = datetime.now()
                file_metrics = {}
                file_counter += 1
                file_metrics["filename"] = file.name
                file_metrics["file_path"] = file.path
                
                self.logger.info(f"Processando arquivos: {file_counter}/{len(pdf_files)}")
                self.logger.info(f"Iniciando o processamento do arquivo: {file.name}", file_path=file.path)
                
                try:
                    # processa o documento, adicionando as páginas, hash e total de páginas ao objeto DocumentFile
                    self.document_processor.process_document(file, normalize_whitespace=self.config.text_normalizer.use_remove_extra_whitespace)

                    file_metrics["total_pages"] = len(file.pages)
                    # Verifica se o documento é uma réplica
                    if self._is_duplicate(file.hash, conn):
                        file_metrics["is_duplicate"] = True
                        file_metrics["file_hash"] = file.hash
                        duplicate_counter += 1
                        original_file = self._find_original_document(file.hash, conn)
                        if original_file:
                            file_metrics["original_file"] = original_file.name
                            self.logger.warning(f"Arquivo duplicado: {file.name} hash: {file.hash}", file_path=file.path)
                            self.logger.warning(f"Arquivo original: {original_file.name} hash: {original_file.hash}", file_path=original_file.path)

                        self.logger.info(f"Descartando alteracoes da transacao", file_path=file.path)
                        file_metrics["file_processing_duration"] = str(datetime.now() - file_start_time)
                        self.metrics_data[file.name] = file_metrics
                        self.metrics_data["duplicate_files"] += 1

                        conn.rollback()
                        continue

                    file_metrics["is_duplicate"] = False
                    file_metrics["file_hash"] = file.hash

                    # Adiciona o hash do documento ao dicionário de hashes
                    self.document_hashes[file.name] = file.hash

                    # Insere o documento no banco de dados e recebe seu id
                    file.id = self.sqlite_manager.insert_document_file(file, conn)
                    file_metrics["file_id"] = file.id

                    if not file.pages:
                        self.logger.error(f"Nenhuma pagina encontrada", file_path=file.path)
                        self.logger.error(f"Descartando alteracoes da transacao")
                        file_metrics["invalid_file"] = True
                        file_metrics["file_processing_duration"] = str(datetime.now() - file_start_time)
                        self.metrics_data[file.name] = file_metrics
                        self.metrics_data["invalid_files"] += 1

                        # Descarta e passa ao próximo arquivo
                        conn.rollback()
                        continue

                    self.logger.info(f"Total de paginas: {len(file.pages)}")
                    self.logger.info(f"Iniciando a criacao de chunks")
                    # Cria os chunks para o documento de acordo com a estratégia selecionada    
                    document_chunks : List[Chunk] = self.text_chunker.create_chunks(file)
                    if not document_chunks:
                        self.logger.error(f"Nenhum chunk gerado", file_path=file.path)
                        self.logger.error(f"Descartando alteracoes da transacao")
                        file_metrics["invalid_file"] = True
                        file_metrics["file_processing_duration"] = str(datetime.now() - file_start_time)
                        self.metrics_data[file.name] = file_metrics
                        self.metrics_data["invalid_files"] += 1
                        # Descarta e passa ao próximo arquivo
                        conn.rollback()
                        continue

                    self.logger.info(f"{file.name} - Total de chunks: {len(document_chunks)}", file_path=file.path)
                    file_metrics["total_chunks"] = len(document_chunks)
                    total_chunk_size += sum(len(chunk.content) for chunk in document_chunks)

                    #Insere os chunks no banco de dados
                    chunk_ids = self.sqlite_manager.insert_chunks(document_chunks, file.id, conn)
                    file_metrics["chunks_inserted"] = True

                    # Normaliza os chunks
                    chunks_content = [chunk.content for chunk in document_chunks]
                    normalized_chunks = self.text_normalizer.normalize(chunks_content)

                    #Gera os embeddings
                    embedding_vectors = self.embedding_generator.generate_embeddings(normalized_chunks)                    
                    if embedding_vectors.size == 0:
                        self.logger.error(f"Nenhum embedding gerado", file_path=file.path)
                        self.logger.error(f"Descartando alteracoes da transacao", file_path=file.path)
                        file_metrics["invalid_file"] = True
                        file_metrics["file_processing_duration"] = str(datetime.now() - file_start_time)
                        self.metrics_data[file.name] = file_metrics
                        self.metrics_data["invalid_files"] += 1
                        conn.rollback()
                        continue

                    file_metrics["total_embeddings"] = len(embedding_vectors)
                    #Adiciona os embeddings ao índice FAISS, recebendo os índices dos embeddings em ordem
                    self.faiss_manager.add_embeddings(embedding_vectors, chunk_ids, domain.vector_store_path, domain.embeddings_dimension)
                    file_metrics["embeddings_added"] = True
                    
                    self.logger.info(f"Salvando alteracoes no banco de dados", file_path=file.path)
                    #Salva as alterações no banco de dados
                    conn.commit()

                    file_metrics["commit_success"] = True
                    file_metrics["file_processing_duration"] = str(datetime.now() - file_start_time)
                    self.metrics_data[file.name] = file_metrics
                    self.metrics_data["processed_files"] += 1
                    self.metrics_data["processed_pages"] += len(file.pages)
                    self.metrics_data["processed_chunks"] += len(document_chunks)
                    self.metrics_data["processed_embeddings"] += embedding_vectors.shape[0]

                    self.logger.info(f"Processamento do arquivo concluido com sucesso", file_path=file.path)

                except Exception as e:
                    self.logger.error(f"Erro ao processar o arquivo: {e}", file_path=file.path)
                    self.logger.error(f"Descartando alteracoes da transacao", file_path=file.path)
                    file_metrics["invalid_file"] = True

                    file_metrics["commit_success"] = False
                    file_metrics["file_processing_duration"] = str(datetime.now() - file_start_time)
                    self.metrics_data["invalid_files"] += 1
                    self.metrics_data[file.name] = file_metrics
                    conn.rollback()
                    continue

        with self.sqlite_manager.get_connection(control=True) as conn:
            try:
                self.sqlite_manager.begin(conn)
                domain.total_documents += self.metrics_data["processed_files"]
                self.sqlite_manager.update_domain(domain, conn, {"total_documents": domain.total_documents})
                conn.commit()
            except Exception as e:
                self.logger.error(f"Erro ao atualizar o domínio de conhecimento: {e}", domain_name=domain.name, domain_total_documents=domain.total_documents)
                raise e

        self.metrics_data["duration"] = str(datetime.now() - self.metrics_data["start_time"])
        self.metrics_data["start_time"] = self.metrics_data["start_time"].strftime("%Y-%m-%d %H:%M:%S")
        self.metrics_data["avg_chunk_size"] = total_chunk_size / self.metrics_data["processed_chunks"] if self.metrics_data["processed_chunks"] > 0 else None
        if self.metrics_data["processed_files"] == 0:
            self.logger.warning(f"Nenhum arquivo processado.")
            
        self.logger.info(f"Processamento do diretorio concluido em {self.metrics_data['duration']}")
        self.logger.info(f"Arquivos processados: {self.metrics_data['processed_files']}, Duplicatas: {self.metrics_data['duplicate_files']}, Invalidos: {self.metrics_data['invalid_files']}")

        return self.metrics_data

 