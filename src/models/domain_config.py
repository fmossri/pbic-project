from pydantic import BaseModel, Field
from typing import Optional

class DomainConfig(BaseModel):
    """
    Configurações estáticas de um domínio.
    """
    id: Optional[int] = Field(None, description="ID", json_schema_extra={"updatable": False})
    domain_id: int = Field(description="ID do domínio", json_schema_extra={"updatable": False})
    embeddings_model: str = Field(description="Modelo de embeddings do domínio", json_schema_extra={"updatable": False})
    normalize_embeddings: bool = Field(default=True, description="Normalizar embeddings", json_schema_extra={"updatable": True})
    combine_embeddings: bool = Field(default=False, description="Combinar embeddings textuais e de metadados", json_schema_extra={"updatable": True})
    embedding_weight: float = Field(default=0.7, description="Peso do embedding textual na combinação", json_schema_extra={"updatable": True})
    faiss_index_type: str = Field(description="Tipo de índice Faiss do domínio", json_schema_extra={"updatable": False})
    chunking_strategy: str = Field(description="Estratégia de chunking do domínio", json_schema_extra={"updatable": True})
    chunk_size: int = Field(default=500, description="Tamanho do chunk em caracteres", json_schema_extra={"updatable": True})
    chunk_overlap: int = Field(default=100, description="Overlap entre chunks em caracteres", json_schema_extra={"updatable": True})
    cluster_distance_threshold: float = Field(default=0.85, description="Threshold para clusterização", json_schema_extra={"updatable": True})
    chunk_max_words: int = Field(default=250, description="Máximo de palavras por chunk", json_schema_extra={"updatable": True})

    
