from pydantic import BaseModel, Field, PositiveInt, conint, confloat, ConfigDict
from typing import Literal, Optional, Dict, Any

class SystemConfig(BaseModel):
    storage_base_path: str = "storage"
    control_db_filename: str = "control.db"

class IngestionConfig(BaseModel):
    chunk_strategy: Literal["recursive"] = "recursive"  # Adicionar "semantic" depois
    chunk_size: PositiveInt = 1000
    chunk_overlap: conint(ge=0) = 200 # type: ignore

class EmbeddingConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Literal["cpu", "cuda"] = "cpu"
    batch_size: PositiveInt = 32
    normalize_embeddings: bool = True

class VectorStoreConfig(BaseModel):
    index_type: Literal["IndexFlatL2"] = "IndexFlatL2" # Adicionar "IndexIDMap" depois
    index_params: Optional[Dict[str, Any]] = None

class QueryConfig(BaseModel):
    retrieval_k: PositiveInt = 5
    # rerank_strategy: Literal["none"] = "none" # Adicionar depois

class LLMConfig(BaseModel):
    model_repo_id: str = "HuggingFaceH4/zephyr-7b-beta"
    max_new_tokens: PositiveInt = 1000
    temperature: confloat(ge=0.0, le=2.0) = 0.7 # type: ignore
    top_p: Optional[confloat(ge=0.0, le=1.0)] = 0.9 # type: ignore
    top_k: Optional[PositiveInt] = 50
    repetition_penalty: Optional[confloat(ge=1.0)] = 1.0 # type: ignore
    max_retries: conint(ge=0) = 3 # type: ignore
    retry_delay_seconds: PositiveInt = 2
    prompt_template: str = Field(default="""Use o seguinte contexto para responder a pergunta no final.
Se você não sabe a resposta, apenas diga que não sabe, não tente inventar uma resposta.
Mantenha a resposta concisa e diretamente ao ponto da pergunta.
Forneça a resposta *apenas* com base no contexto fornecido. Não adicione informações externas.
"""
)

class TextNormalizerConfig(BaseModel):
    """Configuration for TextNormalizer steps."""
    use_unicode_normalization: bool = True
    use_lowercase: bool = True
    use_remove_extra_whitespace: bool = True

class AppConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    text_normalizer: TextNormalizerConfig = Field(default_factory=TextNormalizerConfig)

    model_config = ConfigDict(
        # Opcional: Se desejar permitir campos extras no TOML
        # que não estão definidos nos modelos (útil durante o desenvolvimento)
        # extra = 'ignore'
        # Add any other settings from the old Config class here
    ) 