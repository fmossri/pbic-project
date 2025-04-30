from pydantic import BaseModel, Field, PositiveInt, conint, confloat
from typing import Literal, Optional, Dict, Any

class SystemConfig(BaseModel):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_file: str = "logs/app.log"
    storage_base_path: str = "storage/domains"
    default_domain: Optional[str] = None

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
    model_repo_id: str = "mistralai/Mistral-7B-Instruct-v0.1"
    max_new_tokens: PositiveInt = 1000
    temperature: confloat(ge=0.0, le=2.0) = 0.7 # type: ignore
    top_p: Optional[confloat(ge=0.0, le=1.0)] = 0.9 # Padrão atualizado # type: ignore
    top_k: Optional[PositiveInt] = 50
    repetition_penalty: Optional[confloat(ge=1.0)] = 1.0 # Padrão atualizado # type: ignore
    prompt_template: str = Field(default="""Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer concise and directly address the question.
Provide the answer based *only* on the provided context. Do not add external information.

Context:
{context}

Question: {query}
Helpful Answer:""")

class AppConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    class Config:
        # Opcional: Se desejar permitir campos extras no TOML
        # que não estão definidos nos modelos (útil durante o desenvolvimento)
        # extra = 'ignore'
        pass 