from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from datetime import datetime
from .domain_config import DomainConfig
class Domain(BaseModel):
    id: Optional[int] = Field(None, description="ID do domínio")
    name: str = Field(description="Nome do domínio", json_schema_extra={"updatable": True})
    description: str = Field(description="Descrição do domínio", json_schema_extra={"updatable": True})
    keywords: str = Field(description="Palavras-chave do domínio", json_schema_extra={"updatable": True})
    total_documents: int = Field(default=0, description="Total de documentos do domínio", json_schema_extra={"updatable": True})
    vector_store_path: str = Field(description="Caminho do vector store do domínio", json_schema_extra={"updatable": False})
    db_path: str = Field(description="Caminho do banco de dados do domínio", json_schema_extra={"updatable": False})
    embeddings_dimension: int = Field(default=0, description="Dimensão dos embeddings do domínio", json_schema_extra={"updatable": False})
    created_at: Optional[datetime] = Field(None, description="Data de criação do domínio", json_schema_extra={"updatable": False})
    updated_at: Optional[datetime] = Field(None, description="Data de atualização do domínio", json_schema_extra={"updatable": False})
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def updatable_fields(cls) -> set[str]:
        updatable = set()

        for name, field_info in cls.model_fields.items():
            if field_info.json_schema_extra and field_info.json_schema_extra.get('updatable') is True:
                updatable.add(name)
        return updatable
    