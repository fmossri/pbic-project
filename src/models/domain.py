from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from datetime import datetime

class Domain(BaseModel):
    id: Optional[int] = Field(None, description="ID do domínio")
    name: str = Field(description="Nome do domínio", updatable=True)
    description: str = Field(description="Descrição do domínio", updatable=True)
    keywords: str = Field(description="Palavras-chave do domínio", updatable=True)
    db_path: str = Field(description="Caminho do banco de dados do domínio", updatable=False)
    vector_store_path: str = Field(description="Caminho do vector store do domínio", updatable=False)
    embeddings_dimension: int = Field(default=0, description="Dimensão dos embeddings do domínio", updatable=False) # Definido em process_directory na primeira ingestão
    total_documents: int = Field(default=0, description="Total de documentos do domínio", updatable=True)
    created_at: Optional[datetime] = Field(None, description="Data de criação do domínio", updatable=False)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def updatable_fields(cls) -> set[str]:
        updatable = set()

        for name, field_info in cls.model_fields.items():
            if field_info.json_schema_extra and field_info.json_schema_extra.get('updatable') is True:
                updatable.add(name)
        return updatable
    