from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict
from datetime import datetime

class Chunk(BaseModel):
    """
    Chunk é uma unidade de informação que representa um fragmento de um documento. Seu campo metadata armazena:
        page_list: lista de páginas que compõem o chunk; 
        index_list: lista de índices com a numeração sequencial dos chunks em seu documento;
        keywords: lista de keywords do chunk.
    """
    id: Optional[int] = None
    document_id: int
    content: str
    metadata: Dict[str, Any]  # Armazena page_list, index_list, keywords
    created_at: Optional[datetime] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)