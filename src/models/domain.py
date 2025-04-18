from pydantic import BaseModel, ConfigDict
from typing import Optional, List

class Domain(BaseModel):
    id: Optional[int]
    name: str
    description: str
    keywords: List[str]
    db_path: str
    vector_store_path: str
    total_documents: int
    faiss_index: Optional[int]
    embeddings_dimension: Optional[int]

    model_config = ConfigDict(arbitrary_types_allowed=True)        