from pydantic import BaseModel
from typing import Optional

class Domain(BaseModel):
    id: Optional[int]
    domain_name: str
    domain_description: str
    domain_keywords: str
    domain_db_path: str
    faiss_index_path: str
    total_documents: int

        