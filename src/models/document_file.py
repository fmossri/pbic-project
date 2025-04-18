from typing import Optional, List
from pydantic import BaseModel, ConfigDict
from langchain.schema import Document

class DocumentFile(BaseModel):
    id : Optional[int]
    hash : Optional[str]
    name : str
    path : str
    total_pages : int
    pages : List[Document] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

