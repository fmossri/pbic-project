from typing import Optional
from pydantic import BaseModel

class Chunk(BaseModel):
    id : Optional[int] = None
    document_id : int
    page_number : int
    chunk_page_index : int
    chunk_start_char_position : int
    faiss_index: Optional[int] = None
    content : str