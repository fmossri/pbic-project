from typing import Optional
from pydantic import BaseModel, ConfigDict
import datetime
class Chunk(BaseModel):
    id : Optional[int] = None
    document_id : int
    page_number : int
    chunk_page_index : int
    chunk_start_char_position : int
    content : str
    created_at : Optional[datetime.datetime]

    model_config = ConfigDict(arbitrary_types_allowed=True)