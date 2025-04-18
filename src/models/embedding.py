from pydantic import BaseModel, ConfigDict
import numpy as np
from typing import Optional

class Embedding(BaseModel):
    id : Optional[int]
    chunk_id : int
    faiss_index : Optional[int]
    embedding : np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)
