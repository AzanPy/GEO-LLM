from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
from threading import Lock
from .config import settings


class Embedder:
    """PubMedBERT embedding handler"""

    def __init__(self):
        self.model_name = settings.EMBED_MODEL
        self._model = None
        self._lock = Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    print(f"Loading embedding model: {self.model_name}")
                    self._model = SentenceTransformer(
                        self.model_name,
                        device="cpu"
                    )
                    self._model.eval()
                    print("Embedding model loaded successfully")
        return self._model

    def encode(self, text: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        if isinstance(text, str):
            text = [text]

        embeddings = self.model.encode(
            text,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return embeddings

    def encode_single(self, text: str, normalize: bool = True) -> List[float]:
        return self.encode(text, normalize=normalize)[0].tolist()


embedder = Embedder()