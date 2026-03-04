from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Dict
from tqdm import tqdm
import time
import random

from .config import settings


class QdrantCloudIndexer:
    def __init__(self):
        # ⬇️ timeout বাড়ালাম + retry friendly
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=120,          # seconds (default কম থাকে)
        )
        self._embedder = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            print(f"Loading embedding model for indexing: {settings.EMBED_MODEL}")
            self._embedder = SentenceTransformer(settings.EMBED_MODEL, device="cpu")
            self._embedder.eval()
        return self._embedder

    def ensure_collections(self):
        collections = [settings.STUDY_COLLECTION, settings.SAMPLE_COLLECTION]

        for collection in collections:
            try:
                _ = self.client.get_collection(collection)
                print(f"Collection {collection} exists")
            except Exception:
                print(f"Creating collection {collection}...")
                self.client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(
                        size=768,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created collection {collection}")

    def _safe_upsert(self, collection_name: str, points: List[PointStruct], max_retries: int = 6):
        """
        Upsert with retry + exponential backoff.
        """
        last_err = None
        for attempt in range(max_retries):
            try:
                self.client.upsert(collection_name=collection_name, points=points, wait=True)
                return
            except Exception as e:
                last_err = e
                # backoff: 1s,2s,4s,8s,... + jitter
                sleep_s = (2 ** attempt) + random.random()
                print(f"[Upsert retry {attempt+1}/{max_retries}] {type(e).__name__}: {e} | sleep {sleep_s:.1f}s")
                time.sleep(sleep_s)

        raise RuntimeError(f"Upsert failed after {max_retries} retries: {last_err}")

    def index_documents(self, collection_name: str, documents: List[Dict], batch_size: int = 32):
        """
        batch ছোট রাখলে write timeout কম হবে।
        """
        points = []
        total = 0

        for doc in tqdm(documents, desc=f"Indexing to {collection_name}"):
            text = doc.get("text", "") or ""
            meta = doc.get("metadata", {}) or {}

            emb = self.embedder.encode(
                [text],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False
            )[0]
            embedding = emb.tolist()

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={"text": text, **meta},
                )
            )

            if len(points) >= batch_size:
                self._safe_upsert(collection_name, points)
                total += len(points)
                points = []

        if points:
            self._safe_upsert(collection_name, points)
            total += len(points)

        print(f"Indexed {total} documents to {collection_name}")