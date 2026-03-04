from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Dict
from tqdm import tqdm

from .config import settings


class QdrantCloudIndexer:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
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
                info = self.client.get_collection(collection)
                # (Optional) validate vector config
                # If you want strict validation, uncomment and adapt based on your qdrant client version
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

    def index_documents(self, collection_name: str, documents: List[Dict], batch_size: int = 100):
        points = []
        total = 0

        for doc in tqdm(documents, desc=f"Indexing to {collection_name}"):
            text = doc.get("text", "") or ""
            meta = doc.get("metadata", {}) or {}

            # Safe encode -> always 1 vector
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
                self.client.upsert(collection_name=collection_name, points=points, wait=True)
                total += len(points)
                points = []

        if points:
            self.client.upsert(collection_name=collection_name, points=points, wait=True)
            total += len(points)

        print(f"Indexed {total} documents to {collection_name}")