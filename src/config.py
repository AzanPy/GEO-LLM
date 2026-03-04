import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Qdrant Cloud Configuration
    QDRANT_URL: str = os.getenv("QDRANT_URL", "https://your-cluster.cloud.qdrant.io")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")

    # Collection Names
    STUDY_COLLECTION: str = os.getenv("STUDY_COLLECTION", "geo_study_collection")
    SAMPLE_COLLECTION: str = os.getenv("SAMPLE_COLLECTION", "geo_sample_collection")

    # Embedding Model
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "neuml/pubmedbert-base-embeddings")

    # Groq LLM
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # AWS/EB Settings
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")

    # Search Config
    TOP_K_STUDY: int = int(os.getenv("TOP_K_STUDY", "10"))
    TOP_K_SAMPLE: int = int(os.getenv("TOP_K_SAMPLE", "10"))
    SIM_THRESHOLD: float = float(os.getenv("SIM_THRESHOLD", "0.6"))

    # Hybrid Search Weights
    ALPHA_VECTOR: float = float(os.getenv("ALPHA_VECTOR", "0.65"))
    BETA_BM25: float = float(os.getenv("BETA_BM25", "0.35"))


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()