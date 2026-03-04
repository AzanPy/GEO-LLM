import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PayloadSchemaType

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

STUDY_COLLECTION = os.getenv("STUDY_COLLECTION", "geo_study_collection")
SAMPLE_COLLECTION = os.getenv("SAMPLE_COLLECTION", "geo_sample_collection")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def ensure_index(collection: str, field: str, schema: PayloadSchemaType):
    print(f"→ Ensuring index: {collection}.{field} ({schema})")
    client.create_payload_index(
        collection_name=collection,
        field_name=field,
        field_schema=schema,
        wait=True,
    )

def main():
    # study collection indexes
    ensure_index(STUDY_COLLECTION, "gse_id", PayloadSchemaType.KEYWORD)
    ensure_index(STUDY_COLLECTION, "field", PayloadSchemaType.KEYWORD)
    ensure_index(STUDY_COLLECTION, "organism", PayloadSchemaType.KEYWORD)
    ensure_index(STUDY_COLLECTION, "assay_type", PayloadSchemaType.KEYWORD)
    ensure_index(STUDY_COLLECTION, "tissue", PayloadSchemaType.KEYWORD)
    ensure_index(STUDY_COLLECTION, "level", PayloadSchemaType.KEYWORD)

    # sample collection indexes
    ensure_index(SAMPLE_COLLECTION, "gse_id", PayloadSchemaType.KEYWORD)
    ensure_index(SAMPLE_COLLECTION, "gsm_id", PayloadSchemaType.KEYWORD)
    ensure_index(SAMPLE_COLLECTION, "organism", PayloadSchemaType.KEYWORD)
    ensure_index(SAMPLE_COLLECTION, "assay_type", PayloadSchemaType.KEYWORD)
    ensure_index(SAMPLE_COLLECTION, "tissue", PayloadSchemaType.KEYWORD)
    ensure_index(SAMPLE_COLLECTION, "disease", PayloadSchemaType.KEYWORD)
    ensure_index(SAMPLE_COLLECTION, "genotype", PayloadSchemaType.KEYWORD)
    ensure_index(SAMPLE_COLLECTION, "cell_line", PayloadSchemaType.KEYWORD)
    ensure_index(SAMPLE_COLLECTION, "cell_type", PayloadSchemaType.KEYWORD)
    ensure_index(SAMPLE_COLLECTION, "level", PayloadSchemaType.KEYWORD)

    print("✅ Done. Payload indexes created.")

if __name__ == "__main__":
    main()