import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

url = os.getenv("QDRANT_URL")
key = os.getenv("QDRANT_API_KEY")

print("URL =", url)
print("KEY set =", bool(key))

client = QdrantClient(url=url, api_key=key)
print(client.get_collections())