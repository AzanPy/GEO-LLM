#!/usr/bin/env python3
"""
One-time script to upload your local data to Qdrant Cloud
Run this before first deployment
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.indexer import QdrantCloudIndexer


def chunk_text(text: str, chunk_size: int = 500):
    words = (text or "").split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])


def parse_study_json(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gse_id = data["gse_id"]
    documents = []

    # Core fields
    core_fields = {
        "title": data.get("title", ""),
        "study_summary": data.get("study_summary", ""),
        "overall_design": data.get("experimental_design", {}).get("overall_design", ""),
    }

    for field_name, content in core_fields.items():
        if content:
            for chunk in chunk_text(content):
                documents.append(
                    {
                        "text": chunk,
                        "metadata": {
                            "level": "study",
                            "gse_id": gse_id,
                            "field": field_name,
                            "organism": data.get("organism"),
                            "assay_type": data.get("experimental_design", {}).get("experimental_type"),
                            "tissue": data.get("experimental_design", {}).get("tissue"),
                        },
                    }
                )

    # Protocols
    protocols = data.get("experimental_protocols", {})
    for protocol_name, protocol_content in protocols.items():
        if isinstance(protocol_content, list):
            protocol_content = " ".join(protocol_content)
        if protocol_content:
            for chunk in chunk_text(protocol_content):
                documents.append(
                    {
                        "text": chunk,
                        "metadata": {
                            "level": "study",
                            "gse_id": gse_id,
                            "field": f"protocol_{protocol_name}",
                        },
                    }
                )

    return documents


def parse_sample_json(file_path: Path, study_metadata_map):
    """
    FIXED: no leading indentation spaces in sample text block.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gse_id = data["gse_id"]
    study_meta = study_metadata_map.get(gse_id, {})

    documents = []
    for sample in data.get("samples", []):
        gsm_id = sample["gsm_id"]
        bio = sample.get("biological_metadata", {}) or {}

        # ✅ CLEAN text (no leading spaces)
        text = (
            f"Title: {sample.get('title')}\n"
            f"Tissue: {bio.get('tissue')}\n"
            f"Disease: {bio.get('disease')}\n"
            f"Genotype: {bio.get('genotype')}\n"
            f"Cell Line: {bio.get('cell_line')}\n"
            f"Cell Type: {bio.get('cell_type')}"
        )

        documents.append(
            {
                "text": text,
                "metadata": {
                    "level": "sample",
                    "gse_id": gse_id,
                    "gsm_id": gsm_id,
                    "organism": study_meta.get("organism"),
                    "assay_type": study_meta.get("assay_type"),
                    "tissue": bio.get("tissue"),
                    "disease": bio.get("disease"),
                    "genotype": bio.get("genotype"),
                    "cell_line": bio.get("cell_line"),
                    "cell_type": bio.get("cell_type"),
                },
            }
        )

    return documents


def main():
    print("🔧 Qdrant Cloud Setup")
    print(f"   URL: {settings.QDRANT_URL}")

    indexer = QdrantCloudIndexer()
    indexer.ensure_collections()

    # Paths
    study_dir = Path("data/studies")
    sample_dir = Path("data/samples")

    if not study_dir.exists():
        print(f"❌ Study directory not found: {study_dir}")
        return

    # Build study metadata map
    print("\n📚 Building study metadata map...")
    study_metadata_map = {}
    for file in tqdm(list(study_dir.glob("*.json"))):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        study_metadata_map[data["gse_id"]] = {
            "organism": data.get("organism"),
            "assay_type": data.get("experimental_design", {}).get("experimental_type"),
            "tissue": data.get("experimental_design", {}).get("tissue"),
        }

    # Index studies
    print("\n📤 Indexing studies to Qdrant Cloud...")
    all_study_docs = []
    for file in tqdm(list(study_dir.glob("*.json"))):
        all_study_docs.extend(parse_study_json(file))

    if all_study_docs:
        indexer.index_documents(settings.STUDY_COLLECTION, all_study_docs)

    # Index samples
    all_sample_docs = []
    if sample_dir.exists():
        print("\n📤 Indexing samples to Qdrant Cloud...")
        for file in tqdm(list(sample_dir.glob("*.json"))):
            all_sample_docs.extend(parse_sample_json(file, study_metadata_map))

        if all_sample_docs:
            indexer.index_documents(settings.SAMPLE_COLLECTION, all_sample_docs)

    print("\n✅ Setup complete!")
    print(f"   Studies indexed: {len(all_study_docs)} chunks")
    if sample_dir.exists():
        print(f"   Samples indexed: {len(all_sample_docs)} chunks")


if __name__ == "__main__":
    main()