"""Build FAISS vector index from structured TCM knowledge base.

Usage:
    python -m src.knowledge_base.embedding_builder --input knowledge-base/structured/ --output knowledge-base/embeddings/
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu is required: pip install faiss-cpu")

try:
    import anthropic
except ImportError:
    raise ImportError("anthropic is required: pip install anthropic")


def get_embeddings(texts: list[str], model: str = "voyage-3") -> np.ndarray:
    """Get embeddings from Anthropic's embedding API or fallback.

    Note: As of 2025, Anthropic doesn't have a public embedding API.
    This uses a simple TF-IDF fallback or can be swapped for voyage/openai.
    """
    try:
        # Try voyage AI (recommended for Claude ecosystem)
        import voyageai
        vo = voyageai.Client()
        result = vo.embed(texts, model="voyage-3", input_type="document")
        return np.array(result.embeddings, dtype="float32")
    except ImportError:
        pass

    # Fallback: simple character n-gram hash embedding (for development/testing)
    # Replace with real embedding API in production
    print("Warning: Using fallback hash embeddings. Install voyageai for production quality.")
    return _hash_embeddings(texts)


def _hash_embeddings(texts: list[str], dim: int = 256) -> np.ndarray:
    """Simple hash-based embeddings for development/testing only."""
    embeddings = []
    for text in texts:
        # Character trigram hashing
        vec = np.zeros(dim, dtype="float32")
        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            idx = hash(trigram) % dim
            vec[idx] += 1.0
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        embeddings.append(vec)
    return np.array(embeddings, dtype="float32")


def build_documents(structured_dir: Path) -> list[dict]:
    """Load structured pulse data into searchable documents."""
    documents = []

    # Load pulse type files
    pulse_types_dir = structured_dir / "pulse_types"
    if pulse_types_dir.exists():
        for f in sorted(pulse_types_dir.glob("*.json")):
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)

            # Create searchable text representation
            parts = [data.get("chinese_name", "")]
            for desc in data.get("descriptions", []):
                if desc:
                    parts.append(str(desc))
            for feeling in data.get("finger_feelings", []):
                if feeling:
                    parts.append(str(feeling))
            for sig in data.get("clinical_significance", []):
                if sig:
                    parts.append(str(sig))

            documents.append({
                "id": f.stem,
                "type": "pulse_type",
                "text": " ".join(parts),
                "metadata": data,
            })

    # Load classical texts if available
    classics_dir = structured_dir.parent / "classical_texts"
    if classics_dir.exists():
        for f in sorted(classics_dir.glob("*.json")):
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, list):
                for item in data:
                    documents.append({
                        "id": f"{f.stem}_{item.get('name', '')}",
                        "type": "classical_text",
                        "text": json.dumps(item, ensure_ascii=False),
                        "metadata": item,
                    })
            elif isinstance(data, dict):
                documents.append({
                    "id": f.stem,
                    "type": "classical_text",
                    "text": json.dumps(data, ensure_ascii=False),
                    "metadata": data,
                })

    return documents


def build_index(documents: list[dict], output_dir: Path):
    """Build FAISS index from documents."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not documents:
        print("No documents to index")
        return

    texts = [doc["text"] for doc in documents]
    print(f"Generating embeddings for {len(texts)} documents...")
    embeddings = get_embeddings(texts)

    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")

    # Build index (Inner Product for cosine similarity on normalized vectors)
    index = faiss.IndexFlatIP(dim)
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Save index
    index_path = output_dir / "pulse_kb.faiss"
    faiss.write_index(index, str(index_path))
    print(f"FAISS index saved: {index_path} ({index.ntotal} vectors)")

    # Save metadata
    metadata = [{"id": doc["id"], "type": doc["type"], "text": doc["text"][:200]}
                for doc in documents]
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved: {metadata_path}")


def search_index(
    query: str,
    index_path: str | Path,
    metadata_path: str | Path,
    top_k: int = 5,
) -> list[dict]:
    """Search the FAISS index for similar documents."""
    index = faiss.read_index(str(index_path))
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    query_embedding = get_embeddings([query])
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        result = metadata[idx].copy()
        result["score"] = float(score)
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Build FAISS vector index for TCM knowledge")
    parser.add_argument("--input", default="knowledge-base/structured", help="Structured data directory")
    parser.add_argument("--output", default="knowledge-base/embeddings", help="Output directory for index")
    parser.add_argument("--search", help="Test search query")
    args = parser.parse_args()

    if args.search:
        # Search mode
        index_path = Path(args.output) / "pulse_kb.faiss"
        metadata_path = Path(args.output) / "metadata.json"
        if not index_path.exists():
            print(f"Index not found: {index_path}. Build first without --search.")
            return
        results = search_index(args.search, index_path, metadata_path)
        print(f"Results for: {args.search}")
        for r in results:
            print(f"  [{r['score']:.3f}] {r['id']}: {r['text'][:80]}...")
    else:
        # Build mode
        input_dir = Path(args.input)
        documents = build_documents(input_dir)
        print(f"Loaded {len(documents)} documents")
        build_index(documents, Path(args.output))


if __name__ == "__main__":
    main()
