#!/usr/bin/env python3
"""
Generate test embeddings for performance benchmarking.
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lms.openai_untils import embed_text


def main():
    # Diverse test queries
    queries = [
        "information retrieval",
        "machine learning",
        "neural networks",
        "data mining",
        "natural language processing",
        "computer vision",
        "deep learning",
        "text classification",
        "clustering algorithms",
        "recommendation systems",
        "reinforcement learning",
        "semantic search",
        "knowledge graphs",
        "entity recognition",
        "sentiment analysis",
        "question answering",
        "image segmentation",
        "speech recognition",
        "anomaly detection",
        "transfer learning"
    ]
    
    print(f"Generating embeddings for {len(queries)} queries...")
    
    embeddings = {}
    for i, query in enumerate(queries, 1):
        print(f"  {i}/{len(queries)}: {query}")
        embeddings[query] = embed_text(query)
    
    output_file = Path("data/processed/test_embeddings.json")
    with open(output_file, 'w') as f:
        json.dump(embeddings, f, indent=2)
    
    print(f"\n✓ Saved {len(embeddings)} embeddings to {output_file}")


if __name__ == "__main__":
    main()

