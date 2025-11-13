#!/usr/bin/env python3
"""
Test inverse traversal: Paper → Author → Paper (co-authored papers).
Demonstrates bidirectional graph traversal.
"""
import sys
import json
import time
import csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dbs.graphdb import GraphDB


def load_test_embeddings():
    """Load pre-generated test embeddings."""
    emb_file = Path("data/processed/test_embeddings.json")
    if not emb_file.exists():
        print(f"Error: {emb_file} not found")
        print("Run: python scripts/generate_test_embeddings.py")
        sys.exit(1)
    
    with open(emb_file, 'r') as f:
        return json.load(f)


def benchmark_paper_author_paper(db: GraphDB, embeddings: dict, k: int = 5):
    """
    Benchmark: Paper → Author → Other Papers
    Find authors of papers, then find other papers by those authors.
    """
    queries = list(embeddings.keys())[:10]
    latencies = []
    all_results = []
    
    print("\n=== 2-Hop: Paper → Author → Co-authored Papers ===")
    for query in queries:
        start = time.perf_counter()
        
        # Vector search for papers
        papers = db.search_by_embedding(embeddings[query], k=k)
        
        # Get authors for each paper
        for paper in papers:
            paper['authors'] = db.get_neighbors(
                paper['id'],
                direction='in',
                relation_types=['author_write_paper']
            )
            
            # Get other papers by each author
            for author in paper['authors']:
                author['other_papers'] = db.get_neighbors(
                    author['node_id'],
                    direction='out',
                    relation_types=['author_write_paper']
                )
        
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        
        # Store results for CSV export
        for paper in papers:
            for author in paper['authors']:
                for other_paper in author['other_papers']:
                    # Skip if it's the same paper
                    if other_paper['node_id'] != paper['id']:
                        all_results.append({
                            'query': query,
                            'seed_paper_title': paper['name'],
                            'seed_paper_id': paper['id'],
                            'author': author['node_name'],
                            'author_id': author['node_id'],
                            'coauthored_paper_title': other_paper['node_name'],
                            'coauthored_paper_id': other_paper['node_id'],
                            'latency_ms': f"{latency:.2f}"
                        })
        
        total_authors = sum(len(p['authors']) for p in papers)
        total_papers = sum(
            len(a['other_papers']) 
            for p in papers 
            for a in p['authors']
        )
        print(f"  {query:35s}: {latency:6.2f}ms ({len(papers)} seeds, {total_authors} authors, {total_papers} papers)")
    
    avg = sum(latencies) / len(latencies)
    print(f"  Average: {avg:.2f}ms")
    return latencies, all_results


def main():
    db = GraphDB()
    
    print("=" * 70)
    print("INVERSE TRAVERSAL TEST: Paper → Author → Paper")
    print("=" * 70)
    
    stats = db.stats()
    print(f"Database: {db.db_path}")
    print(f"Nodes: {stats['total_nodes']:,}")
    print(f"Edges: {stats['total_edges']:,}")
    print(f"Embeddings: {stats['total_embeddings']:,}")
    
    embeddings = load_test_embeddings()
    print(f"\nLoaded {len(embeddings)} test embeddings")
    
    # Warm up cache
    print("\nWarming up cache...")
    first_emb = list(embeddings.values())[0]
    db.search_by_embedding(first_emb, k=5)
    print("✓ Ready")
    
    # Run benchmark
    latencies, results = benchmark_paper_author_paper(db, embeddings, k=5)
    
    # Save results to CSV
    output_file = Path("data/processed/test_results_paper_author_paper.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\n✓ Results saved to {output_file}")
    print(f"  Total rows: {len(results)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    avg = sum(latencies) / len(latencies)
    print(f"\n2-hop inverse traversal: {avg:6.2f}ms")
    print("\nThis demonstrates:")
    print("  - Finding papers by semantic search")
    print("  - Traversing to authors (Paper → Author)")
    print("  - Inverse traversal (Author → Other Papers)")
    print("  - Discovering co-authored/related papers")
    
    db.close()


if __name__ == "__main__":
    main()

