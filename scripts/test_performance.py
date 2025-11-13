#!/usr/bin/env python3
"""
Performance test for GraphDB.
Tests vector search and graph traversal with clean methodology.
Results saved to CSV for validation.
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


def benchmark_vector_search(db: GraphDB, embeddings: dict, k: int = 5):
    """Benchmark pure vector search."""
    queries = list(embeddings.keys())[:7]
    latencies = []
    
    print("\n=== Vector Search (0-hop) ===")
    for query in queries:
        start = time.perf_counter()
        results = db.search_by_embedding(embeddings[query], k=k)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        print(f"  {query:35s}: {latency:6.2f}ms ({len(results)} results)")
    
    avg = sum(latencies) / len(latencies)
    print(f"  Average: {avg:.2f}ms")
    return latencies


def benchmark_1hop(db: GraphDB, embeddings: dict, k: int = 5):
    """Benchmark 1-hop: papers + authors."""
    queries = list(embeddings.keys())[7:14]
    latencies = []
    
    print("\n=== 1-Hop: Papers → Authors ===")
    for query in queries:
        start = time.perf_counter()
        
        # Vector search
        papers = db.search_by_embedding(embeddings[query], k=k)
        
        # Get authors for each paper
        for paper in papers:
            paper['authors'] = db.get_neighbors(
                paper['id'], 
                direction='in',
                relation_types=['author_write_paper']
            )
        
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        
        total_authors = sum(len(p['authors']) for p in papers)
        print(f"  {query:35s}: {latency:6.2f}ms ({len(papers)} papers, {total_authors} authors)")
    
    avg = sum(latencies) / len(latencies)
    print(f"  Average: {avg:.2f}ms")
    return latencies


def benchmark_2hop(db: GraphDB, embeddings: dict, k: int = 5):
    """Benchmark 2-hop: papers + authors + affiliations."""
    queries = list(embeddings.keys())[14:]
    latencies = []
    all_results = []
    
    print("\n=== 2-Hop: Papers → Authors → Affiliations ===")
    for query in queries:
        start = time.perf_counter()
        
        # Vector search
        papers = db.search_by_embedding(embeddings[query], k=k)
        
        # Get authors for each paper
        for paper in papers:
            paper['authors'] = db.get_neighbors(
                paper['id'],
                direction='in',
                relation_types=['author_write_paper']
            )
            
            # Get affiliations for each author
            for author in paper['authors']:
                author['affiliations'] = db.get_neighbors(
                    author['node_id'],
                    direction='out',
                    relation_types=['author_in_affiliation']
                )
        
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        
        # Store results for CSV export
        for paper in papers:
            for author in paper['authors']:
                for aff in author['affiliations']:
                    all_results.append({
                        'query': query,
                        'paper_title': paper['name'],
                        'paper_id': paper['id'],
                        'author': author['node_name'],
                        'author_id': author['node_id'],
                        'affiliation': aff['node_name'],
                        'affiliation_id': aff['node_id'],
                        'latency_ms': f"{latency:.2f}"
                    })
        
        total_authors = sum(len(p['authors']) for p in papers)
        total_affs = sum(
            len(a['affiliations']) 
            for p in papers 
            for a in p['authors']
        )
        print(f"  {query:35s}: {latency:6.2f}ms ({len(papers)} papers, {total_authors} authors, {total_affs} affs)")
    
    avg = sum(latencies) / len(latencies)
    print(f"  Average: {avg:.2f}ms")
    return latencies, all_results


def main():
    db = GraphDB()
    
    print("=" * 70)
    print("GRAPHDB PERFORMANCE TEST")
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
    
    # Run benchmarks
    lat_0hop = benchmark_vector_search(db, embeddings, k=5)
    lat_1hop = benchmark_1hop(db, embeddings, k=5)
    lat_2hop, results_2hop = benchmark_2hop(db, embeddings, k=5)
    
    # Save results to CSV
    output_file = Path("data/processed/test_results_paper_author_affiliation.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        if results_2hop:
            writer = csv.DictWriter(f, fieldnames=results_2hop[0].keys())
            writer.writeheader()
            writer.writerows(results_2hop)
    
    print(f"\n✓ Results saved to {output_file}")
    print(f"  Total rows: {len(results_2hop)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (SQLite + sqlite-vec)")
    print("=" * 70)
    
    avg_0 = sum(lat_0hop) / len(lat_0hop)
    avg_1 = sum(lat_1hop) / len(lat_1hop)
    avg_2 = sum(lat_2hop) / len(lat_2hop)
    
    print(f"\n0-hop (vector search):              {avg_0:6.2f}ms")
    print(f"1-hop (+ author lookup):            {avg_1:6.2f}ms")
    print(f"2-hop (+ affiliation lookup):       {avg_2:6.2f}ms")
    
    print(f"\nGraph traversal overhead:")
    print(f"  0 → 1 hop: +{avg_1 - avg_0:5.2f}ms")
    print(f"  1 → 2 hop: +{avg_2 - avg_1:5.2f}ms")
    print(f"  Total:     +{avg_2 - avg_0:5.2f}ms")
    
    db.close()


if __name__ == "__main__":
    main()

