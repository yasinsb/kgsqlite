#!/usr/bin/env python3
"""
Simple GraphDB demo.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dbs.graphdb import GraphDB


def main():
    db = GraphDB()
    
    print("=" * 60)
    print("GraphDB Demo")
    print("=" * 60)
    
    # Stats
    stats = db.stats()
    print(f"\nDatabase Statistics:")
    print(f"  Nodes: {stats['total_nodes']:,}")
    print(f"  Edges: {stats['total_edges']:,}")
    print(f"  Embeddings: {stats['total_embeddings']:,}")
    
    # Vector search
    print("\n" + "=" * 60)
    print("Example 1: Vector Search")
    print("=" * 60)
    results = db.search_by_text("information retrieval", k=3)
    print(f"\nQuery: 'information retrieval'")
    for i, paper in enumerate(results, 1):
        print(f"  {i}. {paper['name']}")
        print(f"     Distance: {paper['distance']:.4f}")
    
    # Graph traversal - 1 hop
    print("\n" + "=" * 60)
    print("Example 2: 1-Hop Traversal (Paper → Authors)")
    print("=" * 60)
    paper_id = results[0]['id']
    paper_name = results[0]['name']
    authors = db.get_neighbors(paper_id, direction='in', relation_types=['author_write_paper'])
    
    print(f"\nPaper: {paper_name}")
    print(f"Authors ({len(authors)}):")
    for author in authors:
        print(f"  - {author['node_name']}")
    
    # Graph traversal - 2 hop
    print("\n" + "=" * 60)
    print("Example 3: 2-Hop Traversal (Paper → Authors → Affiliations)")
    print("=" * 60)
    print(f"\nPaper: {paper_name}")
    for author in authors[:2]:  # Show first 2 authors
        affiliations = db.get_neighbors(
            author['node_id'], 
            direction='out', 
            relation_types=['author_in_affiliation']
        )
        print(f"  Author: {author['node_name']}")
        if affiliations:
            for aff in affiliations:
                print(f"    → {aff['node_name']}")
        else:
            print(f"    → No affiliation")
    
    # Show node details
    print("\n" + "=" * 60)
    print("Example 4: Node Details")
    print("=" * 60)
    node = db.get_node(paper_id)
    print(f"\nNode ID: {node['id']}")
    print(f"Name: {node['name']}")
    print(f"Type: {node['type']}")
    
    db.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

