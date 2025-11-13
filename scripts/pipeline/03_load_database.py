#!/usr/bin/env python3
"""
Step 3: Load all data into GraphDB (nodes, edges, embeddings).
"""
import sys
import json
import uuid
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dbs.graphdb import GraphDB, serialize


def load_nodes(db: GraphDB, entity_csv: str):
    """Load all nodes from entity CSV."""
    print(f"Loading nodes from {entity_csv}...")
    df = pd.read_csv(entity_csv)
    
    cursor = db.db.cursor()
    for _, row in df.iterrows():
        cursor.execute(
            "INSERT OR REPLACE INTO nodes(id, name, type) VALUES(?, ?, ?)",
            (row['id'], row['name'], row['type'])
        )
    
    print(f"✓ Loaded {len(df):,} nodes")
    counts = df['type'].value_counts()
    for node_type, count in counts.items():
        print(f"    {node_type}: {count:,}")


def load_edges(db: GraphDB, train_csv: str, valid_csv: str, test_csv: str):
    """Load all edges from train/valid/test CSVs."""
    print(f"\nLoading edges...")
    
    cursor = db.db.cursor()
    total = 0
    
    for csv_path, split in [(train_csv, 'train'), (valid_csv, 'validation'), (test_csv, 'test')]:
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            metadata = json.dumps({'tvt_type': split})
            cursor.execute(
                "INSERT OR REPLACE INTO edges(id, source_id, target_id, relation_type, metadata) VALUES(?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), row['entity_1_id'], row['entity_2_id'], row['relation_id'], metadata)
            )
        
        total += len(df)
        print(f"    {split}: {len(df):,}")
    
    print(f"✓ Loaded {total:,} edges")


def load_embeddings(db: GraphDB, embeddings_json: str):
    """Load embeddings from JSON."""
    print(f"\nLoading embeddings from {embeddings_json}...")
    
    with open(embeddings_json, 'r') as f:
        items = json.load(f)
    
    cursor = db.db.cursor()
    for item in items:
        cursor.execute(
            "INSERT OR REPLACE INTO vec_embeddings(node_id, embedding) VALUES(?, ?)",
            (item['id'], serialize(item['embedding']))
        )
    
    print(f"✓ Loaded {len(items):,} embeddings")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Load data into GraphDB")
    parser.add_argument('--db', default='data/sqlite/graphdb.db', help='Database path')
    parser.add_argument('--entities', default='data/csv/all_entity_info.csv', help='Entity CSV')
    parser.add_argument('--train', default='data/csv/train.csv', help='Train CSV')
    parser.add_argument('--valid', default='data/csv/valid.csv', help='Validation CSV')
    parser.add_argument('--test', default='data/csv/test.csv', help='Test CSV')
    parser.add_argument('--embeddings', default='data/processed/paper_embeddings.json', help='Embeddings JSON')
    parser.add_argument('--force', action='store_true', help='Force reload')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Step 3: Load Database")
    print("=" * 60)
    
    db = GraphDB(args.db)
    
    # Check if already loaded
    stats = db.stats()
    if stats['total_nodes'] > 0 and not args.force:
        print(f"Database already loaded ({stats['total_nodes']:,} nodes)")
        print("Use --force to reload")
        return
    
    print()
    load_nodes(db, args.entities)
    load_edges(db, args.train, args.valid, args.test)
    load_embeddings(db, args.embeddings)
    
    print("\n" + "=" * 60)
    print("✓ Step 3 complete!")
    print("=" * 60)
    
    stats = db.stats()
    print(f"\nDatabase statistics:")
    print(f"  Nodes: {stats['total_nodes']:,}")
    print(f"  Edges: {stats['total_edges']:,}")
    print(f"  Embeddings: {stats['total_embeddings']:,}")
    
    db.close()


if __name__ == "__main__":
    main()

