#!/usr/bin/env python3
"""
Step 2: Generate embeddings for papers (documents only).
"""
import sys
import json
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lms.openai_untils import embed_text_batch


def generate_embeddings(entity_csv: str, output_json: str, batch_size: int = 100):
    """Generate embeddings for papers only."""
    print("=" * 60)
    print("Step 2: Generate Embeddings (Papers Only)")
    print("=" * 60)
    
    # Load entities
    print(f"Loading entities from {entity_csv}...")
    df = pd.read_csv(entity_csv)
    
    # Filter to papers only
    papers = df[df['type'] == 'paper'].copy()
    print(f"Found {len(papers):,} papers")
    
    # Check if output exists
    output_path = Path(output_json)
    if output_path.exists():
        print(f"\n⚠️  Output file already exists: {output_path}")
        response = input("Regenerate embeddings? This will cost ~$0.10 (y/N): ")
        if response.lower() != 'y':
            print("Skipping embedding generation")
            return
    
    print(f"\n⚠️  Generating {len(papers):,} embeddings via OpenAI API")
    print(f"   Estimated cost: ~${len(papers) * 0.00002:.2f}")
    print()
    
    # Generate embeddings in batches
    results = []
    total_batches = (len(papers) + batch_size - 1) // batch_size
    
    for i in range(0, len(papers), batch_size):
        batch = papers.iloc[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        print(f"Batch {batch_num}/{total_batches}: Processing {len(batch)} papers...", end=" ")
        
        try:
            texts = batch['name'].tolist()
            embeddings = embed_text_batch(texts)
            
            for (_, paper), embedding in zip(batch.iterrows(), embeddings):
                results.append({
                    'id': paper['id'],
                    'name': paper['name'],
                    'type': paper['type'],
                    'embedding': embedding
                })
            
            print("✓")
        except Exception as e:
            print(f"✗\nError: {e}")
            if results:
                partial_file = output_path.parent / f"embeddings_partial_{len(results)}.json"
                print(f"Saving {len(results)} completed embeddings to {partial_file}")
                with open(partial_file, 'w') as f:
                    json.dump(results, f, indent=2)
            raise
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving embeddings to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Step 2 complete!")
    print(f"  Generated {len(results):,} embeddings")
    print(f"  File size: {file_size_mb:.2f} MB")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate embeddings for papers")
    parser.add_argument('--input', default='data/csv/all_entity_info.csv', help='Entity CSV file')
    parser.add_argument('--output', default='data/processed/paper_embeddings.json', help='Output JSON file')
    parser.add_argument('--batch-size', type=int, default=500, help='Batch size for API calls')
    args = parser.parse_args()
    
    generate_embeddings(args.input, args.output, args.batch_size)


if __name__ == "__main__":
    main()

