#!/usr/bin/env python3
"""
Step 1: Transform KG20C TSV files to CSV format.
"""
import pandas as pd
from pathlib import Path


def transform_entity_info(input_dir: Path, output_dir: Path):
    """Transform all_entity_info.txt to CSV."""
    input_file = input_dir / "all_entity_info.txt"
    output_file = output_dir / "all_entity_info.csv"
    
    df = pd.read_csv(input_file, sep='\t')
    df.to_csv(output_file, index=False)
    
    print(f"✓ {input_file.name} -> {output_file.name} ({len(df)} rows)")


def transform_relation_info(input_dir: Path, output_dir: Path):
    """Transform all_relation_info.txt to CSV."""
    input_file = input_dir / "all_relation_info.txt"
    output_file = output_dir / "all_relation_info.csv"
    
    df = pd.read_csv(input_file, sep='\t')
    df.to_csv(output_file, index=False)
    
    print(f"✓ {input_file.name} -> {output_file.name} ({len(df)} rows)")


def transform_triples(filename: str, input_dir: Path, output_dir: Path):
    """Transform triple files (train/valid/test) to CSV."""
    input_file = input_dir / filename
    output_file = output_dir / filename.replace('.txt', '.csv')
    
    df = pd.read_csv(input_file, sep='\t', header=None,
                     names=['entity_1_id', 'relation_id', 'entity_2_id'])
    df.to_csv(output_file, index=False)
    
    print(f"✓ {input_file.name} -> {output_file.name} ({len(df)} rows)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Transform KG20C TSV to CSV")
    parser.add_argument('--input', default='data/kg20c', help='Input directory with TSV files')
    parser.add_argument('--output', default='data/csv', help='Output directory for CSV files')
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Step 1: Transform TSV to CSV")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()
    
    transform_entity_info(input_dir, output_dir)
    transform_relation_info(input_dir, output_dir)
    transform_triples('train.txt', input_dir, output_dir)
    transform_triples('valid.txt', input_dir, output_dir)
    transform_triples('test.txt', input_dir, output_dir)
    
    print()
    print("✓ Step 1 complete!")


if __name__ == "__main__":
    main()

