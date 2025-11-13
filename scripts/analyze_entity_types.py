#!/usr/bin/env python3
"""
Analyze entity types in the KG20C dataset.
Loads all_entity_info.csv and provides statistics on the 'type' column.
"""
import pandas as pd
from pathlib import Path


def main():
    # Define paths
    project_root = Path(__file__).parent.parent
    csv_file = project_root / "data" / "csv" / "all_entity_info.csv"
    
    # Check if file exists
    if not csv_file.exists():
        print(f"Error: File not found at {csv_file}")
        print("Please run the transformation first: python cli.py")
        return
    
    # Load the CSV
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"\nTotal entities: {len(df):,}")
    print(f"\nColumns: {list(df.columns)}")
    
    # Get statistics on the 'type' column
    print("\n" + "="*50)
    print("Entity Type Statistics")
    print("="*50)
    
    type_counts = df['type'].value_counts()
    
    print("\nCount by type:")
    for entity_type, count in type_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {entity_type:15s}: {count:6,} ({percentage:5.1f}%)")
    
    print(f"\n{'Total unique types':15s}: {df['type'].nunique()}")
    
    # Additional statistics
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(df['type'].describe())


if __name__ == "__main__":
    main()

