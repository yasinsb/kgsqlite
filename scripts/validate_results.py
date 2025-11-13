#!/usr/bin/env python3
"""
Validate test results against source CSV data.
Ensures that all relationships in results actually exist in the source data.
"""
import pandas as pd
from pathlib import Path


def load_source_data():
    """Load and combine all source CSV files."""
    print("Loading source data...")
    
    # Load entities
    entities = pd.read_csv("data/csv/all_entity_info.csv")
    print(f"  Entities: {len(entities):,} rows")
    
    # Load and combine edges from train/valid/test
    train = pd.read_csv("data/csv/train.csv")
    valid = pd.read_csv("data/csv/valid.csv")
    test = pd.read_csv("data/csv/test.csv")
    
    # Add source column to track which split
    train['split'] = 'train'
    valid['split'] = 'valid'
    test['split'] = 'test'
    
    edges = pd.concat([train, valid, test], ignore_index=True)
    print(f"  Edges: {len(edges):,} rows (train: {len(train)}, valid: {len(valid)}, test: {len(test)})")
    
    return entities, edges


def validate_paper_author_affiliation():
    """Validate Paper → Author → Affiliation results."""
    print("\n" + "=" * 70)
    print("Validating: Paper → Author → Affiliation")
    print("=" * 70)
    
    results_file = Path("data/processed/test_results_paper_author_affiliation.csv")
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    entities, edges = load_source_data()
    results = pd.read_csv(results_file)
    
    print(f"\nValidating {len(results)} result rows...")
    
    errors = []
    
    # Create lookups for faster validation
    entity_ids = set(entities['id'])
    entity_lookup = dict(zip(entities['id'], entities['name']))
    
    # Create edge lookup: (source, target, relation) -> exists
    edge_set = set(
        zip(edges['entity_1_id'], edges['entity_2_id'], edges['relation_id'])
    )
    
    for idx, row in results.iterrows():
        # Check paper exists
        if row['paper_id'] not in entity_ids:
            errors.append(f"Row {idx}: Paper ID {row['paper_id']} not found in entities")
        elif entity_lookup[row['paper_id']] != row['paper_title']:
            errors.append(f"Row {idx}: Paper name mismatch for {row['paper_id']}")
        
        # Check author exists
        if row['author_id'] not in entity_ids:
            errors.append(f"Row {idx}: Author ID {row['author_id']} not found in entities")
        elif entity_lookup[row['author_id']] != row['author']:
            errors.append(f"Row {idx}: Author name mismatch for {row['author_id']}")
        
        # Check affiliation exists
        if row['affiliation_id'] not in entity_ids:
            errors.append(f"Row {idx}: Affiliation ID {row['affiliation_id']} not found in entities")
        elif entity_lookup[row['affiliation_id']] != row['affiliation']:
            errors.append(f"Row {idx}: Affiliation name mismatch for {row['affiliation_id']}")
        
        # Check paper → author edge (author writes paper, so author -> paper)
        if (row['author_id'], row['paper_id'], 'author_write_paper') not in edge_set:
            errors.append(f"Row {idx}: Edge not found: {row['author']} -> {row['paper_title']}")
        
        # Check author → affiliation edge
        if (row['author_id'], row['affiliation_id'], 'author_in_affiliation') not in edge_set:
            errors.append(f"Row {idx}: Edge not found: {row['author']} -> {row['affiliation']}")
    
    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10
            print(f"   {error}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more")
    else:
        print(f"\n✅ All {len(results)} rows validated successfully!")
        print("   - All entity IDs exist")
        print("   - All entity names match")
        print("   - All relationships exist in source data")


def validate_paper_author_paper():
    """Validate Paper → Author → Paper (co-authored) results."""
    print("\n" + "=" * 70)
    print("Validating: Paper → Author → Co-authored Papers")
    print("=" * 70)
    
    results_file = Path("data/processed/test_results_paper_author_paper.csv")
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    entities, edges = load_source_data()
    results = pd.read_csv(results_file)
    
    print(f"\nValidating {len(results)} result rows...")
    
    errors = []
    
    # Create lookups
    entity_ids = set(entities['id'])
    entity_lookup = dict(zip(entities['id'], entities['name']))
    
    edge_set = set(
        zip(edges['entity_1_id'], edges['entity_2_id'], edges['relation_id'])
    )
    
    for idx, row in results.iterrows():
        # Check seed paper exists
        if row['seed_paper_id'] not in entity_ids:
            errors.append(f"Row {idx}: Seed paper ID {row['seed_paper_id']} not found")
        elif entity_lookup[row['seed_paper_id']] != row['seed_paper_title']:
            errors.append(f"Row {idx}: Seed paper name mismatch for {row['seed_paper_id']}")
        
        # Check author exists
        if row['author_id'] not in entity_ids:
            errors.append(f"Row {idx}: Author ID {row['author_id']} not found")
        elif entity_lookup[row['author_id']] != row['author']:
            errors.append(f"Row {idx}: Author name mismatch for {row['author_id']}")
        
        # Check co-authored paper exists
        if row['coauthored_paper_id'] not in entity_ids:
            errors.append(f"Row {idx}: Co-authored paper ID {row['coauthored_paper_id']} not found")
        elif entity_lookup[row['coauthored_paper_id']] != row['coauthored_paper_title']:
            errors.append(f"Row {idx}: Co-authored paper name mismatch for {row['coauthored_paper_id']}")
        
        # Check author -> seed paper edge
        if (row['author_id'], row['seed_paper_id'], 'author_write_paper') not in edge_set:
            errors.append(f"Row {idx}: Edge not found: {row['author']} -> {row['seed_paper_title']}")
        
        # Check author -> co-authored paper edge
        if (row['author_id'], row['coauthored_paper_id'], 'author_write_paper') not in edge_set:
            errors.append(f"Row {idx}: Edge not found: {row['author']} -> {row['coauthored_paper_title']}")
    
    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors[:10]:
            print(f"   {error}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more")
    else:
        print(f"\n✅ All {len(results)} rows validated successfully!")
        print("   - All entity IDs exist")
        print("   - All entity names match")
        print("   - All relationships exist in source data")


def main():
    print("=" * 70)
    print("VALIDATING TEST RESULTS AGAINST SOURCE DATA")
    print("=" * 70)
    
    validate_paper_author_affiliation()
    validate_paper_author_paper()
    
    print("\n" + "=" * 70)
    print("Validation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

