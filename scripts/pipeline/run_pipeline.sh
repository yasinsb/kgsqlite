#!/bin/bash
# Run complete KG20C data pipeline

set -e  # Exit on error

echo "========================================"
echo "KG20C Data Pipeline"
echo "========================================"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Step 1: Transform TSV to CSV
echo "Running Step 1: Transform TSV to CSV"
python scripts/pipeline/01_transform_tsv.py
echo ""

# Step 2: Generate embeddings (requires OpenAI API key)
echo "Running Step 2: Generate Embeddings"
python scripts/pipeline/02_generate_embeddings.py
echo ""

# Step 3: Load database
echo "Running Step 3: Load Database"
python scripts/pipeline/03_load_database.py
echo ""

echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  - Run demo: python scripts/demo.py"
echo "  - Run tests: python scripts/test_performance.py"

