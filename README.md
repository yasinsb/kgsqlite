# GraphDB: SQLite-based Graph + Vector Database

A minimal implementation demonstrating SQLite's capability as a combined graph database and vector store, eliminating the need for separate systems like Neo4j for small-to-medium scale projects.

## What This Does

- **Graph Database**: Stores nodes and edges in SQLite
- **Vector Search**: Integrates sqlite-vec for semantic search
- **Multi-hop Traversal**: Query and traverse graph relationships
- **Single File**: Everything in one .db file, no external services

## Tech Stack

- **SQLite** + **sqlite-vec**: Database and vector search
- **APSW**: Python SQLite wrapper with extension support
- **OpenAI API**: Embedding generation (text-embedding-3-small)

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Get data (see data/README.md)
# Download KG20C files to data/kg20c/

# Run pipeline
export OPENAI_API_KEY=your_key
bash scripts/pipeline/run_pipeline.sh
```

## Usage

```python
from src.dbs import GraphDB

db = GraphDB()

# Vector search
papers = db.search_by_text("machine learning", k=5)

# Graph traversal
authors = db.get_neighbors(paper_id, direction='in')

# Multi-hop
results = db.traverse(embedding, path=[
    ('author_write_paper', 'in'),
    ('author_in_affiliation', 'out')
])

db.close()
```

## Structure

```
src/                    # Core library
  ├── dbs/graphdb.py   # Graph + vector database
  └── lms/             # Embedding utilities

scripts/pipeline/       # Data pipeline
  ├── 01_transform_tsv.py
  ├── 02_generate_embeddings.py
  └── 03_load_database.py

scripts/
  ├── demo.py          # Usage examples
  └── test_performance.py  # Benchmarks
  └── test_inverse_traversal.py # Testing with inverse relations
```

## Performance

See [EXPERIMENT.md](EXPERIMENT.md) for detailed benchmarks on the KG20C dataset (16K nodes, 55K edges).

**Key finding:** Graph traversal overhead is negligible (<2ms per hop) compared to embedding generation for example hence traversal is not the bottleneck in many cases.

## Dataset

Uses **KG20C** from [Hung-Nghiep Tran](https://github.com/tranhungnghiep/KG20C), derived from Microsoft Academic Graph. See `data/README.md` for setup.

## Author

**Yasin Salimibeni** ([@yasinsb](https://github.com/yasinsb))  
Contact: sbyasin [at] gmail.com

## License

MIT
