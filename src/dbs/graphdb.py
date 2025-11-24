"""
GraphDB: Clean graph database with vector search.
Handles querying only - data loading is separate.
"""
import apsw
import sqlite_vec
import struct
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Literal, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from src.lms.openai_untils import embed_text


def serialize(vector: List[float]) -> bytes:
    """Serialize a list of floats into compact bytes format."""
    return struct.pack("%sf" % len(vector), *vector)


def deserialize(blob: bytes, dim: int = 1536) -> List[float]:
    """Deserialize bytes back to list of floats."""
    return list(struct.unpack("%sf" % dim, blob))


class GraphDB:
    """
    Graph database with vector search capabilities.
    
    Schema:
        nodes: id, name, type (indexed)
        edges: id, source_id (indexed), target_id (indexed), 
               relation_type (indexed), metadata
        vec_embeddings: node_id (FK to nodes), embedding
    
    Usage:
        db = GraphDB()
        results = db.search_by_text("machine learning", k=5)
        neighbors = db.get_neighbors(paper_id, direction='in', 
                                     relation_types=['author_write_paper'])
    """
    
    def __init__(self, db_path: str = "data/sqlite/graphdb.db"):
        """Initialize GraphDB connection."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = None
        self._connect()
    
    def _connect(self):
        """Connect to database and initialize tables."""
        self.db = apsw.Connection(str(self.db_path))
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self._create_tables()
    
    def _create_tables(self):
        """Create tables and indexes if they don't exist."""
        cursor = self.db.cursor()
        
        # Nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes(
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)")
        
        # Edges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges(
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (source_id) REFERENCES nodes(id),
                FOREIGN KEY (target_id) REFERENCES nodes(id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation_type)")
        
        # Vector embeddings table (for documents)
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
                node_id TEXT PRIMARY KEY,
                embedding FLOAT[1536]
            )
        """)
    
    # ========== Vector Search ==========
    
    def search_by_embedding(
        self, 
        embedding: List[float], 
        k: int = 5, 
        node_type: Optional[str] = 'paper'
    ) -> List[Dict]:
        """
        Search nodes by embedding vector.
        
        Args:
            embedding: Query embedding vector
            k: Number of results
            node_type: Filter by node type (default: 'paper')
        
        Returns:
            List of dicts with id, name, type, distance, similarity
        """
        cursor = self.db.cursor()
        
        sql = """
            SELECT
                nodes.id,
                nodes.name,
                nodes.type,
                distance,
                vec_embeddings.embedding
            FROM vec_embeddings
            JOIN nodes ON nodes.id = vec_embeddings.node_id
            WHERE embedding MATCH ?
                AND k = ?
        """
        params = [serialize(embedding), k]
        
        if node_type:
            sql += " AND nodes.type = ?"
            params.append(node_type)
        
        sql += " ORDER BY distance"
        
        results = cursor.execute(sql, params).fetchall()
        
        # Calculate cosine similarity for each result
        query_embedding = np.array(embedding).reshape(1, -1)
        output = []
        for r in results:
            result_embedding = np.array(deserialize(r[4])).reshape(1, -1)
            similarity = cosine_similarity(query_embedding, result_embedding)[0][0]
            output.append({
                'id': r[0], 
                'name': r[1], 
                'type': r[2], 
                'distance': r[3],
                'similarity': float(similarity)
            })
        
        return output
    
    def search_by_text(
        self, 
        query: str, 
        k: int = 5, 
        node_type: Optional[str] = 'paper'
    ) -> List[Dict]:
        """
        Search nodes by text query (generates embedding via API).
        
        Args:
            query: Text query
            k: Number of results
            node_type: Filter by node type
        
        Returns:
            List of dicts with id, name, type, distance
        """
        embedding = embed_text(query)
        return self.search_by_embedding(embedding, k, node_type)
    
    # ========== Graph Queries ==========
    
    def get_node(self, node_id: str) -> Optional[Dict]:
        """
        Get node by ID.
        
        Returns:
            Dict with id, name, type or None
        """
        cursor = self.db.cursor()
        row = cursor.execute(
            "SELECT id, name, type FROM nodes WHERE id = ?",
            (node_id,)
        ).fetchone()
        
        if row:
            return {'id': row[0], 'name': row[1], 'type': row[2]}
        return None
    
    def get_neighbors(
        self,
        node_id: str,
        direction: Literal['in', 'out', 'both'] = 'both',
        relation_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Get neighboring nodes via edges.
        
        Args:
            node_id: Source node ID
            direction: 'in' (incoming), 'out' (outgoing), 'both'
            relation_types: Filter by relation types
        
        Returns:
            List of dicts with edge_id, node_id, node_name, node_type,
            relation_type, direction, metadata
        """
        cursor = self.db.cursor()
        results = []
        
        # Outgoing edges
        if direction in ['out', 'both']:
            sql = """
                SELECT 
                    e.id, e.target_id, n.name, n.type, 
                    e.relation_type, e.metadata, 'out'
                FROM edges e
                JOIN nodes n ON n.id = e.target_id
                WHERE e.source_id = ?
            """
            params = [node_id]
            
            if relation_types:
                placeholders = ','.join('?' * len(relation_types))
                sql += f" AND e.relation_type IN ({placeholders})"
                params.extend(relation_types)
            
            rows = cursor.execute(sql, params).fetchall()
            results.extend(rows)
        
        # Incoming edges
        if direction in ['in', 'both']:
            sql = """
                SELECT 
                    e.id, e.source_id, n.name, n.type, 
                    e.relation_type, e.metadata, 'in'
                FROM edges e
                JOIN nodes n ON n.id = e.source_id
                WHERE e.target_id = ?
            """
            params = [node_id]
            
            if relation_types:
                placeholders = ','.join('?' * len(relation_types))
                sql += f" AND e.relation_type IN ({placeholders})"
                params.extend(relation_types)
            
            rows = cursor.execute(sql, params).fetchall()
            results.extend(rows)
        
        return [
            {
                'edge_id': r[0],
                'node_id': r[1],
                'node_name': r[2],
                'node_type': r[3],
                'relation_type': r[4],
                'metadata': json.loads(r[5]) if r[5] else {},
                'direction': r[6]
            }
            for r in results
        ]
    
    def traverse(
        self,
        embedding: List[float],
        path: List[Tuple[str, str]],
        k: int = 5
    ) -> List[Dict]:
        """
        Multi-hop traversal starting from vector search.
        
        Args:
            embedding: Query embedding
            path: List of (relation_type, direction) tuples
                  e.g., [('author_write_paper', 'in'), ('author_in_affiliation', 'out')]
            k: Number of initial results
        
        Returns:
            List of dicts with full traversal path
        """
        # Start with vector search
        results = self.search_by_embedding(embedding, k=k, node_type='paper')
        
        for paper in results:
            current_nodes = [paper]
            
            # Follow path
            for relation_type, direction in path:
                next_level = []
                for node in current_nodes:
                    neighbors = self.get_neighbors(
                        node['node_id'] if 'node_id' in node else node['id'],
                        direction=direction,
                        relation_types=[relation_type]
                    )
                    next_level.extend(neighbors)
                current_nodes = next_level
                
                # Store at appropriate level
                if not path or (relation_type, direction) == path[0]:
                    if relation_type == 'author_write_paper':
                        paper['authors'] = next_level
                    elif relation_type == 'author_in_affiliation':
                        if 'authors' in paper:
                            for author in paper['authors']:
                                author_neighbors = [n for n in next_level 
                                                   if n.get('direction') == direction]
                                author['affiliations'] = author_neighbors
        
        return results
    
    # ========== Statistics ==========
    
    def stats(self) -> Dict:
        """Get database statistics."""
        cursor = self.db.cursor()
        
        # Node counts by type
        node_counts = dict(cursor.execute(
            "SELECT type, COUNT(*) FROM nodes GROUP BY type"
        ).fetchall())
        
        # Edge counts by relation type
        edge_counts = dict(cursor.execute(
            "SELECT relation_type, COUNT(*) FROM edges GROUP BY relation_type"
        ).fetchall())
        
        # Total counts
        total_nodes = cursor.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        total_edges = cursor.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        total_embeddings = cursor.execute("SELECT COUNT(*) FROM vec_embeddings").fetchone()[0]
        
        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'total_embeddings': total_embeddings,
            'nodes_by_type': node_counts,
            'edges_by_type': edge_counts
        }
    
    def close(self):
        """Close database connection."""
        if self.db:
            self.db.close()

