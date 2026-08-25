"""PostgreSQL storage backend using pgvector.

Production-ready vector storage with full SQL capabilities.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import TYPE_CHECKING, Any

from ai_infra.retriever.backends.base import BaseBackend, validate_sql_identifier

if TYPE_CHECKING:
    import psycopg2

_IVFFLAT_LISTS = 100


class PostgresBackend(BaseBackend):
    """PostgreSQL vector storage using pgvector extension.

    Requires the pgvector extension to be installed in your PostgreSQL database.
    Uses psycopg2 for synchronous operations.

    Example:
        >>> backend = PostgresBackend(
        ...     connection_string="postgresql://user:pass@localhost:5432/mydb"
        ... )
        >>> backend.add([[0.1, 0.2, 0.3]], ["Hello"], [{"source": "test"}])
        >>> results = backend.search([0.1, 0.2, 0.3], k=5)

    Note:
        To enable pgvector in your database:
        ```sql
        CREATE EXTENSION IF NOT EXISTS vector;
        ```
    """

    def __init__(
        self,
        connection_string: str | None = None,
        host: str | None = None,
        port: int = 5432,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        table_name: str = "ai_infra_embeddings",
        embedding_dimension: int = 1536,
        similarity: str = "cosine",
    ) -> None:
        """Initialize the PostgreSQL backend.

        Args:
            connection_string: PostgreSQL connection string.
                If not provided, uses individual connection params.
            host: Database host.
            port: Database port.
            database: Database name.
            user: Database user.
            password: Database password.
            table_name: Table name for storing embeddings.
            embedding_dimension: Dimension of embedding vectors.
            similarity: Similarity metric (only "cosine" is supported for pgvector).
        """
        try:
            import psycopg2 as pg
            from pgvector.psycopg2 import register_vector
        except ImportError as e:
            raise ImportError(
                "PostgreSQL backend requires psycopg2 and pgvector. "
                "Install with: pip install psycopg2-binary pgvector"
            ) from e

        # Build connection string if not provided
        if connection_string is None:
            # Use defaults if not provided
            resolved_host = host or os.getenv("PGHOST") or "localhost"
            resolved_database = database or os.getenv("PGDATABASE") or "postgres"
            resolved_user = user or os.getenv("PGUSER") or "postgres"
            resolved_password = password or os.getenv("PGPASSWORD") or ""
            connection_string = self._build_connection_string(
                host=resolved_host,
                port=port,
                database=resolved_database,
                user=resolved_user,
                password=resolved_password,
            )

        # Validate table name to prevent SQL injection
        self._table_name = validate_sql_identifier(table_name, "table_name")
        self._embedding_dimension = embedding_dimension
        self._similarity = similarity  # Store but currently only cosine is used

        # Validate similarity (pgvector only supports cosine for now)
        if similarity not in ("cosine",):
            import logging

            logging.getLogger(__name__).warning(
                f"PostgreSQL backend only supports cosine similarity, ignoring {similarity!r}"
            )

        # Connect to database
        self._conn: psycopg2.connection = pg.connect(connection_string)
        register_vector(self._conn)

        # Create table if it doesn't exist
        self._create_table()

    def _build_connection_string(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
    ) -> str:
        """Build a PostgreSQL connection string."""
        if password:
            return f"postgresql://{user}:{password}@{host}:{port}/{database}"
        return f"postgresql://{user}@{host}:{port}/{database}"

    def _create_table(self) -> None:
        """Create the embeddings table if it doesn't exist."""
        with self._conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create table
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector({self._embedding_dimension})
                )
            """
            )

            # Create index for similarity search
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self._table_name}_embedding_idx
                ON {self._table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {_IVFFLAT_LISTS})
            """
            )

            self._conn.commit()

    def add(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add vectors to the store.

        Args:
            embeddings: List of embedding vectors.
            texts: List of text content.
            metadatas: Optional list of metadata dicts.
            ids: Optional list of IDs. Generated if not provided.

        Returns:
            List of IDs for the added documents.
        """
        if not embeddings:
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in embeddings]

        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in embeddings]

        result_ids = []
        with self._conn.cursor() as cur:
            for emb, text, meta, doc_id in zip(embeddings, texts, metadatas, ids):
                cur.execute(
                    f"""
                    INSERT INTO {self._table_name} (id, text, metadata, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        doc_id,
                        text,
                        json.dumps(meta),
                        emb,
                    ),
                )
                result_ids.append(doc_id)

            self._conn.commit()
        return result_ids

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors using cosine similarity.

        Args:
            query_embedding: The query vector.
            k: Maximum number of results.
            filter: Optional metadata filters.

        Returns:
            List of result dicts with id, text, score, metadata.
        """
        # Build query
        query = f"""
            SELECT id, text, metadata,
                   1 - (embedding <=> %s::vector) as score
            FROM {self._table_name}
        """

        params: list[Any] = [query_embedding]
        where_clauses = []

        # Add metadata filter
        if filter:
            for key, value in filter.items():
                where_clauses.append("metadata->>%s = %s")
                params.extend([key, str(value)])

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([query_embedding, k])

        results: list[dict[str, Any]] = []
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT set_config('ivfflat.probes', %s, true)",
                (str(_IVFFLAT_LISTS),),
            )
            cur.execute(query, params)
            rows = cur.fetchall()

            for row in rows:
                doc_id, text, metadata_json, score = row
                # metadata_json is already a dict (psycopg auto-parses JSON columns)
                # Only parse if it's actually a string
                if isinstance(metadata_json, str):
                    metadata = json.loads(metadata_json)
                elif metadata_json is not None:
                    metadata = metadata_json
                else:
                    metadata = {}
                results.append(
                    {
                        "id": doc_id,
                        "text": text,
                        "score": float(score),
                        "metadata": metadata,
                    }
                )

        return results

    def delete(self, ids: list[str]) -> int:
        """Delete documents by ID.

        Args:
            ids: List of document IDs to delete.

        Returns:
            Number of documents deleted.
        """
        if not ids:
            return 0

        with self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self._table_name} WHERE id = ANY(%s)",
                (ids,),
            )
            deleted = int(cur.rowcount or 0)
            self._conn.commit()

        return deleted

    def clear(self) -> None:
        """Delete all documents."""
        with self._conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {self._table_name}")
            self._conn.commit()

    def count(self) -> int:
        """Get the number of stored documents.

        Returns:
            Total number of documents.
        """
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table_name}")
            result = cur.fetchone()
            return int(result[0]) if result else 0

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
