"""Unit tests for PostgreSQL vector storage backend.

Tests cover:
- PostgresBackend initialization (connection string parsing, pgvector extension)
- Document operations (add, search, delete, update, clear, count)
- Search functionality (cosine similarity, metadata filtering)
- Connection handling (errors, close)
- Edge cases and error handling

Phase 4.1 of ai-infra test plan.
"""

from __future__ import annotations

import json
import sys
import uuid
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Mock Setup - Create mocks at module level before any PostgresBackend import
# =============================================================================

# Create mock psycopg2 module
_mock_psycopg2 = MagicMock()
_mock_connection = MagicMock()
_mock_cursor = MagicMock()

# Configure connection
_mock_psycopg2.connect.return_value = _mock_connection

# Configure cursor context manager
_mock_cursor_cm = MagicMock()
_mock_cursor_cm.__enter__ = MagicMock(return_value=_mock_cursor)
_mock_cursor_cm.__exit__ = MagicMock(return_value=None)
_mock_connection.cursor.return_value = _mock_cursor_cm

# Create mock pgvector module
_mock_pgvector = MagicMock()
_mock_pgvector_psycopg2 = MagicMock()
_mock_pgvector_psycopg2.register_vector = MagicMock()

# Register mocks in sys.modules BEFORE importing PostgresBackend
sys.modules["psycopg2"] = _mock_psycopg2
sys.modules["pgvector"] = _mock_pgvector
sys.modules["pgvector.psycopg2"] = _mock_pgvector_psycopg2

# Now import the module
from ai_infra.retriever.backends.postgres import PostgresBackend  # noqa: E402

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks before each test."""
    _mock_psycopg2.reset_mock()
    _mock_connection.reset_mock()
    _mock_cursor.reset_mock()
    _mock_pgvector_psycopg2.reset_mock()

    # Re-configure mocks after reset
    _mock_psycopg2.connect.return_value = _mock_connection
    _mock_cursor_cm.__enter__.return_value = _mock_cursor
    _mock_connection.cursor.return_value = _mock_cursor_cm

    yield


@pytest.fixture
def backend():
    """Create a PostgresBackend instance with mocked dependencies."""
    return PostgresBackend(connection_string="postgresql://test:test@localhost:5432/testdb")


# =============================================================================
# PostgresBackend Initialization Tests
# =============================================================================


class TestPostgresBackendInit:
    """Tests for PostgresBackend initialization."""

    def test_init_with_connection_string(self):
        """Test initialization with connection string."""
        _backend = PostgresBackend(connection_string="postgresql://user:pass@localhost:5432/testdb")

        _mock_psycopg2.connect.assert_called_with("postgresql://user:pass@localhost:5432/testdb")
        _mock_pgvector_psycopg2.register_vector.assert_called_with(_mock_connection)

    def test_init_with_individual_params(self):
        """Test initialization with individual connection parameters."""
        _backend = PostgresBackend(
            host="myhost",
            port=5433,
            database="mydb",
            user="myuser",
            password="mypass",
        )

        expected_conn_str = "postgresql://myuser:mypass@myhost:5433/mydb"
        _mock_psycopg2.connect.assert_called_with(expected_conn_str)

    def test_init_with_individual_params_no_password(self):
        """Test initialization without password."""
        _backend = PostgresBackend(
            host="myhost",
            port=5432,
            database="mydb",
            user="myuser",
            password="",
        )

        expected_conn_str = "postgresql://myuser@myhost:5432/mydb"
        _mock_psycopg2.connect.assert_called_with(expected_conn_str)

    def test_init_uses_env_vars_as_defaults(self):
        """Test initialization uses environment variables as defaults."""
        with patch.dict(
            "os.environ",
            {
                "PGHOST": "envhost",
                "PGDATABASE": "envdb",
                "PGUSER": "envuser",
                "PGPASSWORD": "envpass",
            },
        ):
            _backend = PostgresBackend()

            expected_conn_str = "postgresql://envuser:envpass@envhost:5432/envdb"
            _mock_psycopg2.connect.assert_called_with(expected_conn_str)

    def test_init_defaults_without_env_vars(self):
        """Test initialization defaults when no env vars."""
        with patch.dict("os.environ", {}, clear=True):
            _backend = PostgresBackend()

            expected_conn_str = "postgresql://postgres@localhost:5432/postgres"
            _mock_psycopg2.connect.assert_called_with(expected_conn_str)

    def test_init_custom_table_name(self):
        """Test initialization with custom table name."""
        backend = PostgresBackend(
            connection_string="postgresql://localhost/db", table_name="custom_table"
        )

        assert backend._table_name == "custom_table"

    def test_init_custom_embedding_dimension(self):
        """Test initialization with custom embedding dimension."""
        backend = PostgresBackend(
            connection_string="postgresql://localhost/db", embedding_dimension=768
        )

        assert backend._embedding_dimension == 768

    def test_init_creates_table(self):
        """Test initialization creates table with correct SQL."""
        _backend = PostgresBackend(
            connection_string="postgresql://localhost/db",
            table_name="test_embeddings",
            embedding_dimension=1536,
        )

        # Verify pgvector extension creation
        calls = [str(c) for c in _mock_cursor.execute.call_args_list]
        assert any("CREATE EXTENSION IF NOT EXISTS vector" in c for c in calls)

        # Verify table creation
        assert any("CREATE TABLE IF NOT EXISTS test_embeddings" in c for c in calls)

        # Verify index creation
        assert any("CREATE INDEX IF NOT EXISTS" in c for c in calls)

        # Verify commit was called
        _mock_connection.commit.assert_called()

    def test_init_similarity_stores_value(self):
        """Test initialization stores similarity value."""
        backend = PostgresBackend(
            connection_string="postgresql://localhost/db", similarity="euclidean"
        )

        assert backend._similarity == "euclidean"


# =============================================================================
# PostgresBackend Connection String Building Tests
# =============================================================================


class TestPostgresBackendConnectionString:
    """Tests for connection string building."""

    def test_build_connection_string_with_password(self, backend):
        """Test connection string building with password."""
        result = backend._build_connection_string(
            host="myhost",
            port=5432,
            database="mydb",
            user="myuser",
            password="secret",
        )

        assert result == "postgresql://myuser:secret@myhost:5432/mydb"

    def test_build_connection_string_without_password(self, backend):
        """Test connection string building without password."""
        result = backend._build_connection_string(
            host="myhost",
            port=5432,
            database="mydb",
            user="myuser",
            password="",
        )

        assert result == "postgresql://myuser@myhost:5432/mydb"

    def test_build_connection_string_custom_port(self, backend):
        """Test connection string building with custom port."""
        result = backend._build_connection_string(
            host="db.example.com",
            port=15432,
            database="production",
            user="admin",
            password="secure123",
        )

        assert result == "postgresql://admin:secure123@db.example.com:15432/production"


# =============================================================================
# PostgresBackend Add Tests
# =============================================================================


class TestPostgresBackendAdd:
    """Tests for PostgresBackend.add() method."""

    def test_add_single_document(self, backend):
        """Test adding a single document."""
        embedding = [0.1, 0.2, 0.3]
        text = "Hello world"
        metadata = {"source": "test"}

        result = backend.add(embeddings=[embedding], texts=[text], metadatas=[metadata])

        # Should return a list with one ID
        assert len(result) == 1
        assert isinstance(result[0], str)

        # Verify INSERT was executed
        _mock_cursor.execute.assert_called()
        call_args = _mock_cursor.execute.call_args[0]
        assert "INSERT INTO" in call_args[0]

        _mock_connection.commit.assert_called()

    def test_add_multiple_documents(self, backend):
        """Test adding multiple documents."""
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        texts = ["First", "Second", "Third"]
        metadatas = [{"idx": 1}, {"idx": 2}, {"idx": 3}]

        result = backend.add(embeddings=embeddings, texts=texts, metadatas=metadatas)

        assert len(result) == 3

    def test_add_with_custom_ids(self, backend):
        """Test adding documents with custom IDs."""
        custom_ids = ["id-1", "id-2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        texts = ["First", "Second"]

        result = backend.add(embeddings=embeddings, texts=texts, ids=custom_ids)

        assert result == custom_ids

    def test_add_without_metadatas(self, backend):
        """Test adding documents without metadata."""
        embeddings = [[0.1, 0.2]]
        texts = ["No metadata"]

        result = backend.add(embeddings=embeddings, texts=texts)

        assert len(result) == 1

        # Verify empty dict was used for metadata
        call_args = _mock_cursor.execute.call_args[0][1]
        assert json.loads(call_args[2]) == {}

    def test_add_empty_list(self, backend):
        """Test adding empty list returns empty list."""
        _mock_cursor.reset_mock()

        result = backend.add(embeddings=[], texts=[])

        assert result == []

    def test_add_upserts_on_conflict(self, backend):
        """Test add performs upsert on ID conflict."""
        embedding = [0.1, 0.2]
        text = "Updated text"
        doc_id = "existing-id"

        backend.add(embeddings=[embedding], texts=[text], ids=[doc_id])

        call_args = _mock_cursor.execute.call_args[0][0]
        assert "ON CONFLICT" in call_args
        assert "DO UPDATE SET" in call_args

    def test_add_generates_uuid_ids(self, backend):
        """Test add generates valid UUIDs when no IDs provided."""
        result = backend.add(embeddings=[[0.1]], texts=["test"])

        # Should be a valid UUID
        assert len(result) == 1
        uuid.UUID(result[0])  # This will raise if not valid UUID


# =============================================================================
# PostgresBackend Search Tests
# =============================================================================


class TestPostgresBackendSearch:
    """Tests for PostgresBackend.search() method."""

    def test_search_basic(self, backend):
        """Test basic search query."""
        # Configure cursor to return results
        _mock_cursor.fetchall.return_value = [
            ("id-1", "Hello world", {"source": "test"}, 0.95),
            ("id-2", "Hello there", {"source": "other"}, 0.85),
        ]

        query_embedding = [0.1, 0.2, 0.3]
        results = backend.search(query_embedding, k=5)

        assert len(results) == 2
        assert results[0]["id"] == "id-1"
        assert results[0]["text"] == "Hello world"
        assert results[0]["score"] == 0.95
        assert results[0]["metadata"] == {"source": "test"}

    def test_search_probes_all_configured_ivfflat_lists(self, backend):
        """Sparse IVFFlat partitions cannot make a non-empty store appear empty."""
        _mock_cursor.fetchall.return_value = []

        backend.search(query_embedding=[0.1, 0.2, 0.3])

        assert any(
            call.args == ("SELECT set_config('ivfflat.probes', %s, true)", ("100",))
            for call in _mock_cursor.execute.call_args_list
        )

    def test_search_with_filter(self, backend):
        """Test search with metadata filter."""
        _mock_cursor.fetchall.return_value = [
            ("id-1", "Filtered result", {"category": "important"}, 0.90),
        ]

        _results = backend.search(
            query_embedding=[0.1, 0.2],
            k=10,
            filter={"category": "important"},
        )

        # Verify filter was included in query
        call_args = _mock_cursor.execute.call_args[0]
        assert "WHERE" in call_args[0]
        assert "metadata->>" in call_args[0]

    def test_search_with_multiple_filters(self, backend):
        """Test search with multiple metadata filters."""
        _mock_cursor.fetchall.return_value = []

        backend.search(
            query_embedding=[0.1, 0.2],
            k=10,
            filter={"type": "document", "status": "active"},
        )

        call_args = _mock_cursor.execute.call_args[0]
        # Should have AND for multiple filters
        assert call_args[0].count("metadata->>") == 2

    def test_search_returns_empty_list_when_no_results(self, backend):
        """Test search returns empty list when no results."""
        _mock_cursor.fetchall.return_value = []

        results = backend.search(query_embedding=[0.1, 0.2], k=5)

        assert results == []

    def test_search_uses_cosine_similarity(self, backend):
        """Test search uses cosine similarity operator."""
        _mock_cursor.fetchall.return_value = []

        backend.search(query_embedding=[0.1, 0.2], k=5)

        call_args = _mock_cursor.execute.call_args[0][0]
        # <=> is the cosine distance operator in pgvector
        assert "<=>" in call_args
        # Score is 1 - distance for cosine
        assert "1 - (embedding <=> " in call_args

    def test_search_respects_k_parameter(self, backend):
        """Test search respects k (limit) parameter."""
        _mock_cursor.fetchall.return_value = []

        backend.search(query_embedding=[0.1, 0.2], k=100)

        call_args = _mock_cursor.execute.call_args[0]
        assert "LIMIT" in call_args[0]
        # k should be in the params
        assert 100 in call_args[1]

    def test_search_handles_string_metadata(self, backend):
        """Test search handles metadata returned as string."""
        # Some psycopg2 configs return JSON as string
        _mock_cursor.fetchall.return_value = [
            ("id-1", "Test", '{"key": "value"}', 0.90),
        ]

        results = backend.search(query_embedding=[0.1], k=5)

        assert results[0]["metadata"] == {"key": "value"}

    def test_search_handles_dict_metadata(self, backend):
        """Test search handles metadata returned as dict."""
        # psycopg2 with JSONB typically returns as dict
        _mock_cursor.fetchall.return_value = [
            ("id-1", "Test", {"key": "value"}, 0.90),
        ]

        results = backend.search(query_embedding=[0.1], k=5)

        assert results[0]["metadata"] == {"key": "value"}

    def test_search_handles_null_metadata(self, backend):
        """Test search handles null metadata."""
        _mock_cursor.fetchall.return_value = [
            ("id-1", "Test", None, 0.90),
        ]

        results = backend.search(query_embedding=[0.1], k=5)

        assert results[0]["metadata"] == {}

    def test_search_orders_by_similarity(self, backend):
        """Test search orders results by similarity."""
        _mock_cursor.fetchall.return_value = []

        backend.search(query_embedding=[0.1, 0.2], k=5)

        call_args = _mock_cursor.execute.call_args[0][0]
        assert "ORDER BY" in call_args
        assert "<=>" in call_args  # Orders by distance


# =============================================================================
# PostgresBackend Delete Tests
# =============================================================================


class TestPostgresBackendDelete:
    """Tests for PostgresBackend.delete() method."""

    def test_delete_single_id(self, backend):
        """Test deleting a single document."""
        _mock_cursor.rowcount = 1

        deleted = backend.delete(["id-1"])

        assert deleted == 1
        _mock_cursor.execute.assert_called()
        _mock_connection.commit.assert_called()

    def test_delete_multiple_ids(self, backend):
        """Test deleting multiple documents."""
        _mock_cursor.rowcount = 3

        deleted = backend.delete(["id-1", "id-2", "id-3"])

        assert deleted == 3

        # Verify uses ANY for batch delete
        call_args = _mock_cursor.execute.call_args[0][0]
        assert "ANY" in call_args

    def test_delete_empty_list(self, backend):
        """Test deleting empty list returns 0."""
        _mock_cursor.reset_mock()

        deleted = backend.delete([])

        assert deleted == 0

    def test_delete_nonexistent_ids(self, backend):
        """Test deleting non-existent IDs returns 0."""
        _mock_cursor.rowcount = 0

        deleted = backend.delete(["nonexistent-id"])

        assert deleted == 0

    def test_delete_returns_actual_count(self, backend):
        """Test delete returns actual number of deleted rows."""
        # Only 2 of 5 IDs existed
        _mock_cursor.rowcount = 2

        deleted = backend.delete(["id-1", "id-2", "id-3", "id-4", "id-5"])

        assert deleted == 2

    def test_delete_handles_none_rowcount(self, backend):
        """Test delete handles None rowcount."""
        _mock_cursor.rowcount = None

        deleted = backend.delete(["id-1"])

        assert deleted == 0


# =============================================================================
# PostgresBackend Clear Tests
# =============================================================================


class TestPostgresBackendClear:
    """Tests for PostgresBackend.clear() method."""

    def test_clear_truncates_table(self, backend):
        """Test clear truncates the table."""
        backend.clear()

        call_args = _mock_cursor.execute.call_args[0][0]
        assert "TRUNCATE TABLE" in call_args
        _mock_connection.commit.assert_called()


# =============================================================================
# PostgresBackend Count Tests
# =============================================================================


class TestPostgresBackendCount:
    """Tests for PostgresBackend.count() method."""

    def test_count_returns_row_count(self, backend):
        """Test count returns correct row count."""
        _mock_cursor.fetchone.return_value = (42,)

        count = backend.count()

        assert count == 42
        call_args = _mock_cursor.execute.call_args[0][0]
        assert "SELECT COUNT(*)" in call_args

    def test_count_returns_zero_for_empty_table(self, backend):
        """Test count returns 0 for empty table."""
        _mock_cursor.fetchone.return_value = (0,)

        count = backend.count()

        assert count == 0

    def test_count_handles_none_result(self, backend):
        """Test count handles None result."""
        _mock_cursor.fetchone.return_value = None

        count = backend.count()

        assert count == 0


# =============================================================================
# PostgresBackend Close Tests
# =============================================================================


class TestPostgresBackendClose:
    """Tests for PostgresBackend.close() method."""

    def test_close_closes_connection(self, backend):
        """Test close closes the database connection."""
        backend.close()

        _mock_connection.close.assert_called_once()

    def test_close_handles_none_connection(self, backend):
        """Test close handles case when connection is None."""
        backend._conn = None

        # Should not raise
        backend.close()


# =============================================================================
# PostgresBackend Edge Cases Tests
# =============================================================================


class TestPostgresBackendEdgeCases:
    """Tests for edge cases and error handling."""

    def test_large_embedding_dimension(self):
        """Test with large embedding dimension."""
        backend = PostgresBackend(
            connection_string="postgresql://localhost/db", embedding_dimension=4096
        )

        assert backend._embedding_dimension == 4096

    def test_special_characters_in_table_name(self):
        """Test with table name (note: should be validated in production)."""
        backend = PostgresBackend(
            connection_string="postgresql://localhost/db",
            table_name="my_special_table_123",
        )

        assert backend._table_name == "my_special_table_123"

    def test_search_with_high_dimensional_embedding(self, backend):
        """Test search with high-dimensional embedding."""
        _mock_cursor.fetchall.return_value = []

        # Create a large embedding
        query_embedding = [0.1] * 4096

        results = backend.search(query_embedding, k=5)

        assert results == []

    def test_add_with_unicode_text(self, backend):
        """Test adding documents with unicode text."""
        unicode_text = "Hello! 你好! مرحبا! "

        result = backend.add(embeddings=[[0.1, 0.2]], texts=[unicode_text])

        assert len(result) == 1
        # Verify the text was passed correctly
        call_args = _mock_cursor.execute.call_args[0][1]
        assert call_args[1] == unicode_text

    def test_add_with_complex_metadata(self, backend):
        """Test adding documents with complex nested metadata."""
        complex_metadata = {
            "nested": {"level1": {"level2": "value"}},
            "list": [1, 2, 3],
            "mixed": [{"key": "value"}, "string", 123],
        }

        result = backend.add(embeddings=[[0.1, 0.2]], texts=["test"], metadatas=[complex_metadata])

        assert len(result) == 1
        # Verify metadata was JSON serialized
        call_args = _mock_cursor.execute.call_args[0][1]
        assert json.loads(call_args[2]) == complex_metadata

    def test_search_filter_with_numeric_value(self, backend):
        """Test search filter with numeric value converts to string."""
        _mock_cursor.fetchall.return_value = []

        backend.search(query_embedding=[0.1], k=5, filter={"count": 42})

        # Verify the filter value was converted to string
        call_args = _mock_cursor.execute.call_args[0][1]
        assert "42" in call_args

    def test_add_preserves_order_of_ids(self, backend):
        """Test add returns IDs in the same order as input."""
        custom_ids = ["first", "second", "third"]
        embeddings = [[0.1], [0.2], [0.3]]
        texts = ["a", "b", "c"]

        result = backend.add(embeddings=embeddings, texts=texts, ids=custom_ids)

        assert result == custom_ids


# =============================================================================
# PostgresBackend Integration Pattern Tests
# =============================================================================


class TestPostgresBackendIntegrationPatterns:
    """Tests for common usage patterns."""

    def test_add_then_search_pattern(self, backend):
        """Test typical add-then-search usage pattern."""
        _mock_cursor.fetchall.return_value = [
            ("doc-1", "Matching document", {"category": "test"}, 0.95),
        ]

        # Add documents
        _ids = backend.add(
            embeddings=[[0.1, 0.2, 0.3]],
            texts=["Matching document"],
            metadatas=[{"category": "test"}],
            ids=["doc-1"],
        )

        # Search for similar
        results = backend.search(query_embedding=[0.1, 0.2, 0.3], k=5)

        assert len(results) == 1
        assert results[0]["id"] == "doc-1"

    def test_add_update_pattern(self, backend):
        """Test document update pattern (upsert)."""
        doc_id = "update-test"

        # Initial add
        backend.add(
            embeddings=[[0.1, 0.2]],
            texts=["Original text"],
            metadatas=[{"version": 1}],
            ids=[doc_id],
        )

        # Update (same ID)
        backend.add(
            embeddings=[[0.3, 0.4]],
            texts=["Updated text"],
            metadatas=[{"version": 2}],
            ids=[doc_id],
        )

        # Verify ON CONFLICT was used (upsert)
        calls = [str(c) for c in _mock_cursor.execute.call_args_list]
        # Both calls should use ON CONFLICT
        for c in calls:
            if "INSERT" in c:
                assert "ON CONFLICT" in c

    def test_delete_all_and_repopulate_pattern(self, backend):
        """Test clearing and repopulating pattern."""
        _mock_cursor.fetchone.return_value = (0,)

        # Clear all
        backend.clear()

        # Verify cleared
        count = backend.count()
        assert count == 0

        # Repopulate
        backend.add(embeddings=[[0.1]], texts=["New data"])
