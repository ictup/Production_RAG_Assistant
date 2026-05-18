from collections.abc import Sequence

from backend.app.rag.embeddings import EmbeddingClient, validate_embedding_batch
from ingestion.models import Chunk


async def embed_chunks(
    chunks: Sequence[Chunk],
    embedding_client: EmbeddingClient,
) -> list[list[float]]:
    if not chunks:
        return []

    embeddings = await embedding_client.embed_texts([chunk.text for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise ValueError(
            f"embedding count {len(embeddings)} does not match "
            f"chunk count {len(chunks)}"
        )

    validate_embedding_batch(
        embeddings,
        expected_dimension=embedding_client.dimension,
    )
    return embeddings
