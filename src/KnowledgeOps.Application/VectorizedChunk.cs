namespace KnowledgeOps.Application;

/// <summary>
/// A document chunk with a precomputed vector and a half-open UTF-16 source range.
/// </summary>
public sealed record VectorizedChunk(
    Guid PointId,
    Guid DocumentId,
    int ChunkIndex,
    string Content,
    int StartOffset,
    int EndOffset,
    float[] Vector,
    string Title = "",
    string? Source = null);
