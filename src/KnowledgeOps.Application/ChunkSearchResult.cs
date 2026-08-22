namespace KnowledgeOps.Application;

public sealed record ChunkSearchResult(
    Guid PointId,
    Guid DocumentId,
    int ChunkIndex,
    string Content,
    int StartOffset,
    int EndOffset,
    float Score,
    string Title = "",
    string? Source = null);
