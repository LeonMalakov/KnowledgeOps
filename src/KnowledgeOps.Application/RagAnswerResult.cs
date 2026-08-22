namespace KnowledgeOps.Application;

/// <summary>The answer and the source excerpts passed to the model as context.</summary>
public sealed record RagAnswerResult(string Answer, IReadOnlyList<RagSource> Sources);

public sealed record RagSource(
    int Number,
    Guid DocumentId,
    int ChunkIndex,
    string Content,
    int StartOffset,
    int EndOffset,
    float Score,
    string Title = "",
    string? Source = null,
    float? RerankScore = null);
