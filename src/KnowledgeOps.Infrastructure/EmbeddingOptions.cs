namespace KnowledgeOps.Infrastructure;

public sealed class EmbeddingOptions {
    public const string SECTION_NAME = "Embedding";

    public string BaseUrl { get; init; } = string.Empty;

    public string Model { get; init; } = string.Empty;

    public int VectorSize { get; init; }
}
