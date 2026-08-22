namespace KnowledgeOps.Application;

public interface ITextEmbeddingService {
    Task<float[]> EmbedQueryAsync(string query, CancellationToken cancellationToken = default);

    Task<IReadOnlyList<float[]>> EmbedDocumentsAsync(
        IReadOnlyList<string> texts, CancellationToken cancellationToken = default);
}
