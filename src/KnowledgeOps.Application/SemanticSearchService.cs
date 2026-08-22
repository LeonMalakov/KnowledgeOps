namespace KnowledgeOps.Application;

public sealed class SemanticSearchService(ITextEmbeddingService embeddings, IVectorStore vectorStore) {
    public const int MIN_TOP_K = 1;
    public const int MAX_TOP_K = 100;

    public async Task<IReadOnlyList<ChunkSearchResult>> SearchAsync(
        string query, int topK, CancellationToken cancellationToken = default) {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        ArgumentOutOfRangeException.ThrowIfLessThan(topK, MIN_TOP_K);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(topK, MAX_TOP_K);
        cancellationToken.ThrowIfCancellationRequested();

        var vector = await embeddings.EmbedQueryAsync(query, cancellationToken);
        await vectorStore.InitializeAsync(cancellationToken);
        return await vectorStore.SearchAsync(vector, topK, cancellationToken);
    }
}
