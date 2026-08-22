namespace KnowledgeOps.Application;

public sealed class HybridSearchService(ITextEmbeddingService embeddings, IVectorStore vectorStore) {
    public async Task<IReadOnlyList<ChunkSearchResult>> SearchAsync(
        string query, int topK, CancellationToken cancellationToken = default) {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        ArgumentOutOfRangeException.ThrowIfLessThan(topK, SemanticSearchService.MIN_TOP_K);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(topK, SemanticSearchService.MAX_TOP_K);
        cancellationToken.ThrowIfCancellationRequested();

        var vector = await embeddings.EmbedQueryAsync(query, cancellationToken);
        await vectorStore.InitializeAsync(cancellationToken);
        return await vectorStore.SearchHybridAsync(query, vector, topK, cancellationToken);
    }
}
