namespace KnowledgeOps.Application;

public interface IVectorStore {
    /// <summary>Ensures storage is ready for reads and writes.</summary>
    Task InitializeAsync(CancellationToken cancellationToken = default);

    Task UpsertAsync(IReadOnlyList<VectorizedChunk> chunks, CancellationToken cancellationToken = default);

    Task<IReadOnlyList<Guid>> GetDocumentPointIdsAsync(Guid documentId, CancellationToken cancellationToken = default);

    Task DeletePointsAsync(IReadOnlyList<Guid> pointIds, CancellationToken cancellationToken = default);

    Task DeleteDocumentAsync(Guid documentId, CancellationToken cancellationToken = default);

    Task<IReadOnlyList<ChunkSearchResult>> SearchAsync(
        float[] vector, int topK, CancellationToken cancellationToken = default);

    Task<IReadOnlyList<ChunkSearchResult>> SearchHybridAsync(
        string query, float[] vector, int topK, CancellationToken cancellationToken = default);
}
