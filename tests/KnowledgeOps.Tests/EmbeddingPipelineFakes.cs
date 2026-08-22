using KnowledgeOps.Application;

namespace KnowledgeOps.Tests;

internal sealed class FakeTextEmbeddingService : ITextEmbeddingService {
    public float[] QueryVector { get; init; } = [1f, 0f, 0f];
    public IReadOnlyList<float[]>? DocumentVectors { get; init; }
    public Exception? DocumentEmbeddingError { get; init; }
    public int DocumentCallCount { get; private set; }
    public int QueryCallCount { get; private set; }
    public IReadOnlyList<string>? DocumentTexts { get; private set; }
    public string? Query { get; private set; }
    public CancellationToken ReceivedToken { get; private set; }

    public Task<float[]> EmbedQueryAsync(string query, CancellationToken cancellationToken = default) {
        QueryCallCount++;
        Query = query;
        ReceivedToken = cancellationToken;
        return Task.FromResult(QueryVector);
    }

    public Task<IReadOnlyList<float[]>> EmbedDocumentsAsync(
        IReadOnlyList<string> texts, CancellationToken cancellationToken = default) {
        DocumentCallCount++;
        DocumentTexts = texts.ToArray();
        ReceivedToken = cancellationToken;
        if (DocumentEmbeddingError is not null) {
            return Task.FromException<IReadOnlyList<float[]>>(DocumentEmbeddingError);
        }

        return Task.FromResult(DocumentVectors ?? texts.Select((_, i) => new[] { i + 1f, 0f, 0f }).ToArray());
    }
}

internal sealed class FakeVectorStore : IVectorStore {
    public List<(string Operation, CancellationToken Token)> Calls { get; } = [];
    public IReadOnlyList<VectorizedChunk>? Points { get; private set; }
    public float[]? SearchVector { get; private set; }
    public string? HybridQuery { get; private set; }
    public int? TopK { get; private set; }
    public List<Guid> DeletedDocumentIds { get; } = [];
    public IReadOnlyList<Guid> ExistingPointIds { get; init; } = [];
    public Guid? RequestedDocumentId { get; private set; }
    public IReadOnlyList<Guid>? DeletedPointIds { get; private set; }
    public Func<Task>? BeforeUpsertCompletion { get; init; }
    public Exception? DeletePointsError { get; init; }
    public IReadOnlyList<ChunkSearchResult> SearchResults { get; init; } = [];
    public IReadOnlyList<ChunkSearchResult> HybridSearchResults { get; init; } = [];

    public Task InitializeAsync(CancellationToken cancellationToken = default) {
        Calls.Add((nameof(InitializeAsync), cancellationToken));
        return Task.CompletedTask;
    }

    public async Task UpsertAsync(IReadOnlyList<VectorizedChunk> chunks, CancellationToken cancellationToken = default) {
        Calls.Add((nameof(UpsertAsync), cancellationToken));
        if (BeforeUpsertCompletion is not null) {
            await BeforeUpsertCompletion();
        }

        Points = chunks.ToArray();
    }

    public Task<IReadOnlyList<Guid>> GetDocumentPointIdsAsync(Guid documentId, CancellationToken cancellationToken = default) {
        Calls.Add((nameof(GetDocumentPointIdsAsync), cancellationToken));
        RequestedDocumentId = documentId;
        return Task.FromResult(ExistingPointIds);
    }

    public Task DeletePointsAsync(IReadOnlyList<Guid> pointIds, CancellationToken cancellationToken = default) {
        Calls.Add((nameof(DeletePointsAsync), cancellationToken));
        DeletedPointIds = pointIds.ToArray();
        return DeletePointsError is null ? Task.CompletedTask : Task.FromException(DeletePointsError);
    }

    public Task DeleteDocumentAsync(Guid documentId, CancellationToken cancellationToken = default) {
        Calls.Add((nameof(DeleteDocumentAsync), cancellationToken));
        DeletedDocumentIds.Add(documentId);
        return Task.CompletedTask;
    }

    public Task<IReadOnlyList<ChunkSearchResult>> SearchAsync(
        float[] vector, int topK, CancellationToken cancellationToken = default) {
        Calls.Add((nameof(SearchAsync), cancellationToken));
        SearchVector = vector;
        TopK = topK;
        return Task.FromResult(SearchResults);
    }

    public Task<IReadOnlyList<ChunkSearchResult>> SearchHybridAsync(
        string query, float[] vector, int topK, CancellationToken cancellationToken = default) {
        Calls.Add((nameof(SearchHybridAsync), cancellationToken));
        HybridQuery = query;
        SearchVector = vector;
        TopK = topK;
        return Task.FromResult(HybridSearchResults);
    }
}
