namespace KnowledgeOps.Application;

public sealed class DocumentIndexingService(
    TextChunker chunker, ITextEmbeddingService embeddings, IVectorStore vectorStore) {
    public const int MAX_CONTENT_LENGTH = 50_000;
    public const int MAX_TITLE_LENGTH = 200;
    public const int MAX_SOURCE_LENGTH = 1_000;

    public async Task<DocumentIndexingResult> IndexAsync(
        string title, string content, string? source = null, CancellationToken cancellationToken = default) {
        ValidateDocument(title, content, source);
        cancellationToken.ThrowIfCancellationRequested();

        var documentId = Guid.NewGuid();
        var points = await CreatePointsAsync(documentId, title, content, source, cancellationToken);
        await vectorStore.InitializeAsync(cancellationToken);
        await vectorStore.UpsertAsync(points, cancellationToken);
        return new DocumentIndexingResult(documentId, points.Length);
    }

    public async Task<DocumentIndexingResult?> UpdateAsync(
        Guid documentId, string title, string content, string? source = null, CancellationToken cancellationToken = default) {
        ValidateDocumentId(documentId);
        ValidateDocument(title, content, source);
        cancellationToken.ThrowIfCancellationRequested();

        await vectorStore.InitializeAsync(cancellationToken);
        var oldPointIds = await vectorStore.GetDocumentPointIdsAsync(documentId, cancellationToken);
        if (oldPointIds.Count == 0) {
            return null;
        }

        var points = await CreatePointsAsync(documentId, title, content, source, cancellationToken);
        await vectorStore.UpsertAsync(points, cancellationToken);

        try {
            await vectorStore.DeletePointsAsync(oldPointIds, cancellationToken);
        } catch (Exception exception) {
            // A failed response does not establish whether Qdrant applied the deletion.
            throw new DocumentUpdateCleanupException(documentId, exception);
        }

        return new DocumentIndexingResult(documentId, points.Length);
    }

    public async Task DeleteDocumentAsync(Guid documentId, CancellationToken cancellationToken = default) {
        ValidateDocumentId(documentId);
        cancellationToken.ThrowIfCancellationRequested();
        await vectorStore.InitializeAsync(cancellationToken);
        await vectorStore.DeleteDocumentAsync(documentId, cancellationToken);
    }

    private async Task<VectorizedChunk[]> CreatePointsAsync(
        Guid documentId, string title, string content, string? source, CancellationToken cancellationToken) {
        string? sourceIdentifier = string.IsNullOrWhiteSpace(source) ? null : source;
        var chunks = chunker.Chunk(content).Where(chunk => !string.IsNullOrWhiteSpace(chunk.Content)).ToArray();
        var texts = chunks.Select(chunk => chunk.Content).ToArray();
        var vectors = await embeddings.EmbedDocumentsAsync(texts, cancellationToken);
        if (vectors.Count != chunks.Length) {
            throw new InvalidOperationException($"Received {vectors.Count} embeddings for {chunks.Length} document chunks.");
        }

        var points = new VectorizedChunk[chunks.Length];
        for (int i = 0; i < points.Length; i++) {
            var chunk = chunks[i];
            points[i] = new VectorizedChunk(Guid.NewGuid(), documentId, chunk.Index, chunk.Content,
                chunk.StartOffset, chunk.EndOffset, vectors[i], title, sourceIdentifier);
        }

        return points;
    }

    private static void ValidateDocument(string title, string content, string? source) {
        ArgumentException.ThrowIfNullOrWhiteSpace(title);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(title.Length, MAX_TITLE_LENGTH, nameof(title));
        ArgumentOutOfRangeException.ThrowIfGreaterThan(source?.Length ?? 0, MAX_SOURCE_LENGTH, nameof(source));
        ArgumentException.ThrowIfNullOrWhiteSpace(content);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(content.Length, MAX_CONTENT_LENGTH, nameof(content));
    }

    private static void ValidateDocumentId(Guid documentId) {
        if (documentId == Guid.Empty) {
            throw new ArgumentException("Document ID must be non-empty.", nameof(documentId));
        }
    }
}

public sealed record DocumentIndexingResult(Guid DocumentId, int ChunkCount);

public sealed class DocumentUpdateCleanupException(Guid documentId, Exception innerException)
    : InvalidOperationException(
        $"New chunks for document '{documentId}' were saved, but deletion of the old chunks could not be confirmed.",
        innerException);
