using KnowledgeOps.Application;

namespace KnowledgeOps.Tests;

public class DocumentIndexingServiceTests {
    [Fact]
    public async Task Index_ShortDocument_PreservesContentAndOffsets() {
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore();
        var service = new DocumentIndexingService(new TextChunker(), embeddings, store);
        const string text = "  Invoice 🙂\n";

        var result = await service.IndexAsync("Invoice", text);

        Assert.Equal(1, result.ChunkCount);
        Assert.NotEqual(Guid.Empty, result.DocumentId);
        Assert.Equal(new[] { text }, embeddings.DocumentTexts);
        var point = Assert.Single(store.Points!);
        Assert.NotEqual(Guid.Empty, point.PointId);
        Assert.Equal(result.DocumentId, point.DocumentId);
        Assert.Equal("Invoice", point.Title);
        Assert.Null(point.Source);
        Assert.Equal((0, text, 0, text.Length), (point.ChunkIndex, point.Content, point.StartOffset, point.EndOffset));
    }

    [Fact]
    public async Task Index_MultipleChunks_BatchesInOrderAndInitializesBeforeUpsert() {
        using var cancellation = new CancellationTokenSource();
        float[][] vectors = [[1f, 0f, 0f], [0f, 2f, 0f], [0f, 0f, 3f]];
        var embeddings = new FakeTextEmbeddingService { DocumentVectors = vectors };
        var store = new FakeVectorStore();
        var service = new DocumentIndexingService(new TextChunker(maxChunkSize: 5, chunkOverlap: 2), embeddings, store);
        const string text = "abcdefghijk";

        var result = await service.IndexAsync("Guide", text, "https://example.test/guide", cancellation.Token);

        Assert.Equal(3, result.ChunkCount);
        Assert.Equal(1, embeddings.DocumentCallCount);
        Assert.Equal(new[] { "abcde", "defgh", "ghijk" }, embeddings.DocumentTexts);
        Assert.Equal(cancellation.Token, embeddings.ReceivedToken);
        Assert.Equal(new[] { (nameof(IVectorStore.InitializeAsync), cancellation.Token),
            (nameof(IVectorStore.UpsertAsync), cancellation.Token) }, store.Calls);
        var points = store.Points!;
        Assert.Equal(3, points.Select(point => point.PointId).Distinct().Count());
        for (int i = 0; i < points.Count; i++) {
            var point = points[i];
            Assert.NotEqual(Guid.Empty, point.PointId);
            Assert.Equal(result.DocumentId, point.DocumentId);
            Assert.Equal("Guide", point.Title);
            Assert.Equal("https://example.test/guide", point.Source);
            Assert.Equal(i, point.ChunkIndex);
            Assert.Equal((i * 3, i * 3 + 5), (point.StartOffset, point.EndOffset));
            Assert.Equal(text[point.StartOffset..point.EndOffset], point.Content);
            Assert.Equal(vectors[i], point.Vector);
        }
    }

    [Fact]
    public async Task Index_WhitespaceOnlyChunk_SkipsEmbeddingAndStorageButKeepsOriginalIndices() {
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore();
        var service = new DocumentIndexingService(new TextChunker(maxChunkSize: 4, chunkOverlap: 0), embeddings, store);

        var result = await service.IndexAsync("Guide", "abcd    efgh");

        Assert.Equal(2, result.ChunkCount);
        Assert.Equal(new[] { "abcd", "efgh" }, embeddings.DocumentTexts);
        Assert.Collection(store.Points!,
            point => Assert.Equal((0, 0, 4), (point.ChunkIndex, point.StartOffset, point.EndOffset)),
            point => Assert.Equal((2, 8, 12), (point.ChunkIndex, point.StartOffset, point.EndOffset)));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData(" \t\n")]
    public async Task Index_BlankContent_RejectsBeforeCallingDependencies(string? content) {
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore();
        var service = new DocumentIndexingService(new TextChunker(), embeddings, store);

        await Assert.ThrowsAnyAsync<ArgumentException>(() => service.IndexAsync("Guide", content!));
        Assert.Equal(0, embeddings.DocumentCallCount);
        Assert.Empty(store.Calls);
    }

    [Fact]
    public async Task Index_OversizedDocument_RejectsBeforeCallingDependencies() {
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore();
        var service = new DocumentIndexingService(new TextChunker(), embeddings, store);

        await Assert.ThrowsAsync<ArgumentOutOfRangeException>("content", () => service.IndexAsync("Guide", new string('x', 50_001)));
        Assert.Equal(0, embeddings.DocumentCallCount);
        Assert.Empty(store.Calls);
    }

    [Fact]
    public async Task Index_WrongEmbeddingCount_RejectsBeforeWriting() {
        var embeddings = new FakeTextEmbeddingService { DocumentVectors = [] };
        var store = new FakeVectorStore();
        var service = new DocumentIndexingService(new TextChunker(), embeddings, store);

        await Assert.ThrowsAsync<InvalidOperationException>(() => service.IndexAsync("Guide", "Text"));
        Assert.Empty(store.Calls);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData(" \n")]
    public async Task Index_BlankTitle_RejectsBeforeCallingDependencies(string? title) {
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore();
        var service = new DocumentIndexingService(new TextChunker(), embeddings, store);

        var exception = await Assert.ThrowsAnyAsync<ArgumentException>(() => service.IndexAsync(title!, "Text"));
        Assert.Equal("title", exception.ParamName);
        Assert.Equal(0, embeddings.DocumentCallCount);
        Assert.Empty(store.Calls);
    }

    [Theory]
    [InlineData(201, 1000, "title")]
    [InlineData(200, 1001, "source")]
    public async Task Index_OversizedMetadata_RejectsBeforeCallingDependencies(int titleLength, int sourceLength, string parameter) {
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore();
        var service = new DocumentIndexingService(new TextChunker(), embeddings, store);

        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(parameter,
            () => service.IndexAsync(new string('t', titleLength), "Text", new string('s', sourceLength)));
        Assert.Equal(0, embeddings.DocumentCallCount);
        Assert.Empty(store.Calls);
    }

    [Fact]
    public async Task DeleteDocument_InitializesThenForwardsDocumentIdAndCancellationToken() {
        using var cancellation = new CancellationTokenSource();
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore();
        var service = new DocumentIndexingService(new TextChunker(), embeddings, store);
        var documentId = Guid.NewGuid();

        await service.DeleteDocumentAsync(documentId, cancellation.Token);

        Assert.Equal(new[] {
            (nameof(IVectorStore.InitializeAsync), cancellation.Token),
            (nameof(IVectorStore.DeleteDocumentAsync), cancellation.Token)
        }, store.Calls);
        Assert.Equal(documentId, Assert.Single(store.DeletedDocumentIds));
        Assert.Equal(0, embeddings.DocumentCallCount);
    }
}
