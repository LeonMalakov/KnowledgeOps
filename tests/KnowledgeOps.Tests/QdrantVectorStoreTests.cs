using KnowledgeOps.Application;
using KnowledgeOps.Infrastructure;
using Qdrant.Client;
using Qdrant.Client.Grpc;
using Testcontainers.Qdrant;

namespace KnowledgeOps.Tests;

public class QdrantVectorStoreTests(QdrantFixture fixture) : IClassFixture<QdrantFixture> {
    [Fact]
    public async Task Initialize_CreatesAnEmptySearchableCollection() {
        var options = fixture.NewCollectionOptions();
        using var store = new QdrantVectorStore(options);
        using var client = new QdrantClient(options.Host, options.GrpcPort);

        await store.InitializeAsync();

        var collection = await client.GetCollectionInfoAsync(options.CollectionName);
        Assert.Equal(PayloadSchemaType.Keyword, collection.PayloadSchema["document_id"].DataType);
        Assert.Equal(PayloadSchemaType.Text, collection.PayloadSchema["content"].DataType);
        Assert.Equal(TextScoringType.Bm25, collection.PayloadSchema["content"].Params.TextIndexParams.Scoring.Type);
        Assert.Empty(await store.SearchAsync([1f, 0f, 0f], topK: 1));
    }

    [Fact]
    public async Task Initialize_ExistingCollectionAddsIndexConcurrentlyAndPreservesData() {
        var options = fixture.NewCollectionOptions();
        using var client = new QdrantClient(options.Host, options.GrpcPort);
        using var first = new QdrantVectorStore(options);
        using var second = new QdrantVectorStore(options);
        await client.CreateCollectionAsync(options.CollectionName, new VectorParams { Size = 3, Distance = Distance.Cosine });
        var chunk = new VectorizedChunk(Guid.NewGuid(), Guid.NewGuid(), 0, "Stored", 0, 6, [1f, 0f, 0f], "Guide");
        await first.UpsertAsync([chunk]);
        var before = await client.GetCollectionInfoAsync(options.CollectionName);
        Assert.False(before.PayloadSchema.ContainsKey("document_id"));

        await Task.WhenAll(first.InitializeAsync(), second.InitializeAsync(), first.InitializeAsync());
        await second.InitializeAsync();

        var after = await client.GetCollectionInfoAsync(options.CollectionName);
        Assert.Equal(PayloadSchemaType.Keyword, after.PayloadSchema["document_id"].DataType);
        Assert.Equal(TextScoringType.Bm25, after.PayloadSchema["content"].Params.TextIndexParams.Scoring.Type);
        var hit = Assert.Single(await first.SearchAsync([1f, 0f, 0f], topK: 1));
        Assert.Equal(ExpectedHit(chunk, hit.Score), hit);
        var hybridHit = Assert.Single(await first.SearchHybridAsync("stored", [1f, 0f, 0f], topK: 1));
        Assert.Equal(chunk.PointId, hybridHit.PointId);
    }

    [Fact]
    public async Task Initialize_ContentIndexWithoutScoring_ReportsIncompatibleConfiguration() {
        var options = fixture.NewCollectionOptions();
        using var client = new QdrantClient(options.Host, options.GrpcPort);
        using var store = new QdrantVectorStore(options);
        await client.CreateCollectionAsync(options.CollectionName, new VectorParams { Size = 3, Distance = Distance.Cosine });
        await client.CreatePayloadIndexAsync(options.CollectionName, "content", PayloadSchemaType.Text,
            new PayloadIndexParams { TextIndexParams = new TextIndexParams { Tokenizer = TokenizerType.Word } }, wait: true);

        var exception = await Assert.ThrowsAsync<InvalidOperationException>(() => store.InitializeAsync());

        Assert.Contains("'content' text index without BM25 scoring", exception.Message);
        var collection = await client.GetCollectionInfoAsync(options.CollectionName);
        Assert.Null(collection.PayloadSchema["content"].Params.TextIndexParams.Scoring);
    }

    [Fact]
    public async Task HybridSearch_UsesServerBm25ToPromoteLexicalHitWhileDenseKeepsCosineRanking() {
        var options = fixture.NewCollectionOptions();
        using var store = new QdrantVectorStore(options);
        using var client = new QdrantClient(options.Host, options.GrpcPort);
        await store.InitializeAsync();
        const string lexicalText = "Код SEC-240 означает смену номера.";
        const string semanticText = "Данные учетной записи и профиль.";
        var lexical = new VectorizedChunk(Guid.NewGuid(), Guid.NewGuid(), 0, lexicalText, 10, 10 + lexicalText.Length,
            [0f, 1f, 0f], "Смена номера", "help:sec240");
        var semantic = new VectorizedChunk(Guid.NewGuid(), Guid.NewGuid(), 0, semanticText, 0, semanticText.Length,
            [1f, 0f, 0f], "Профиль");
        await store.UpsertAsync([lexical, semantic]);
        float[] vector = [1f, 0f, 0f];

        var bm25 = await client.QueryAsync(options.CollectionName, query: new TextQuery { Query = "sec-240" },
            usingVector: "content", limit: 2);
        var dense = await store.SearchAsync(vector, topK: 1);
        var hybrid = await store.SearchHybridAsync("sec-240", vector, topK: 1);

        Assert.Equal(lexical.PointId, Guid.Parse(Assert.Single(bm25).Id.Uuid));
        Assert.Equal(semantic.PointId, Assert.Single(dense).PointId);
        var hit = Assert.Single(hybrid);
        Assert.Equal(ExpectedHit(lexical, hit.Score), hit);
        Assert.True(hit.Score > 0f);
        Assert.NotEqual(Assert.Single(dense).Score, hit.Score);
    }

    [Fact]
    public async Task Initialize_WrongPayloadIndexType_ThrowsAndLeavesExistingIndexIntact() {
        var options = fixture.NewCollectionOptions();
        using var client = new QdrantClient(options.Host, options.GrpcPort);
        using var store = new QdrantVectorStore(options);
        await client.CreateCollectionAsync(options.CollectionName, new VectorParams { Size = 3, Distance = Distance.Cosine });
        await client.CreatePayloadIndexAsync(options.CollectionName, "document_id", PayloadSchemaType.Integer, wait: true);

        var exception = await Assert.ThrowsAsync<InvalidOperationException>(() => store.InitializeAsync());

        Assert.Contains(options.CollectionName, exception.Message);
        Assert.Contains("'document_id' payload index of type Integer, expected Keyword", exception.Message);
        var collection = await client.GetCollectionInfoAsync(options.CollectionName);
        Assert.Equal(PayloadSchemaType.Integer, collection.PayloadSchema["document_id"].DataType);
    }

    [Fact]
    public async Task Search_ReturnsTopKByCosineScoreWithPayload() {
        using var store = new QdrantVectorStore(fixture.NewCollectionOptions());
        await store.InitializeAsync();
        var documentId = Guid.NewGuid();
        var exact = new VectorizedChunk(Guid.NewGuid(), documentId, 0, "Первый 🙂", 2, 11, [2f, 0f, 0f], "Руководство", "help:guide");
        var close = new VectorizedChunk(Guid.NewGuid(), documentId, 1, "Второй", 20, 26, [4f, 3f, 0f], "Руководство", "help:guide");
        var orthogonal = new VectorizedChunk(Guid.NewGuid(), Guid.NewGuid(), 0, "Other", 0, 5, [0f, 0f, 1f]);
        await store.UpsertAsync([orthogonal, close, exact]);

        var results = await store.SearchAsync([1f, 0f, 0f], topK: 2);

        Assert.Collection(results,
            hit => Assert.Equal(ExpectedHit(exact, hit.Score), hit),
            hit => Assert.Equal(ExpectedHit(close, hit.Score), hit));
        Assert.True(results[0].Score > results[1].Score);
        Assert.InRange(results[0].Score, 0.999f, 1.001f);
        Assert.InRange(results[1].Score, 0.799f, 0.801f);
    }

    [Fact]
    public async Task Search_LegacyPayloadWithoutTitleOrSource_ReturnsSafeDefaults() {
        var options = fixture.NewCollectionOptions();
        using var store = new QdrantVectorStore(options);
        using var client = new QdrantClient(options.Host, options.GrpcPort);
        await store.InitializeAsync();
        var documentId = Guid.NewGuid();
        var pointId = Guid.NewGuid();
        await client.UpsertAsync(options.CollectionName, [new PointStruct {
            Id = pointId,
            Vectors = new[] { 1f, 0f, 0f },
            Payload = {
                ["document_id"] = documentId.ToString(),
                ["chunk_index"] = 0,
                ["content"] = "Legacy",
                ["start_offset"] = 0,
                ["end_offset"] = 6
            }
        }], wait: true);

        var hit = Assert.Single(await store.SearchAsync([1f, 0f, 0f], topK: 1));

        Assert.Equal(pointId, hit.PointId);
        Assert.Equal(documentId, hit.DocumentId);
        Assert.Equal("Legacy", hit.Content);
        Assert.Equal(string.Empty, hit.Title);
        Assert.Null(hit.Source);
    }

    [Fact]
    public async Task DeleteDocument_DeletesAllMatchingChunksAndKeepsOtherDocumentsAfterRepeatedDelete() {
        using var store = new QdrantVectorStore(fixture.NewCollectionOptions());
        await store.InitializeAsync();
        var documentId = Guid.Parse("fba91461-61b0-4b16-bbc3-f260103a0011");
        var otherDocumentId = Guid.Parse("fba91461-61b0-4b16-bbc3-f260103a0012");
        float[] vector = [1f, 0f, 0f];
        var first = new VectorizedChunk(Guid.NewGuid(), documentId, 0, "First", 0, 5, vector, "Delete me");
        var second = new VectorizedChunk(Guid.NewGuid(), documentId, 1, "Second", 5, 11, vector, "Delete me");
        var kept = new VectorizedChunk(documentId, otherDocumentId, 0, "Keep", 0, 4, vector, "Keep me", "help:keep");
        await store.UpsertAsync([first, second, kept]);
        Assert.Equal(3, (await store.SearchAsync(vector, topK: 5)).Count);

        await store.DeleteDocumentAsync(documentId);
        var remaining = Assert.Single(await store.SearchAsync(vector, topK: 5));
        Assert.Equal(ExpectedHit(kept, remaining.Score), remaining);

        await store.DeleteDocumentAsync(documentId);
        Assert.Equal(remaining, Assert.Single(await store.SearchAsync(vector, topK: 5)));
    }

    [Fact]
    public async Task GetDocumentPointIds_ReturnsAllMatchingPointsAcrossMultiplePages() {
        using var store = new QdrantVectorStore(fixture.NewCollectionOptions());
        await store.InitializeAsync();
        var documentId = Guid.NewGuid();
        var otherDocumentId = Guid.NewGuid();
        var points = Enumerable.Range(0, 225).Select(index => new VectorizedChunk(
            Guid.NewGuid(), documentId, index, "x", index, index + 1, [1f, 0f, 0f], "Paged document")).ToArray();
        await store.UpsertAsync([.. points, new VectorizedChunk(Guid.NewGuid(), otherDocumentId, 0, "y", 0, 1, [1f, 0f, 0f], "Other")]);

        var ids = await store.GetDocumentPointIdsAsync(documentId);

        Assert.Equal(points.Length, ids.Count);
        Assert.Equal(points.Select(point => point.PointId).Order(), ids.Order());
        Assert.Empty(await store.GetDocumentPointIdsAsync(Guid.NewGuid()));
    }

    [Fact]
    public async Task UpdateDocument_ReplacesAllOldChunksAndPreservesAnotherDocumentAcrossRepeatedUpdates() {
        using var store = new QdrantVectorStore(fixture.NewCollectionOptions());
        var indexing = new DocumentIndexingService(
            new TextChunker(maxChunkSize: 4, chunkOverlap: 0), new FakeTextEmbeddingService(), store);
        var original = await indexing.IndexAsync("Original", "abcdefghijkl", "help:before");
        var other = await indexing.IndexAsync("Other", "kept", "help:other");
        var originalIds = await store.GetDocumentPointIdsAsync(original.DocumentId);
        var otherId = Assert.Single(await store.GetDocumentPointIdsAsync(other.DocumentId));
        Assert.Equal(3, original.ChunkCount);
        Assert.Equal(3, originalIds.Count);

        var updated = await indexing.UpdateAsync(original.DocumentId, "Updated", "five", "help:after");

        Assert.Equal(new DocumentIndexingResult(original.DocumentId, 1), updated);
        var updatedId = Assert.Single(await store.GetDocumentPointIdsAsync(original.DocumentId));
        Assert.DoesNotContain(updatedId, originalIds);
        var results = await store.SearchAsync([1f, 0f, 0f], topK: 10);
        Assert.Equal(2, results.Count);
        var hit = Assert.Single(results, result => result.DocumentId == original.DocumentId);
        Assert.Equal((updatedId, "Updated", "help:after", "five", 0, 0, 4),
            (hit.PointId, hit.Title, hit.Source, hit.Content, hit.ChunkIndex, hit.StartOffset, hit.EndOffset));
        Assert.Equal(otherId, Assert.Single(results, result => result.DocumentId == other.DocumentId).PointId);

        var repeated = await indexing.UpdateAsync(original.DocumentId, "Latest", "six");

        Assert.Equal(new DocumentIndexingResult(original.DocumentId, 1), repeated);
        var finalResults = await store.SearchAsync([1f, 0f, 0f], topK: 10);
        Assert.Equal(2, finalResults.Count);
        var finalHit = Assert.Single(finalResults, result => result.DocumentId == original.DocumentId);
        Assert.NotEqual(updatedId, finalHit.PointId);
        Assert.DoesNotContain(finalHit.PointId, originalIds);
        Assert.Equal(("Latest", "six", 3), (finalHit.Title, finalHit.Content, finalHit.EndOffset));
        Assert.Null(finalHit.Source);
        Assert.Equal(otherId, Assert.Single(finalResults, result => result.DocumentId == other.DocumentId).PointId);
    }

    [Fact]
    public async Task Upsert_SamePointId_ReplacesPayloadAndVector() {
        using var store = new QdrantVectorStore(fixture.NewCollectionOptions());
        await store.InitializeAsync();
        var original = new VectorizedChunk(Guid.NewGuid(), Guid.NewGuid(), 0, "Before", 0, 6, [1f, 0f, 0f]);
        var updated = original with { ChunkIndex = 1, Content = "After", EndOffset = 5, Vector = [0f, 1f, 0f] };

        await store.UpsertAsync([original]);
        await store.UpsertAsync([updated]);

        var hit = Assert.Single(await store.SearchAsync([0f, 1f, 0f], topK: 5));
        Assert.Equal(ExpectedHit(updated, hit.Score), hit);
        Assert.InRange(hit.Score, 0.999f, 1.001f);
    }

    [Fact]
    public async Task Initialize_ExistingCollectionPreservesDataAndRejectsWrongSize() {
        var options = fixture.NewCollectionOptions();
        var chunk = new VectorizedChunk(Guid.NewGuid(), Guid.NewGuid(), 0, "Stored", 0, 6, [1f, 0f, 0f]);
        using (var original = new QdrantVectorStore(options)) {
            await original.InitializeAsync();
            await original.UpsertAsync([chunk]);
        }

        using var wrongSize = new QdrantVectorStore(options with { VectorSize = 4 });
        var exception = await Assert.ThrowsAsync<InvalidOperationException>(() => wrongSize.InitializeAsync());
        Assert.Contains("vector size 3, expected 4", exception.Message);

        using var reopened = new QdrantVectorStore(options);
        await reopened.InitializeAsync();
        var hit = Assert.Single(await reopened.SearchAsync([1f, 0f, 0f], topK: 1));
        Assert.Equal(ExpectedHit(chunk, hit.Score), hit);
    }

    [Fact]
    public async Task Initialize_ConcurrentInitializersUseTheSameCollection() {
        var options = fixture.NewCollectionOptions();
        using var first = new QdrantVectorStore(options);
        using var second = new QdrantVectorStore(options);

        await Task.WhenAll(first.InitializeAsync(), second.InitializeAsync());

        var chunk = new VectorizedChunk(Guid.NewGuid(), Guid.NewGuid(), 0, "Shared", 0, 6, [1f, 0f, 0f]);
        await first.UpsertAsync([chunk]);
        var hit = Assert.Single(await second.SearchAsync([1f, 0f, 0f], topK: 1));
        Assert.Equal(chunk.PointId, hit.PointId);
    }

    [Fact]
    public async Task Upsert_ZeroVector_RejectsEntireBatchBeforeWriting() {
        using var store = new QdrantVectorStore(fixture.NewCollectionOptions());
        await store.InitializeAsync();
        var valid = new VectorizedChunk(Guid.NewGuid(), Guid.NewGuid(), 0, "Valid", 0, 5, [1f, 0f, 0f]);
        var zero = valid with { PointId = Guid.NewGuid(), Vector = [0f, 0f, 0f] };

        var exception = await Assert.ThrowsAsync<ArgumentException>("vector", () => store.UpsertAsync([valid, zero]));

        Assert.Contains("non-zero norm", exception.Message);
        Assert.Empty(await store.SearchAsync(valid.Vector, topK: 2));
    }

    [Fact]
    public async Task Search_ZeroVector_RejectsQueryBeforeCallingQdrant() {
        using var store = new QdrantVectorStore(fixture.NewCollectionOptions());

        var exception = await Assert.ThrowsAsync<ArgumentException>("vector", () => store.SearchAsync([0f, -0f, 0f], topK: 1));

        Assert.Contains("non-zero norm", exception.Message);
    }

    [Fact]
    public async Task InvalidInputs_AreRejectedBeforeWritingAnyPoints() {
        using var store = new QdrantVectorStore(fixture.NewCollectionOptions());
        await store.InitializeAsync();
        var valid = new VectorizedChunk(Guid.NewGuid(), Guid.NewGuid(), 0, "Valid", 0, 5, [1f, 0f, 0f]);

        await Assert.ThrowsAsync<ArgumentOutOfRangeException>("topK", () => store.SearchAsync(valid.Vector, topK: 0));
        foreach (float[] invalidVector in new float[][] { [1f, 0f], [1f, float.NaN, 0f], [1f, 0f, float.PositiveInfinity] }) {
            await Assert.ThrowsAsync<ArgumentException>("vector", () => store.SearchAsync(invalidVector, topK: 1));
            await Assert.ThrowsAsync<ArgumentException>("vector", () => store.UpsertAsync([valid, valid with { Vector = invalidVector }]));
        }

        var invalidChunks = new[] {
            valid with { PointId = Guid.Empty },
            valid with { DocumentId = Guid.Empty },
            valid with { ChunkIndex = -1 },
            valid with { StartOffset = -1 },
            valid with { EndOffset = 0 },
            valid with { EndOffset = 4 }
        };
        foreach (var invalid in invalidChunks) {
            await Assert.ThrowsAsync<ArgumentException>("chunks", () => store.UpsertAsync([valid, invalid]));
        }

        Assert.Empty(await store.SearchAsync(valid.Vector, topK: 1));
    }

    private static ChunkSearchResult ExpectedHit(VectorizedChunk chunk, float score) {
        return new ChunkSearchResult(
            chunk.PointId, chunk.DocumentId, chunk.ChunkIndex, chunk.Content, chunk.StartOffset, chunk.EndOffset, score,
            chunk.Title, chunk.Source);
    }
}

public sealed class QdrantFixture : IAsyncLifetime {
    private readonly QdrantContainer _container = new QdrantBuilder("qdrant/qdrant:v1.20.0").Build();

    public Task InitializeAsync() => _container.StartAsync();

    public Task DisposeAsync() => _container.DisposeAsync().AsTask();

    public QdrantVectorStoreOptions NewCollectionOptions() {
        var endpoint = new Uri(_container.GetGrpcConnectionString());
        return new QdrantVectorStoreOptions(endpoint.Host, endpoint.Port, $"chunks_test_{Guid.NewGuid():N}", VectorSize: 3);
    }
}
