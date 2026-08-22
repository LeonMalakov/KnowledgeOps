using KnowledgeOps.Application;

namespace KnowledgeOps.Tests;

public class DocumentUpdateServiceTests {
    [Fact]
    public async Task Update_PreservesDocumentIdAndDeletesOldIdsOnlyAfterUpsertCompletes() {
        using var cancellation = new CancellationTokenSource();
        var releaseUpsert = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        Guid[] oldIds = [Guid.NewGuid(), Guid.NewGuid()];
        var store = new FakeVectorStore {
            ExistingPointIds = oldIds,
            BeforeUpsertCompletion = () => releaseUpsert.Task
        };
        var embeddings = new FakeTextEmbeddingService();
        var service = CreateService(embeddings, store);
        var documentId = Guid.NewGuid();
        const string text = "abcde🙂ghi";

        var update = service.UpdateAsync(documentId, "Updated title", text, "help:updated", cancellation.Token);

        Assert.Contains(store.Calls, call => call.Operation == nameof(IVectorStore.UpsertAsync));
        Assert.False(update.IsCompleted);
        Assert.Null(store.DeletedPointIds);
        releaseUpsert.SetResult();
        var result = await update;

        Assert.Equal(new DocumentIndexingResult(documentId, 2), result);
        Assert.Equal(documentId, store.RequestedDocumentId);
        Assert.Equal(new[] { "abcde", "🙂ghi" }, embeddings.DocumentTexts);
        Assert.Equal(1, embeddings.DocumentCallCount);
        Assert.Equal(cancellation.Token, embeddings.ReceivedToken);
        Assert.Equal(oldIds, store.DeletedPointIds);
        Assert.Empty(store.DeletedDocumentIds);
        Assert.Equal(new[] {
            (nameof(IVectorStore.InitializeAsync), cancellation.Token),
            (nameof(IVectorStore.GetDocumentPointIdsAsync), cancellation.Token),
            (nameof(IVectorStore.UpsertAsync), cancellation.Token),
            (nameof(IVectorStore.DeletePointsAsync), cancellation.Token)
        }, store.Calls);

        var points = store.Points!;
        Assert.Equal(2, points.Select(point => point.PointId).Distinct().Count());
        Assert.All(points, point => {
            Assert.NotEqual(Guid.Empty, point.PointId);
            Assert.DoesNotContain(point.PointId, oldIds);
            Assert.Equal(documentId, point.DocumentId);
            Assert.Equal("Updated title", point.Title);
            Assert.Equal("help:updated", point.Source);
        });
        Assert.Equal((0, "abcde", 0, 5), (points[0].ChunkIndex, points[0].Content, points[0].StartOffset, points[0].EndOffset));
        Assert.Equal((1, "🙂ghi", 5, 10), (points[1].ChunkIndex, points[1].Content, points[1].StartOffset, points[1].EndOffset));
    }

    [Fact]
    public async Task Update_MissingDocument_ReturnsNullBeforeEmbeddingOrWriting() {
        var store = new FakeVectorStore();
        var embeddings = new FakeTextEmbeddingService();
        var documentId = Guid.NewGuid();

        var result = await CreateService(embeddings, store).UpdateAsync(documentId, "Title", "Text");

        Assert.Null(result);
        Assert.Equal(documentId, store.RequestedDocumentId);
        Assert.Equal(0, embeddings.DocumentCallCount);
        Assert.Equal(new[] { nameof(IVectorStore.InitializeAsync), nameof(IVectorStore.GetDocumentPointIdsAsync) },
            store.Calls.Select(call => call.Operation));
    }

    [Fact]
    public async Task Update_EmbeddingFailure_DoesNotWriteOrDeletePoints() {
        var failure = new InvalidOperationException("Embedding failed.");
        var embeddings = new FakeTextEmbeddingService { DocumentEmbeddingError = failure };
        var store = new FakeVectorStore { ExistingPointIds = [Guid.NewGuid()] };
        var service = CreateService(embeddings, store);

        var exception = await Assert.ThrowsAsync<InvalidOperationException>(
            () => service.UpdateAsync(Guid.NewGuid(), "Title", "Text"));

        Assert.Same(failure, exception);
        Assert.Equal(new[] { nameof(IVectorStore.InitializeAsync), nameof(IVectorStore.GetDocumentPointIdsAsync) },
            store.Calls.Select(call => call.Operation));
    }

    [Fact]
    public async Task Update_UpsertFailure_DoesNotDeleteOldPoints() {
        var failure = new InvalidOperationException("Upsert failed.");
        var store = new FakeVectorStore {
            ExistingPointIds = [Guid.NewGuid()],
            BeforeUpsertCompletion = () => Task.FromException(failure)
        };
        var service = CreateService(new FakeTextEmbeddingService(), store);

        var exception = await Assert.ThrowsAsync<InvalidOperationException>(
            () => service.UpdateAsync(Guid.NewGuid(), "Title", "Text"));

        Assert.Same(failure, exception);
        Assert.Equal(new[] { nameof(IVectorStore.InitializeAsync), nameof(IVectorStore.GetDocumentPointIdsAsync),
            nameof(IVectorStore.UpsertAsync) }, store.Calls.Select(call => call.Operation));
    }

    [Fact]
    public async Task Update_DeleteFailure_ReportsUncertainOutcomeWithoutRollingBackNewPoints() {
        var failure = new InvalidOperationException("Deletion response was lost.");
        var documentId = Guid.NewGuid();
        Guid[] oldIds = [Guid.NewGuid(), Guid.NewGuid()];
        var store = new FakeVectorStore { ExistingPointIds = oldIds, DeletePointsError = failure };
        var service = CreateService(new FakeTextEmbeddingService(), store);

        var exception = await Assert.ThrowsAsync<DocumentUpdateCleanupException>(
            () => service.UpdateAsync(documentId, "Title", "Text"));

        Assert.Same(failure, exception.InnerException);
        Assert.Contains(documentId.ToString(), exception.Message);
        Assert.Equal(oldIds, store.DeletedPointIds);
        Assert.Equal(documentId, Assert.Single(store.Points!).DocumentId);
        Assert.Equal(new[] { nameof(IVectorStore.InitializeAsync), nameof(IVectorStore.GetDocumentPointIdsAsync),
            nameof(IVectorStore.UpsertAsync), nameof(IVectorStore.DeletePointsAsync) }, store.Calls.Select(call => call.Operation));
    }

    private static DocumentIndexingService CreateService(FakeTextEmbeddingService embeddings, FakeVectorStore store) {
        return new DocumentIndexingService(new TextChunker(maxChunkSize: 5, chunkOverlap: 0), embeddings, store);
    }
}
