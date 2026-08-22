using KnowledgeOps.Application;

namespace KnowledgeOps.Tests;

public class HybridSearchServiceTests {
    [Fact]
    public async Task Search_EmbedsOnceAndForwardsRawQueryVectorTopKAndCancellation() {
        using var cancellation = new CancellationTokenSource();
        var hit = new ChunkSearchResult(Guid.NewGuid(), Guid.NewGuid(), 1, "Код DL-17", 10, 19, 0.83f, "Доставка", "help:delivery");
        var embeddings = new FakeTextEmbeddingService { QueryVector = [0f, 2f, 1f] };
        var store = new FakeVectorStore { HybridSearchResults = [hit] };
        var service = new HybridSearchService(embeddings, store);
        const string query = "  Что значит DL-17? ";

        var results = await service.SearchAsync(query, 5, cancellation.Token);

        Assert.Equal(1, embeddings.QueryCallCount);
        Assert.Equal(query, embeddings.Query);
        Assert.Equal(query, store.HybridQuery);
        Assert.Same(embeddings.QueryVector, store.SearchVector);
        Assert.Equal(5, store.TopK);
        Assert.Equal(cancellation.Token, embeddings.ReceivedToken);
        Assert.Equal(new[] {
            (nameof(IVectorStore.InitializeAsync), cancellation.Token),
            (nameof(IVectorStore.SearchHybridAsync), cancellation.Token)
        }, store.Calls);
        Assert.Equal(hit, Assert.Single(results));
    }

    [Theory]
    [InlineData("", 5)]
    [InlineData(" \t", 5)]
    [InlineData("DL-17", 0)]
    [InlineData("DL-17", 101)]
    public async Task Search_InvalidInput_DoesNotCallDependencies(string query, int topK) {
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore();
        var service = new HybridSearchService(embeddings, store);

        await Assert.ThrowsAnyAsync<ArgumentException>(() => service.SearchAsync(query, topK));
        Assert.Equal(0, embeddings.QueryCallCount);
        Assert.Empty(store.Calls);
    }
}
