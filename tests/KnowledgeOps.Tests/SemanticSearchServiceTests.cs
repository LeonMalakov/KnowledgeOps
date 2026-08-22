using KnowledgeOps.Application;

namespace KnowledgeOps.Tests;

public class SemanticSearchServiceTests {
    [Fact]
    public async Task Search_EmbedsQueryAndReturnsStoreResultsWithTopKAndToken() {
        using var cancellation = new CancellationTokenSource();
        var hit = new ChunkSearchResult(Guid.NewGuid(), Guid.NewGuid(), 0, "Delivery", 4, 12, 0.9f);
        var embeddings = new FakeTextEmbeddingService { QueryVector = [0f, 2f, 1f] };
        var store = new FakeVectorStore { SearchResults = [hit] };
        var service = new SemanticSearchService(embeddings, store);

        var results = await service.SearchAsync("  When will it arrive? ", 7, cancellation.Token);

        Assert.Equal("  When will it arrive? ", embeddings.Query);
        Assert.Equal(cancellation.Token, embeddings.ReceivedToken);
        Assert.Equal(embeddings.QueryVector, store.SearchVector);
        Assert.Equal(7, store.TopK);
        Assert.Equal(new[] { (nameof(IVectorStore.InitializeAsync), cancellation.Token),
            (nameof(IVectorStore.SearchAsync), cancellation.Token) }, store.Calls);
        Assert.Equal(hit, Assert.Single(results));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData(" \t\n")]
    public async Task Search_BlankQuery_RejectsBeforeCallingDependencies(string? query) {
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore();
        var service = new SemanticSearchService(embeddings, store);

        await Assert.ThrowsAnyAsync<ArgumentException>(() => service.SearchAsync(query!, 1));
        Assert.Null(embeddings.Query);
        Assert.Empty(store.Calls);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(101)]
    public async Task Search_TopKOutsideRange_RejectsBeforeCallingDependencies(int topK) {
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore();
        var service = new SemanticSearchService(embeddings, store);

        await Assert.ThrowsAsync<ArgumentOutOfRangeException>("topK", () => service.SearchAsync("query", topK));
        Assert.Null(embeddings.Query);
        Assert.Empty(store.Calls);
    }
}
