using KnowledgeOps.Application;

namespace KnowledgeOps.Tests;

public class RerankedSearchServiceTests {
    [Fact]
    public async Task Search_ReordersByIndexedScoresAndKeepsSeparateScoresAndMetadata() {
        using var cancellation = new CancellationTokenSource();
        var first = Hit("First", retrievalScore: 0.9f);
        var second = Hit("Second", retrievalScore: 0.7f) with { DocumentId = first.DocumentId, ChunkIndex = 1 };
        var third = Hit("Third", retrievalScore: 0.6f);
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore { SearchResults = [first, second, third] };
        var reranker = new FakeTextReranker { Scores = [new(2, -2f), new(0, 1f), new(1, 3f)] };
        var service = CreateService(embeddings, store, reranker);

        var hits = await service.SearchAsync("original query", 2, cancellationToken: cancellation.Token);

        Assert.Equal(1, embeddings.QueryCallCount);
        Assert.Equal(RerankedSearchService.DEFAULT_CANDIDATE_K, store.TopK);
        Assert.Equal(1, reranker.CallCount);
        Assert.Equal("original query", reranker.Query);
        Assert.Equal(new[] { "First", "Second", "Third" }, reranker.Texts);
        Assert.Equal(cancellation.Token, embeddings.ReceivedToken);
        Assert.All(store.Calls, call => Assert.Equal(cancellation.Token, call.Token));
        Assert.Equal(cancellation.Token, reranker.Token);
        Assert.Collection(hits,
            hit => Assert.Equal(new RerankedChunkSearchResult(second.PointId, second.DocumentId, 1, "Second", 3, 9,
                0.7f, 3f, "Guide", "help:guide"), hit),
            hit => Assert.Equal(new RerankedChunkSearchResult(first.PointId, first.DocumentId, 0, "First", 3, 8,
                0.9f, 1f, "Guide", "help:guide"), hit));
    }

    [Fact]
    public async Task Search_TiedScoresKeepDenseOrder() {
        var first = Hit("First");
        var second = Hit("Second");
        var service = CreateService(new FakeTextEmbeddingService(), new FakeVectorStore { SearchResults = [first, second] },
            new FakeTextReranker { Scores = [new(1, 4f), new(0, 4f)] });

        var hits = await service.SearchAsync("query", 2, 5);

        Assert.Equal(new[] { first.PointId, second.PointId }, hits.Select(hit => hit.PointId));
    }

    [Theory]
    [InlineData(" ", 1, 20)]
    [InlineData("query", 0, 20)]
    [InlineData("query", 11, 20)]
    [InlineData("query", 5, 4)]
    [InlineData("query", 5, 51)]
    public async Task Search_InvalidArgumentsDoNotCallDependencies(string query, int topK, int candidateK) {
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore();
        var reranker = new FakeTextReranker();

        await Assert.ThrowsAnyAsync<ArgumentException>(() => CreateService(embeddings, store, reranker).SearchAsync(query, topK, candidateK));

        Assert.Equal(0, embeddings.QueryCallCount);
        Assert.Empty(store.Calls);
        Assert.Equal(0, reranker.CallCount);
    }

    [Fact]
    public async Task Search_NoCandidatesSkipsReranker() {
        var reranker = new FakeTextReranker();
        var service = CreateService(new FakeTextEmbeddingService(), new FakeVectorStore(), reranker);

        Assert.Empty(await service.SearchAsync("query", 3));
        Assert.Equal(0, reranker.CallCount);
    }

    private static ChunkSearchResult Hit(string content, float retrievalScore = 0.8f) =>
        new(Guid.NewGuid(), Guid.NewGuid(), 0, content, 3, 3 + content.Length, retrievalScore, "Guide", "help:guide");

    private static RerankedSearchService CreateService(FakeTextEmbeddingService embeddings, FakeVectorStore store, FakeTextReranker reranker) =>
        new(new SemanticSearchService(embeddings, store), reranker);
}

internal sealed class FakeTextReranker : ITextReranker {
    public IReadOnlyList<TextRerankScore> Scores { get; init; } = [];
    public TextRerankerException? Failure { get; init; }
    public string? Query { get; private set; }
    public IReadOnlyList<string>? Texts { get; private set; }
    public int CallCount { get; private set; }
    public CancellationToken Token { get; private set; }

    public Task<IReadOnlyList<TextRerankScore>> RerankAsync(
        string query, IReadOnlyList<string> texts, CancellationToken cancellationToken = default) {
        CallCount++;
        Query = query;
        Texts = texts.ToArray();
        Token = cancellationToken;
        return Failure is null ? Task.FromResult(Scores) : Task.FromException<IReadOnlyList<TextRerankScore>>(Failure);
    }
}
