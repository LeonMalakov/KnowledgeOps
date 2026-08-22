using KnowledgeOps.Application;

namespace KnowledgeOps.Tests;

public class RagAnswerServiceTests {
    [Fact]
    public async Task Answer_UsesOneSearchAndPassesNumberedSourcesWithOriginalMetadata() {
        using var cancellation = new CancellationTokenSource();
        var embeddings = new FakeTextEmbeddingService();
        var first = new ChunkSearchResult(Guid.NewGuid(), Guid.NewGuid(), 2, "Доставка", 10, 18, 0.91f, "Доставка заказа", "help:delivery");
        var second = new ChunkSearchResult(Guid.NewGuid(), Guid.NewGuid(), 7, "Три дня", 30, 37, 0.86f);
        var store = new FakeVectorStore { SearchResults = [first, second] };
        var generator = new FakeAnswerGenerator("Доставка займёт три дня [2].");
        var service = new RagAnswerService(new SemanticSearchService(embeddings, store), generator, 100);
        const string question = "  Сколько ждать? ";

        var result = await service.AnswerAsync(question, 3, cancellation.Token);

        Assert.Equal(1, embeddings.QueryCallCount);
        Assert.Equal(question, embeddings.Query);
        Assert.Equal(cancellation.Token, embeddings.ReceivedToken);
        Assert.Equal(3, store.TopK);
        Assert.Equal(new[] {
            (nameof(IVectorStore.InitializeAsync), cancellation.Token),
            (nameof(IVectorStore.SearchAsync), cancellation.Token)
        }, store.Calls);
        Assert.Equal(1, generator.CallCount);
        Assert.Equal(question, generator.Question);
        Assert.Equal(cancellation.Token, generator.Token);
        Assert.Equal("Доставка займёт три дня [2].", result.Answer);
        Assert.Equal(generator.Sources, result.Sources);
        Assert.Collection(result.Sources,
            source => Assert.Equal(new RagSource(1, first.DocumentId, 2, "Доставка", 10, 18, 0.91f, "Доставка заказа", "help:delivery"), source),
            source => Assert.Equal(new RagSource(2, second.DocumentId, 7, "Три дня", 30, 37, 0.86f), source));
    }

    [Fact]
    public async Task Answer_ContextBudget_SelectsWholeChunksInSearchOrderAndRenumbersThem() {
        var store = new FakeVectorStore {
            SearchResults = [
                new(Guid.NewGuid(), Guid.NewGuid(), 0, "12345", 0, 5, 0.9f),
                new(Guid.NewGuid(), Guid.NewGuid(), 1, "longer", 10, 16, 0.8f),
                new(Guid.NewGuid(), Guid.NewGuid(), 2, "🙂a", 20, 23, 0.7f),
                new(Guid.NewGuid(), Guid.NewGuid(), 3, "x", 30, 31, 0.6f)
            ]
        };
        var generator = new FakeAnswerGenerator("Answer");
        var service = new RagAnswerService(new SemanticSearchService(new FakeTextEmbeddingService(), store), generator, 8);

        var result = await service.AnswerAsync("Question", 4);

        Assert.Equal(generator.Sources, result.Sources);
        Assert.Equal(8, result.Sources.Sum(source => source.Content.Length));
        Assert.Collection(result.Sources,
            source => Assert.Equal((1, 0, "12345", 0, 5), (source.Number, source.ChunkIndex, source.Content, source.StartOffset, source.EndOffset)),
            source => Assert.Equal((2, 2, "🙂a", 20, 23), (source.Number, source.ChunkIndex, source.Content, source.StartOffset, source.EndOffset)));
    }

    [Fact]
    public async Task Answer_ReranksBeforeTopKAndContextBudget_PreservingScoresAndCancellationToken() {
        using var cancellation = new CancellationTokenSource();
        var embeddings = new FakeTextEmbeddingService();
        var last = new ChunkSearchResult(Guid.NewGuid(), Guid.NewGuid(), 0, "x", 0, 1, 0.99f);
        var first = new ChunkSearchResult(Guid.NewGuid(), Guid.NewGuid(), 1, "AB", 10, 12, 0.90f, "Title", "help:example");
        var tooLong = new ChunkSearchResult(Guid.NewGuid(), Guid.NewGuid(), 2, "Too long", 20, 28, 0.85f);
        var second = new ChunkSearchResult(Guid.NewGuid(), Guid.NewGuid(), 3, "CD", 30, 32, 0.75f);
        var store = new FakeVectorStore { SearchResults = [last, first, tooLong, second] };
        var reranker = new FakeTextReranker { Scores = [new(0, -1), new(1, 9), new(2, 7), new(3, 6)] };
        var search = new SemanticSearchService(embeddings, store);
        var generator = new FakeAnswerGenerator("Answer");
        var service = new RagAnswerService(search, generator, 5, new RerankedSearchService(search, reranker), candidateK: 25);

        var result = await service.AnswerAsync("Question", 3, cancellation.Token);

        Assert.Equal(25, store.TopK);
        Assert.Equal(1, embeddings.QueryCallCount);
        Assert.Equal(1, reranker.CallCount);
        Assert.Equal("Question", reranker.Query);
        Assert.Equal(new[] { "x", "AB", "Too long", "CD" }, reranker.Texts);
        Assert.Equal(cancellation.Token, embeddings.ReceivedToken);
        Assert.All(store.Calls, call => Assert.Equal(cancellation.Token, call.Token));
        Assert.Equal(cancellation.Token, reranker.Token);
        Assert.Equal(cancellation.Token, generator.Token);
        Assert.Equal(generator.Sources, result.Sources);
        Assert.Collection(result.Sources,
            source => Assert.Equal(new RagSource(1, first.DocumentId, 1, "AB", 10, 12, 0.90f, "Title", "help:example", 9), source),
            source => Assert.Equal(new RagSource(2, second.DocumentId, 3, "CD", 30, 32, 0.75f, RerankScore: 6), source));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Answer_NoUsableSources_ReturnsFallbackWithoutCallingGenerator(bool hasOversizedSource) {
        var store = new FakeVectorStore {
            SearchResults = hasOversizedSource ? [new(Guid.NewGuid(), Guid.NewGuid(), 0, "Too long", 0, 8, 0.9f)] : []
        };
        var generator = new FakeAnswerGenerator("Unused");
        var service = new RagAnswerService(new SemanticSearchService(new FakeTextEmbeddingService(), store), generator, 4);

        var result = await service.AnswerAsync("Вопрос", 1);

        Assert.Contains("недостаточно информации", result.Answer);
        Assert.Empty(result.Sources);
        Assert.Equal(0, generator.CallCount);
    }

    [Fact]
    public async Task Answer_BlankModelResponse_ReturnsFallbackAndPreservesContext() {
        var store = new FakeVectorStore {
            SearchResults = [new(Guid.NewGuid(), Guid.NewGuid(), 0, "Данные", 0, 6, 0.9f)]
        };
        var generator = new FakeAnswerGenerator(" \n");
        var service = new RagAnswerService(new SemanticSearchService(new FakeTextEmbeddingService(), store), generator, 100);

        var result = await service.AnswerAsync("Вопрос", 1);

        Assert.Equal("Модель не вернула текст ответа.", result.Answer);
        Assert.Single(result.Sources);
        Assert.Equal(generator.Sources, result.Sources);
    }

    private sealed class FakeAnswerGenerator(string answer) : IAnswerGenerator {
        public int CallCount { get; private set; }
        public string? Question { get; private set; }
        public IReadOnlyList<RagSource> Sources { get; private set; } = [];
        public CancellationToken Token { get; private set; }

        public Task<string> GenerateAsync(
            string question, IReadOnlyList<RagSource> sources, CancellationToken cancellationToken = default) {
            CallCount++;
            Question = question;
            Sources = sources.ToArray();
            Token = cancellationToken;
            return Task.FromResult(answer);
        }
    }
}
