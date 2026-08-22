using System.Net;
using System.Net.Http.Json;
using System.Text;
using KnowledgeOps.Application;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Options;

namespace KnowledgeOps.Tests;

public class RagEndpointsTests(WebApplicationFactory<Program> factory) : IClassFixture<WebApplicationFactory<Program>> {
    [Theory]
    [InlineData(1, "Три дня [1].", "Три дня [1].")]
    [InlineData(10, " \n", "Модель не вернула текст ответа.")]
    public async Task PostAsk_ReturnsAnswerAndSources(int topK, string modelAnswer, string expectedAnswer) {
        var embeddings = new FakeTextEmbeddingService();
        var hit = new ChunkSearchResult(Guid.NewGuid(), Guid.NewGuid(), 2, "Три дня", 10, 17, 0.9f,
            "Сроки доставки", "https://example.test/delivery");
        var store = new FakeVectorStore { SearchResults = [hit] };
        using var chat = new FakeChatClient(modelAnswer);
        var reranker = new FakeTextReranker { Failure = new TextRerankerException(TextRerankerFailure.Unavailable, "TEI test failure.") };
        using var app = WithFakes(embeddings, store, chat, reranker, retrievalMode: "dense");
        using var client = app.CreateClient();

        using var response = await client.PostAsJsonAsync("/api/ask", new { query = "Когда доставят?", topK });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var result = await response.Content.ReadFromJsonAsync<RagAnswerResult>();
        Assert.NotNull(result);
        Assert.Equal(expectedAnswer, result.Answer);
        Assert.Equal(new RagSource(1, hit.DocumentId, 2, "Три дня", 10, 17, 0.9f, "Сроки доставки", "https://example.test/delivery"),
            Assert.Single(result.Sources));
        Assert.Equal(topK, store.TopK);
        Assert.Equal(1, embeddings.QueryCallCount);
        Assert.Equal(1, chat.CallCount);
        Assert.Equal(0, reranker.CallCount);
        Assert.Contains(store.Calls, call => call.Operation == nameof(IVectorStore.SearchAsync));
        Assert.DoesNotContain(store.Calls, call => call.Operation == nameof(IVectorStore.SearchHybridAsync));
    }

    [Theory]
    [InlineData(null, 20)]
    [InlineData("reranked", 50)]
    public async Task PostAsk_ReranksUsingConfiguredCandidatesAndReturnsOnlyTopK(string? retrievalMode, int candidateK) {
        var first = new ChunkSearchResult(Guid.NewGuid(), Guid.NewGuid(), 0, "First", 0, 5, 0.9f);
        var second = new ChunkSearchResult(Guid.NewGuid(), Guid.NewGuid(), 2, "Second", 10, 16, 0.7f, "Guide", "help:guide");
        var store = new FakeVectorStore { SearchResults = [first, second] };
        var reranker = new FakeTextReranker { Scores = [new(0, 1), new(1, 3)] };
        using var chat = new FakeChatClient("Answer [1].");
        using var app = WithFakes(new FakeTextEmbeddingService(), store, chat, reranker, retrievalMode: retrievalMode,
            candidateK: retrievalMode is null ? null : candidateK.ToString(System.Globalization.CultureInfo.InvariantCulture));
        using var client = app.CreateClient();

        using var response = await client.PostAsJsonAsync("/api/ask", new { query = "Question", topK = 1 });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var result = await response.Content.ReadFromJsonAsync<RagAnswerResult>();
        Assert.NotNull(result);
        Assert.Equal("Answer [1].", result.Answer);
        Assert.Equal(new RagSource(1, second.DocumentId, 2, "Second", 10, 16, 0.7f, "Guide", "help:guide", 3),
            Assert.Single(result.Sources));
        Assert.Equal(candidateK, store.TopK);
        Assert.Equal(1, reranker.CallCount);
        Assert.Equal(1, chat.CallCount);
    }

    [Theory]
    [InlineData(TextRerankerFailure.Unavailable, HttpStatusCode.ServiceUnavailable)]
    [InlineData(TextRerankerFailure.Timeout, HttpStatusCode.GatewayTimeout)]
    [InlineData(TextRerankerFailure.UpstreamError, HttpStatusCode.BadGateway)]
    public async Task PostAsk_TeiFailureReturnsProblemWithoutCallingChat(
        TextRerankerFailure failure, HttpStatusCode expectedStatus) {
        var hit = new ChunkSearchResult(Guid.NewGuid(), Guid.NewGuid(), 0, "Text", 0, 4, 0.9f);
        var store = new FakeVectorStore { SearchResults = [hit] };
        var reranker = new FakeTextReranker { Failure = new TextRerankerException(failure, "TEI test failure.") };
        using var chat = new FakeChatClient("Unused");
        using var app = WithFakes(new FakeTextEmbeddingService(), store, chat, reranker);
        using var client = app.CreateClient();
        var request = new { query = "Question", topK = 1 };

        using var response = await client.PostAsJsonAsync("/api/ask", request);

        Assert.Equal(expectedStatus, response.StatusCode);
        Assert.Equal("TEI test failure.", (await response.Content.ReadFromJsonAsync<ProblemDetails>())?.Title);
        Assert.Equal(0, chat.CallCount);
        using var denseResponse = await client.PostAsJsonAsync("/api/search", request);
        using var hybridResponse = await client.PostAsJsonAsync("/api/search/hybrid", request);
        Assert.Equal(HttpStatusCode.OK, denseResponse.StatusCode);
        Assert.Equal(HttpStatusCode.OK, hybridResponse.StatusCode);
        Assert.Equal(1, reranker.CallCount);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task PostAsk_NoUsableSources_ReturnsFallbackWithoutChat(bool hasOversizedSource) {
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore {
            SearchResults = hasOversizedSource ? [new(Guid.NewGuid(), Guid.NewGuid(), 0, "Too long", 0, 8, 0.9f)] : []
        };
        var reranker = new FakeTextReranker { Scores = [new(0, 2)] };
        using var chat = new FakeChatClient("Unused");
        using var app = WithFakes(embeddings, store, chat, reranker, maxContextCharacters: "4");
        using var client = app.CreateClient();

        using var response = await client.PostAsJsonAsync("/api/ask", new { query = new string('x', 2_000), topK = 3 });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var result = await response.Content.ReadFromJsonAsync<RagAnswerResult>();
        Assert.NotNull(result);
        Assert.Contains("недостаточно информации", result.Answer);
        Assert.Empty(result.Sources);
        Assert.Equal(0, chat.CallCount);
        Assert.Equal(hasOversizedSource ? 1 : 0, reranker.CallCount);
    }

    [Theory]
    [InlineData("""{"topK":3}""")]
    [InlineData("""{"query":"","topK":3}""")]
    [InlineData("""{"query":" \t","topK":3}""")]
    [InlineData("""{"query":"question"}""")]
    [InlineData("""{"query":"question","topK":0}""")]
    [InlineData("""{"query":"question","topK":11}""")]
    [InlineData("{")]
    public async Task PostAsk_InvalidBody_ReturnsBadRequest(string body) {
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore();
        using var chat = new FakeChatClient("Unused");
        using var app = WithFakes(embeddings, store, chat);
        using var client = app.CreateClient();
        using var json = new StringContent(body, Encoding.UTF8, "application/json");

        using var response = await client.PostAsync("/api/ask", json);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Equal(0, embeddings.QueryCallCount);
        Assert.Empty(store.Calls);
        Assert.Equal(0, chat.CallCount);
    }

    [Fact]
    public async Task PostAsk_QueryTooLong_ReturnsBadRequest() {
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore();
        using var chat = new FakeChatClient("Unused");
        using var app = WithFakes(embeddings, store, chat);
        using var client = app.CreateClient();

        using var response = await client.PostAsJsonAsync("/api/ask", new { query = new string('x', 2_001), topK = 3 });

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Equal(0, embeddings.QueryCallCount);
        Assert.Empty(store.Calls);
        Assert.Equal(0, chat.CallCount);
    }

    [Theory]
    [InlineData("hybrid", "20", "RetrievalMode")]
    [InlineData("dense", "0", "CandidateK")]
    [InlineData("reranked", "9", "CandidateK")]
    [InlineData("reranked", "51", "CandidateK")]
    public void AppStart_RejectsInvalidRagConfiguration(string retrievalMode, string candidateK, string invalidSetting) {
        using var chat = new FakeChatClient("Unused");
        using var app = WithFakes(new FakeTextEmbeddingService(), new FakeVectorStore(), chat,
            retrievalMode: retrievalMode, candidateK: candidateK);

        var error = Assert.Throws<OptionsValidationException>(() => app.CreateClient());
        Assert.Contains($"Rag:{invalidSetting}", error.Message);
    }

    private WebApplicationFactory<Program> WithFakes(
        FakeTextEmbeddingService embeddings, FakeVectorStore store, FakeChatClient chat,
        FakeTextReranker? reranker = null, string maxContextCharacters = "6000",
        string? retrievalMode = null, string? candidateK = null) {
        return factory.WithWebHostBuilder(builder => {
            builder.ConfigureAppConfiguration((_, configuration) => {
                var settings = new Dictionary<string, string?> { ["Rag:MaxContextCharacters"] = maxContextCharacters };
                if (retrievalMode is not null) {
                    settings["Rag:RetrievalMode"] = retrievalMode;
                }

                if (candidateK is not null) {
                    settings["Rag:CandidateK"] = candidateK;
                }

                configuration.AddInMemoryCollection(settings);
            });
            builder.ConfigureTestServices(services => {
                services.RemoveAll<ITextEmbeddingService>();
                services.RemoveAll<IVectorStore>();
                services.RemoveAll<IChatClient>();
                services.RemoveAll<ITextReranker>();
                services.AddSingleton<ITextEmbeddingService>(embeddings);
                services.AddSingleton<IVectorStore>(store);
                services.AddSingleton<IChatClient>(chat);
                services.AddSingleton<ITextReranker>(reranker ?? new FakeTextReranker());
            });
        });
    }
}
