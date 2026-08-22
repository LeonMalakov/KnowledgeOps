using System.Net;
using System.Net.Http.Json;
using KnowledgeOps.Application;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace KnowledgeOps.Tests;

public class RerankingEndpointsTests(WebApplicationFactory<Program> factory) : IClassFixture<WebApplicationFactory<Program>> {
    [Fact]
    public async Task PostRerankedSearch_ReturnsMetadataAndSeparateScoresWithDefaultCandidateK() {
        var first = Hit("First", 0.9f);
        var second = Hit("Second", 0.7f);
        var store = new FakeVectorStore { SearchResults = [first, second] };
        var reranker = new FakeTextReranker { Scores = [new(0, -2), new(1, 5)] };
        using var app = WithFakes(store, reranker);
        using var client = app.CreateClient();

        using var response = await client.PostAsJsonAsync("/api/search/reranked", new { query = "Доставка", topK = 1 });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var hits = await response.Content.ReadFromJsonAsync<RerankedChunkSearchResult[]>();
        Assert.Equal(new RerankedChunkSearchResult(second.PointId, second.DocumentId, 0, "Second", 0, 6,
            0.7f, 5f, "Guide", "help:guide"), Assert.Single(hits!));
        Assert.Equal(20, store.TopK);
        Assert.Equal("Доставка", reranker.Query);
    }

    [Theory]
    [InlineData(" ", 3, 20)]
    [InlineData("query", 11, 20)]
    [InlineData("query", 3, 2)]
    [InlineData("query", 3, 51)]
    public async Task PostRerankedSearch_InvalidInputReturnsBadRequest(string query, int topK, int candidateK) {
        var store = new FakeVectorStore();
        var reranker = new FakeTextReranker();
        using var app = WithFakes(store, reranker);
        using var client = app.CreateClient();

        using var response = await client.PostAsJsonAsync("/api/search/reranked", new { query, topK, candidateK });

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Empty(store.Calls);
        Assert.Equal(0, reranker.CallCount);
    }

    [Theory]
    [InlineData(TextRerankerFailure.Unavailable, HttpStatusCode.ServiceUnavailable)]
    [InlineData(TextRerankerFailure.Timeout, HttpStatusCode.GatewayTimeout)]
    [InlineData(TextRerankerFailure.UpstreamError, HttpStatusCode.BadGateway)]
    public async Task PostRerankedSearch_ProviderFailureDoesNotAffectDenseSearch(
        TextRerankerFailure failure, HttpStatusCode expected) {
        var hit = Hit("First", 0.9f);
        var store = new FakeVectorStore { SearchResults = [hit] };
        var reranker = new FakeTextReranker { Failure = new TextRerankerException(failure, "TEI test failure.") };
        using var app = WithFakes(store, reranker);
        using var client = app.CreateClient();

        using var response = await client.PostAsJsonAsync("/api/search/reranked", new { query = "query", topK = 1, candidateK = 5 });

        Assert.Equal(expected, response.StatusCode);
        Assert.Equal(5, store.TopK);
        Assert.Equal("TEI test failure.", (await response.Content.ReadFromJsonAsync<ProblemDetails>())?.Title);
        using var dense = await client.PostAsJsonAsync("/api/search", new { query = "query", topK = 1 });
        Assert.Equal(HttpStatusCode.OK, dense.StatusCode);
        Assert.Equal(hit, Assert.Single((await dense.Content.ReadFromJsonAsync<ChunkSearchResult[]>())!));
        Assert.Equal(1, reranker.CallCount);
    }

    private WebApplicationFactory<Program> WithFakes(FakeVectorStore store, FakeTextReranker reranker) {
        return factory.WithWebHostBuilder(builder => builder.ConfigureTestServices(services => {
            services.RemoveAll<ITextEmbeddingService>();
            services.RemoveAll<IVectorStore>();
            services.RemoveAll<ITextReranker>();
            services.AddSingleton<ITextEmbeddingService>(new FakeTextEmbeddingService());
            services.AddSingleton<IVectorStore>(store);
            services.AddSingleton<ITextReranker>(reranker);
        }));
    }

    private static ChunkSearchResult Hit(string content, float retrievalScore) =>
        new(Guid.NewGuid(), Guid.NewGuid(), 0, content, 0, content.Length, retrievalScore, "Guide", "help:guide");
}
