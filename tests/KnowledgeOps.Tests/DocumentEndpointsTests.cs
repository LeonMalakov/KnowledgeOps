using System.Net;
using System.Net.Http.Json;
using System.Text;
using KnowledgeOps.Application;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace KnowledgeOps.Tests;

public class DocumentEndpointsTests(WebApplicationFactory<Program> factory) : IClassFixture<WebApplicationFactory<Program>> {
    [Theory]
    [InlineData(4, 1)]
    [InlineData(50_000, 59)]
    public async Task PostDocuments_IndexesTextAndReturnsDocumentIdAndChunkCount(int length, int expectedCount) {
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore();
        using var app = WithFakes(embeddings, store);
        using var client = app.CreateClient();

        string title = new string('t', 200);
        string source = new string('s', 1000);
        using var response = await client.PostAsJsonAsync("/api/documents", new { title, content = new string('x', length), source });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var result = await response.Content.ReadFromJsonAsync<DocumentIndexingResult>();
        Assert.NotNull(result);
        Assert.NotEqual(Guid.Empty, result.DocumentId);
        Assert.Equal(expectedCount, result.ChunkCount);
        Assert.Equal(1, embeddings.DocumentCallCount);
        Assert.Equal(expectedCount, store.Points!.Count);
        Assert.All(store.Points, point => {
            Assert.Equal(result.DocumentId, point.DocumentId);
            Assert.Equal(title, point.Title);
            Assert.Equal(source, point.Source);
        });
    }

    [Theory]
    [InlineData("{}")]
    [InlineData("""{"title":"Guide","content":null}""")]
    [InlineData("""{"title":"Guide","content":""}""")]
    [InlineData("""{"title":"Guide","content":" \t\n"}""")]
    [InlineData("""{"content":"Text"}""")]
    [InlineData("""{"title":" \t","content":"Text"}""")]
    [InlineData("{")]
    public async Task PostDocuments_InvalidBody_ReturnsBadRequest(string body) {
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore();
        using var app = WithFakes(embeddings, store);
        using var client = app.CreateClient();
        using var json = new StringContent(body, Encoding.UTF8, "application/json");

        using var response = await client.PostAsync("/api/documents", json);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Equal(0, embeddings.DocumentCallCount);
        Assert.Empty(store.Calls);
    }

    [Fact]
    public async Task PostDocuments_TooLong_ReturnsBadRequest() {
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore();
        using var app = WithFakes(embeddings, store);
        using var client = app.CreateClient();

        using var response = await client.PostAsJsonAsync("/api/documents", new { title = "Guide", content = new string('x', 50_001) });

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Equal(0, embeddings.DocumentCallCount);
        Assert.Empty(store.Calls);
    }

    [Theory]
    [InlineData(201, 1000, "title")]
    [InlineData(200, 1001, "source")]
    public async Task PostDocuments_OversizedMetadata_ReturnsBadRequest(int titleLength, int sourceLength, string field) {
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore();
        using var app = WithFakes(embeddings, store);
        using var client = app.CreateClient();

        using var response = await client.PostAsJsonAsync("/api/documents",
            new { title = new string('t', titleLength), content = "Text", source = new string('s', sourceLength) });

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var problem = await response.Content.ReadFromJsonAsync<Microsoft.AspNetCore.Mvc.ValidationProblemDetails>();
        Assert.NotNull(problem);
        Assert.Contains(field, problem.Errors.Keys);
        Assert.Equal(0, embeddings.DocumentCallCount);
        Assert.Empty(store.Calls);
    }

    [Fact]
    public async Task PutDocuments_ReindexesAndReturnsTheSameDocumentId() {
        var embeddings = new FakeTextEmbeddingService();
        Guid[] oldIds = [Guid.NewGuid(), Guid.NewGuid()];
        var store = new FakeVectorStore { ExistingPointIds = oldIds };
        using var app = WithFakes(embeddings, store);
        using var client = app.CreateClient();
        var documentId = Guid.NewGuid();

        using var response = await client.PutAsJsonAsync($"/api/documents/{documentId}",
            new { title = "Новые правила доставки", content = "Доставка занимает пять рабочих дней.", source = "help:delivery" });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal(new DocumentIndexingResult(documentId, 1), await response.Content.ReadFromJsonAsync<DocumentIndexingResult>());
        Assert.Equal(documentId, store.RequestedDocumentId);
        Assert.Equal(oldIds, store.DeletedPointIds);
        var point = Assert.Single(store.Points!);
        Assert.Equal(documentId, point.DocumentId);
        Assert.Equal("Новые правила доставки", point.Title);
        Assert.Equal("help:delivery", point.Source);
        Assert.Equal("Доставка занимает пять рабочих дней.", point.Content);
    }

    [Fact]
    public async Task PutDocuments_MissingDocument_ReturnsNotFoundWithoutEmbedding() {
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore();
        using var app = WithFakes(embeddings, store);
        using var client = app.CreateClient();

        using var response = await client.PutAsJsonAsync($"/api/documents/{Guid.NewGuid()}", new { title = "Guide", content = "Text" });

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
        Assert.Equal(0, embeddings.DocumentCallCount);
        Assert.Null(store.Points);
        Assert.Null(store.DeletedPointIds);
    }

    [Theory]
    [InlineData("00000000-0000-0000-0000-000000000000", "Guide", "Text", HttpStatusCode.BadRequest)]
    [InlineData("invalid", "Guide", "Text", HttpStatusCode.NotFound)]
    [InlineData("46a0194d-34e8-4b01-9d23-0feb0dc144bc", " ", "Text", HttpStatusCode.BadRequest)]
    [InlineData("46a0194d-34e8-4b01-9d23-0feb0dc144bc", "Guide", "", HttpStatusCode.BadRequest)]
    public async Task PutDocuments_InvalidRequest_DoesNotCallDependencies(
        string documentId, string title, string content, HttpStatusCode expectedStatus) {
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore();
        using var app = WithFakes(embeddings, store);
        using var client = app.CreateClient();

        using var response = await client.PutAsJsonAsync($"/api/documents/{documentId}", new { title, content });

        Assert.Equal(expectedStatus, response.StatusCode);
        Assert.Equal(0, embeddings.DocumentCallCount);
        Assert.Empty(store.Calls);
    }

    [Fact]
    public async Task PutDocuments_UnconfirmedCleanup_ReturnsProblemAndLeavesNewPoints() {
        Guid[] oldIds = [Guid.NewGuid()];
        var store = new FakeVectorStore {
            ExistingPointIds = oldIds,
            DeletePointsError = new InvalidOperationException("Deletion response was lost.")
        };
        using var app = WithFakes(new FakeTextEmbeddingService(), store);
        using var client = app.CreateClient();
        var documentId = Guid.NewGuid();

        using var response = await client.PutAsJsonAsync($"/api/documents/{documentId}", new { title = "Guide", content = "Text" });

        Assert.Equal(HttpStatusCode.InternalServerError, response.StatusCode);
        var problem = await response.Content.ReadFromJsonAsync<Microsoft.AspNetCore.Mvc.ProblemDetails>();
        Assert.Equal("Document update could not be confirmed.", problem?.Title);
        Assert.Contains(documentId.ToString(), problem?.Detail);
        Assert.Equal(oldIds, store.DeletedPointIds);
        Assert.Equal(documentId, Assert.Single(store.Points!).DocumentId);
        Assert.Empty(store.DeletedDocumentIds);
    }

    [Fact]
    public async Task DeleteDocuments_ReturnsNoContentAndPassesDocumentIdOnEveryRequest() {
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore();
        using var app = WithFakes(embeddings, store);
        using var client = app.CreateClient();
        var documentId = Guid.NewGuid();

        using var response = await client.DeleteAsync($"/api/documents/{documentId}");
        using var repeated = await client.DeleteAsync($"/api/documents/{documentId}");

        Assert.Equal(HttpStatusCode.NoContent, response.StatusCode);
        Assert.Equal(HttpStatusCode.NoContent, repeated.StatusCode);
        Assert.Equal(new[] { documentId, documentId }, store.DeletedDocumentIds);
        Assert.Equal(0, embeddings.DocumentCallCount);
    }

    [Theory]
    [InlineData("00000000-0000-0000-0000-000000000000", HttpStatusCode.BadRequest)]
    [InlineData("invalid", HttpStatusCode.NotFound)]
    public async Task DeleteDocuments_InvalidId_DoesNotCallStore(string documentId, HttpStatusCode expectedStatus) {
        var store = new FakeVectorStore();
        using var app = WithFakes(new FakeTextEmbeddingService(), store);
        using var client = app.CreateClient();

        using var response = await client.DeleteAsync($"/api/documents/{documentId}");

        Assert.Equal(expectedStatus, response.StatusCode);
        Assert.Empty(store.Calls);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(100)]
    public async Task PostSearch_ReturnsMatchingChunks(int topK) {
        var hit = new ChunkSearchResult(Guid.NewGuid(), Guid.NewGuid(), 0, "Delivery", 0, 8, 0.95f, "Delivery guide", "help:delivery");
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore { SearchResults = [hit] };
        using var app = WithFakes(embeddings, store);
        using var client = app.CreateClient();

        using var response = await client.PostAsJsonAsync("/api/search", new { query = "When will it arrive?", topK });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var results = await response.Content.ReadFromJsonAsync<ChunkSearchResult[]>();
        Assert.Equal(hit, Assert.Single(results!));
        Assert.Equal("When will it arrive?", embeddings.Query);
        Assert.Equal(topK, store.TopK);
        Assert.Equal(embeddings.QueryVector, store.SearchVector);
        Assert.Contains(store.Calls, call => call.Operation == nameof(IVectorStore.SearchAsync));
        Assert.Null(store.HybridQuery);
    }

    [Fact]
    public async Task PostHybridSearch_ReturnsHybridHitsAndForwardsQuery() {
        var hit = new ChunkSearchResult(Guid.NewGuid(), Guid.NewGuid(), 0, "SEC-240", 0, 7, 0.8f, "Security", "help:sec240");
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore { HybridSearchResults = [hit] };
        using var app = WithFakes(embeddings, store);
        using var client = app.CreateClient();

        using var response = await client.PostAsJsonAsync("/api/search/hybrid", new { query = "Что значит SEC-240?", topK = 5 });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var results = await response.Content.ReadFromJsonAsync<ChunkSearchResult[]>();
        Assert.Equal(hit, Assert.Single(results!));
        Assert.Equal("Что значит SEC-240?", store.HybridQuery);
        Assert.Equal(embeddings.Query, store.HybridQuery);
        Assert.Equal(1, embeddings.QueryCallCount);
        Assert.Equal(5, store.TopK);
        Assert.DoesNotContain(store.Calls, call => call.Operation == nameof(IVectorStore.SearchAsync));
    }

    [Theory]
    [InlineData(" ", 5)]
    [InlineData("SEC-240", 0)]
    [InlineData("SEC-240", 101)]
    public async Task PostHybridSearch_InvalidInputReturnsBadRequestWithoutRetrieval(string query, int topK) {
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore();
        using var app = WithFakes(embeddings, store);
        using var client = app.CreateClient();

        using var response = await client.PostAsJsonAsync("/api/search/hybrid", new { query, topK });

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Equal(0, embeddings.QueryCallCount);
        Assert.Empty(store.Calls);
    }

    [Theory]
    [InlineData("""{"topK":5}""")]
    [InlineData("""{"query":"","topK":5}""")]
    [InlineData("""{"query":"  ","topK":5}""")]
    [InlineData("""{"query":"query"}""")]
    [InlineData("""{"query":"query","topK":0}""")]
    [InlineData("""{"query":"query","topK":101}""")]
    public async Task PostSearch_InvalidBody_ReturnsBadRequest(string body) {
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore();
        using var app = WithFakes(embeddings, store);
        using var client = app.CreateClient();
        using var json = new StringContent(body, Encoding.UTF8, "application/json");

        using var response = await client.PostAsync("/api/search", json);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Null(embeddings.Query);
        Assert.Empty(store.Calls);
    }

    private WebApplicationFactory<Program> WithFakes(FakeTextEmbeddingService embeddings, FakeVectorStore store) {
        return factory.WithWebHostBuilder(builder => builder.ConfigureTestServices(services => {
            services.RemoveAll<ITextEmbeddingService>();
            services.RemoveAll<IVectorStore>();
            services.AddSingleton<ITextEmbeddingService>(embeddings);
            services.AddSingleton<IVectorStore>(store);
        }));
    }
}
