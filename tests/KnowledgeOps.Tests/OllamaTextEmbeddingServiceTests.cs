using System.Net;
using System.Text;
using System.Text.Json;
using KnowledgeOps.Infrastructure;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Options;
using OllamaSharp;

namespace KnowledgeOps.Tests;

public class OllamaTextEmbeddingServiceTests {
    private const string BASE_URL = "http://localhost:11434";
    private const string MODEL = "embeddinggemma";

    [Fact]
    public async Task EmbedQuery_AddsSearchPrefixAndReturnsOneVector() {
        var generator = new FakeEmbeddingGenerator([2f, 0f, 0f]);
        var service = CreateService(generator);

        var vector = await service.EmbedQueryAsync("  Find this?\n");

        Assert.Equal("task: search result | query:   Find this?\n", Assert.Single(generator.Inputs));
        Assert.Equal(1, generator.CallCount);
        Assert.Equal(new[] { 2f, 0f, 0f }, vector);
    }

    [Fact]
    public async Task EmbedDocuments_AddsDocumentPrefixesInOneBatchAndPreservesOrder() {
        var generator = new FakeEmbeddingGenerator([2f, 0f, 0f], [0f, 3f, 0f]);
        var service = CreateService(generator);
        string[] texts = ["  First\n", "Second"];

        var vectors = await service.EmbedDocumentsAsync(texts);

        Assert.Equal(1, generator.CallCount);
        Assert.Equal(new[] { "title: none | text:   First\n", "title: none | text: Second" }, generator.Inputs);
        Assert.Equal(new[] { "  First\n", "Second" }, texts);
        Assert.Collection(vectors,
            vector => Assert.Equal(new[] { 2f, 0f, 0f }, vector),
            vector => Assert.Equal(new[] { 0f, 3f, 0f }, vector));
    }

    [Fact]
    public async Task EmbedDocuments_EmptyList_DoesNotCallGenerator() {
        var generator = new FakeEmbeddingGenerator();

        Assert.Empty(await CreateService(generator).EmbedDocumentsAsync([]));
        Assert.Equal(0, generator.CallCount);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData(" \t\n")]
    public async Task InvalidText_IsRejectedBeforeCallingGenerator(string? invalid) {
        var generator = new FakeEmbeddingGenerator();
        var service = CreateService(generator);

        await Assert.ThrowsAnyAsync<ArgumentException>(() => service.EmbedQueryAsync(invalid!));
        var exception = await Assert.ThrowsAsync<ArgumentException>("texts",
            () => service.EmbedDocumentsAsync(["valid", invalid!]));
        Assert.Contains("index 1", exception.Message);
        Assert.Equal(0, generator.CallCount);
    }

    [Fact]
    public async Task EmbedDocuments_NullList_IsRejectedBeforeCallingGenerator() {
        var generator = new FakeEmbeddingGenerator();

        await Assert.ThrowsAsync<ArgumentNullException>("texts", () => CreateService(generator).EmbedDocumentsAsync(null!));
        Assert.Equal(0, generator.CallCount);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    public async Task WrongResultCount_IsRejected(int count) {
        var vectors = Enumerable.Repeat(new[] { 1f, 0f, 0f }, count).ToArray();
        var service = CreateService(new FakeEmbeddingGenerator(vectors));

        var exception = await Assert.ThrowsAsync<InvalidOperationException>(() => service.EmbedQueryAsync("query"));

        Assert.Contains($"returned {count} embeddings for 1 inputs", exception.Message);
    }

    [Fact]
    public async Task WrongDimensionInBatch_IsRejected() {
        var service = CreateService(new FakeEmbeddingGenerator([1f, 0f, 0f], [1f, 0f]));

        var exception = await Assert.ThrowsAsync<InvalidOperationException>(
            () => service.EmbedDocumentsAsync(["First", "Second"]));

        Assert.Contains("at index 1 has dimension 2, expected 3", exception.Message);
    }

    [Theory]
    [InlineData(float.NaN, "finite")]
    [InlineData(float.PositiveInfinity, "finite")]
    [InlineData(float.NegativeInfinity, "finite")]
    [InlineData(0f, "non-zero norm")]
    public async Task InvalidVector_IsRejected(float value, string expectedError) {
        var service = CreateService(new FakeEmbeddingGenerator([0f, value, 0f]));

        var exception = await Assert.ThrowsAsync<InvalidOperationException>(() => service.EmbedQueryAsync("query"));

        Assert.Contains(expectedError, exception.Message);
    }

    [Fact]
    public async Task BothOperations_ForwardCancellationToken() {
        using var source = new CancellationTokenSource();
        var generator = new FakeEmbeddingGenerator([1f, 0f, 0f]);
        var service = CreateService(generator);

        await service.EmbedQueryAsync("query", source.Token);
        Assert.Equal(source.Token, generator.CancellationToken);

        await service.EmbedDocumentsAsync(["document"], source.Token);
        Assert.Equal(source.Token, generator.CancellationToken);
    }

    [Fact]
    public async Task CancellationFromGenerator_IsPropagated() {
        using var source = new CancellationTokenSource();
        var generator = new FakeEmbeddingGenerator {
            OnGenerate = token => {
                source.Cancel();
                token.ThrowIfCancellationRequested();
            }
        };

        var exception = await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => CreateService(generator).EmbedQueryAsync("query", source.Token));

        Assert.Equal(source.Token, exception.CancellationToken);
        Assert.Equal(1, generator.CallCount);
    }

    [Fact]
    public async Task UnreachableOllama_ErrorIncludesEndpointAndModel() {
        var failure = new HttpRequestException("Connection refused.");
        var generator = new FakeEmbeddingGenerator { OnGenerate = _ => throw failure };

        var exception = await Assert.ThrowsAsync<InvalidOperationException>(
            () => CreateService(generator).EmbedQueryAsync("query"));

        Assert.Contains(BASE_URL, exception.Message);
        Assert.Contains(MODEL, exception.Message);
        Assert.Same(failure, exception.InnerException);
        Assert.Equal(1, generator.CallCount);
    }

    [Fact]
    public async Task OllamaSharp_SendsOneEmbedRequestWithExactlyOnePrefixPerDocument() {
        using var handler = new StubOllamaHandler(HttpStatusCode.OK, """{"model":"embeddinggemma","embeddings":[[1,0,0],[0,1,0]]}""");
        using var httpClient = new HttpClient(handler) { BaseAddress = new Uri(BASE_URL) };
        using var generator = new OllamaApiClient(httpClient, MODEL);

        await CreateService(generator).EmbedDocumentsAsync(["First", "Second"]);

        Assert.Equal(1, handler.CallCount);
        Assert.Equal(HttpMethod.Post, handler.Method);
        Assert.Equal("/api/embed", handler.Path);
        using var body = JsonDocument.Parse(handler.Body);
        Assert.Equal(MODEL, body.RootElement.GetProperty("model").GetString());
        var inputs = body.RootElement.GetProperty("input").EnumerateArray().Select(input => input.GetString());
        Assert.Equal(new[] { "title: none | text: First", "title: none | text: Second" }, inputs);
    }

    [Fact]
    public async Task MissingModel_ErrorSuggestsPullingIt() {
        using var handler = new StubOllamaHandler(HttpStatusCode.NotFound, """{"error":"model 'embeddinggemma' not found"}""");
        using var httpClient = new HttpClient(handler) { BaseAddress = new Uri(BASE_URL) };
        using var generator = new OllamaApiClient(httpClient, MODEL);

        var exception = await Assert.ThrowsAsync<InvalidOperationException>(
            () => CreateService(generator).EmbedQueryAsync("query"));

        Assert.Contains($"ollama pull {MODEL}", exception.Message);
        Assert.Equal(1, handler.CallCount);
    }

    private static OllamaTextEmbeddingService CreateService(IEmbeddingGenerator<string, Embedding<float>> generator) {
        return new OllamaTextEmbeddingService(generator,
            Options.Create(new EmbeddingOptions { BaseUrl = BASE_URL, Model = MODEL, VectorSize = 3 }));
    }

    private sealed class FakeEmbeddingGenerator(params float[][] vectors) : IEmbeddingGenerator<string, Embedding<float>> {
        public int CallCount { get; private set; }
        public string[] Inputs { get; private set; } = [];
        public CancellationToken CancellationToken { get; private set; }
        public Action<CancellationToken>? OnGenerate { get; init; }

        public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
            IEnumerable<string> values, EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default) {
            CallCount++;
            Inputs = values.ToArray();
            CancellationToken = cancellationToken;
            OnGenerate?.Invoke(cancellationToken);
            return Task.FromResult(new GeneratedEmbeddings<Embedding<float>>(vectors.Select(vector => new Embedding<float>(vector))));
        }

        public object? GetService(Type serviceType, object? serviceKey = null) => null;

        public void Dispose() { }
    }

    private sealed class StubOllamaHandler(HttpStatusCode statusCode, string response) : HttpMessageHandler {
        public int CallCount { get; private set; }
        public HttpMethod? Method { get; private set; }
        public string? Path { get; private set; }
        public string Body { get; private set; } = string.Empty;

        protected override async Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken) {
            CallCount++;
            Method = request.Method;
            Path = request.RequestUri!.AbsolutePath;
            Body = await request.Content!.ReadAsStringAsync(cancellationToken);
            return new HttpResponseMessage(statusCode) { Content = new StringContent(response, Encoding.UTF8, "application/json") };
        }
    }
}
