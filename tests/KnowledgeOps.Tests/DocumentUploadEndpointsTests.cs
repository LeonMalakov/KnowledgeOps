using System.Net;
using System.Net.Http.Json;
using System.Net.Http.Headers;
using System.Text;
using KnowledgeOps.Application;
using KnowledgeOps.Infrastructure;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace KnowledgeOps.Tests;

public class DocumentUploadEndpointsTests(WebApplicationFactory<Program> factory) : IClassFixture<WebApplicationFactory<Program>> {
    [Fact]
    public async Task Upload_UsesExplicitTitleAndSourceAndIndexesMarkdownText() {
        var store = new FakeVectorStore();
        var embeddings = new FakeTextEmbeddingService();
        using var app = WithFakes(store, embeddings);
        using var client = app.CreateClient();
        const string content = "# Доставка\n\nТри **рабочих** дня.";
        using var form = CreateUpload("ignored.md", Encoding.UTF8.GetBytes(content), "Условия доставки", "help:delivery");

        using var response = await client.PostAsync("/api/documents/upload", form);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var result = await response.Content.ReadFromJsonAsync<DocumentIndexingResult>();
        Assert.NotNull(result);
        Assert.Equal(1, result.ChunkCount);
        var point = Assert.Single(store.Points!);
        Assert.Equal(result.DocumentId, point.DocumentId);
        Assert.Equal(("Условия доставки", "help:delivery", content), (point.Title, point.Source, point.Content));
        Assert.Equal(new[] { content }, embeddings.DocumentTexts);
    }

    [Theory]
    [InlineData("../../delivery.txt")]
    [InlineData("C:\\fakepath\\delivery.txt")]
    public async Task Upload_DefaultTitleUsesOnlyFileNameAndBuffersInMemory(string clientFileName) {
        var store = new FakeVectorStore();
        var embeddings = new FakeTextEmbeddingService();
        var extractor = new NoDiskSpillExtractor();
        using var app = WithFakes(store, embeddings, extractor);
        using var client = app.CreateClient();
        using var form = CreateUpload(clientFileName, Encoding.UTF8.GetBytes(new string('界', 25_000)));

        using var response = await client.PostAsync("/api/documents/upload", form);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var result = await response.Content.ReadFromJsonAsync<DocumentIndexingResult>();
        Assert.NotNull(result);
        Assert.Equal(30, result.ChunkCount);
        Assert.True(extractor.CheckedBuffering);
        Assert.All(store.Points!, point => {
            Assert.Equal("delivery", point.Title);
            Assert.Null(point.Source);
        });
    }

    [Fact]
    public async Task Upload_PdfIndexesExtractedPages() {
        var store = new FakeVectorStore();
        var embeddings = new FakeTextEmbeddingService();
        using var app = WithFakes(store, embeddings);
        using var client = app.CreateClient();
        using var form = CreateUpload("guide.pdf", DocumentTextExtractorTests.CreatePdf("First page", "Second page"));

        using var response = await client.PostAsync("/api/documents/upload", form);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("First page\nSecond page", Assert.Single(store.Points!).Content);
    }

    [Theory]
    [InlineData("unknown.docx", "data", HttpStatusCode.UnsupportedMediaType)]
    [InlineData("fake.pdf", "not PDF", HttpStatusCode.BadRequest)]
    [InlineData("broken.pdf", "%PDF-1.7\nbroken", HttpStatusCode.BadRequest)]
    [InlineData("empty.txt", "", HttpStatusCode.BadRequest)]
    [InlineData("whitespace.md", " \t", HttpStatusCode.BadRequest)]
    public async Task Upload_InvalidFilesReturnClientErrors(string fileName, string content, HttpStatusCode status) {
        var store = new FakeVectorStore();
        var embeddings = new FakeTextEmbeddingService();
        using var app = WithFakes(store, embeddings);
        using var client = app.CreateClient();
        using var form = CreateUpload(fileName, Encoding.UTF8.GetBytes(content));

        using var response = await client.PostAsync("/api/documents/upload", form);

        Assert.Equal(status, response.StatusCode);
        Assert.Equal(0, embeddings.DocumentCallCount);
        Assert.Empty(store.Calls);
    }

    [Fact]
    public async Task Upload_MissingFileReturnsBadRequest() {
        var store = new FakeVectorStore();
        var embeddings = new FakeTextEmbeddingService();
        using var app = WithFakes(store, embeddings);
        using var client = app.CreateClient();
        using var form = new MultipartFormDataContent { { new StringContent("Guide"), "title" } };

        using var response = await client.PostAsync("/api/documents/upload", form);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Empty(store.Calls);
    }

    [Theory]
    [InlineData(201, 1)]
    [InlineData(1, 1001)]
    public async Task Upload_InvalidMetadataReturnsBadRequest(int titleLength, int sourceLength) {
        var store = new FakeVectorStore();
        var embeddings = new FakeTextEmbeddingService();
        using var app = WithFakes(store, embeddings);
        using var client = app.CreateClient();
        using var form = CreateUpload("guide.txt", "Content"u8.ToArray(), new string('t', titleLength), new string('s', sourceLength));

        using var response = await client.PostAsync("/api/documents/upload", form);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Empty(store.Calls);
    }

    [Fact]
    public async Task Upload_TooLargeMultipartSectionReturnsClientError() {
        var store = new FakeVectorStore();
        var embeddings = new FakeTextEmbeddingService();
        using var app = WithFakes(store, embeddings);
        using var client = app.CreateClient();
        using var form = CreateUpload("large.txt", new byte[IDocumentTextExtractor.MAX_FILE_BYTES + 1]);

        using var response = await client.PostAsync("/api/documents/upload", form);

        Assert.True(response.StatusCode is HttpStatusCode.BadRequest or HttpStatusCode.RequestEntityTooLarge);
        Assert.Empty(store.Calls);
    }

    [Fact]
    public async Task Upload_BrowserRequestFromForeignOriginIsRejected() {
        var store = new FakeVectorStore();
        var embeddings = new FakeTextEmbeddingService();
        using var app = WithFakes(store, embeddings);
        using var client = app.CreateClient();
        using var request = new HttpRequestMessage(HttpMethod.Post, "/api/documents/upload") {
            Content = CreateUpload("guide.txt", "Text"u8.ToArray())
        };
        request.Headers.Add("Origin", "https://foreign.example");

        using var response = await client.SendAsync(request);

        Assert.Equal(HttpStatusCode.Forbidden, response.StatusCode);
        Assert.Empty(store.Calls);
    }

    private WebApplicationFactory<Program> WithFakes(
        FakeVectorStore store, FakeTextEmbeddingService embeddings, IDocumentTextExtractor? extractor = null) {
        return factory.WithWebHostBuilder(builder => builder.ConfigureTestServices(services => {
            services.RemoveAll<ITextEmbeddingService>();
            services.RemoveAll<IVectorStore>();
            services.AddSingleton<ITextEmbeddingService>(embeddings);
            services.AddSingleton<IVectorStore>(store);
            if (extractor is not null) {
                services.RemoveAll<IDocumentTextExtractor>();
                services.AddSingleton(extractor);
            }
        }));
    }

    private static MultipartFormDataContent CreateUpload(string fileName, byte[] bytes, string? title = null, string? source = null) {
        var form = new MultipartFormDataContent();
        var file = new ByteArrayContent(bytes);
        file.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
        form.Add(file, "file", fileName);
        if (title is not null) {
            form.Add(new StringContent(title), "title");
        }

        if (source is not null) {
            form.Add(new StringContent(source), "source");
        }

        return form;
    }

    private sealed class NoDiskSpillExtractor : IDocumentTextExtractor {
        private readonly string _temp = Environment.GetEnvironmentVariable("ASPNETCORE_TEMP") ?? Path.GetTempPath();
        private readonly HashSet<string> _before;

        public NoDiskSpillExtractor() {
            _before = Directory.EnumerateFiles(_temp, "ASPNETCORE_*.tmp").ToHashSet();
        }

        public bool CheckedBuffering { get; private set; }

        public Task<string> ExtractAsync(string fileName, Stream file, CancellationToken cancellationToken = default) {
            Assert.True(file.Length > 64 * 1024);
            Assert.Empty(Directory.EnumerateFiles(_temp, "ASPNETCORE_*.tmp").Except(_before));
            CheckedBuffering = true;
            return new DocumentTextExtractor().ExtractAsync(fileName, file, cancellationToken);
        }
    }
}
