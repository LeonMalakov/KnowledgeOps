using System.Net;
using System.Text;
using System.Text.Json;
using KnowledgeOps.Application;
using KnowledgeOps.Infrastructure;

namespace KnowledgeOps.Tests;

public class TeiTextRerankerTests {
    [Fact]
    public async Task Rerank_PostsOneBatchAndMapsRawScoresByIndex() {
        int calls = 0;
        using var client = CreateClient(async (request, token) => {
            calls++;
            Assert.Equal(HttpMethod.Post, request.Method);
            Assert.Equal("/rerank", request.RequestUri!.AbsolutePath);
            using var json = JsonDocument.Parse(await request.Content!.ReadAsStringAsync(token));
            Assert.Equal("Доставка", json.RootElement.GetProperty("query").GetString());
            Assert.Equal(new[] { "Первый", "Второй" }, json.RootElement.GetProperty("texts").EnumerateArray().Select(item => item.GetString()));
            Assert.True(json.RootElement.GetProperty("raw_scores").GetBoolean());
            return JsonResponse("""[{"index":1,"score":3.4},{"index":0,"score":-2}]""");
        });

        var scores = await new TeiTextReranker(client).RerankAsync("Доставка", ["Первый", "Второй"]);

        Assert.Equal(1, calls);
        Assert.Equal(new[] { new TextRerankScore(0, -2f), new TextRerankScore(1, 3.4f) }, scores);
    }

    [Theory]
    [InlineData("null")]
    [InlineData("not json")]
    [InlineData("""[{"index":0,"score":1}]""")]
    [InlineData("""[{"index":0,"score":1},{"index":0,"score":2}]""")]
    [InlineData("""[{"index":0,"score":1},{"index":2,"score":2}]""")]
    [InlineData("""[{"index":0,"score":1},{"index":-1,"score":2}]""")]
    [InlineData("""[{"index":0,"score":1},{"score":2}]""")]
    [InlineData("""[{"index":0,"score":1},{"index":1}]""")]
    [InlineData("""[{"index":0,"score":1},{"index":1,"score":1e999}]""")]
    public async Task Rerank_RejectsInvalidResponses(string json) {
        using var client = CreateClient((_, _) => Task.FromResult(JsonResponse(json)));

        var exception = await Assert.ThrowsAsync<TextRerankerException>(
            () => new TeiTextReranker(client).RerankAsync("query", ["first", "second"]));

        Assert.Equal(TextRerankerFailure.UpstreamError, exception.Failure);
    }

    [Theory]
    [InlineData(HttpStatusCode.BadRequest, TextRerankerFailure.UpstreamError)]
    [InlineData(HttpStatusCode.ServiceUnavailable, TextRerankerFailure.Unavailable)]
    public async Task Rerank_ReportsHttpFailures(HttpStatusCode status, TextRerankerFailure expected) {
        using var client = CreateClient((_, _) => Task.FromResult(new HttpResponseMessage(status)));

        var exception = await Assert.ThrowsAsync<TextRerankerException>(
            () => new TeiTextReranker(client).RerankAsync("query", ["candidate"]));

        Assert.Equal(expected, exception.Failure);
        Assert.Contains(((int)status).ToString(), exception.Message);
    }

    [Fact]
    public async Task Rerank_ReportsNetworkFailureAndTimeout() {
        using var unreachable = CreateClient((_, _) => Task.FromException<HttpResponseMessage>(new HttpRequestException("Connection refused.")));
        using var timeout = CreateClient((_, _) => Task.FromException<HttpResponseMessage>(new TaskCanceledException("Timed out.")));

        var unavailable = await Assert.ThrowsAsync<TextRerankerException>(
            () => new TeiTextReranker(unreachable).RerankAsync("query", ["candidate"]));
        var timedOut = await Assert.ThrowsAsync<TextRerankerException>(
            () => new TeiTextReranker(timeout).RerankAsync("query", ["candidate"]));

        Assert.Equal(TextRerankerFailure.Unavailable, unavailable.Failure);
        Assert.Equal(TextRerankerFailure.Timeout, timedOut.Failure);
    }

    [Fact]
    public async Task Rerank_CancellationAbortsHttpRequestAndPropagates() {
        using var cancellation = new CancellationTokenSource();
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var client = CreateClient(async (_, token) => {
            entered.SetResult();
            await Task.Delay(Timeout.InfiniteTimeSpan, token);
            return JsonResponse("[]");
        });
        var rerank = new TeiTextReranker(client).RerankAsync("query", ["candidate"], cancellation.Token);

        await entered.Task.WaitAsync(TimeSpan.FromSeconds(5));
        cancellation.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => rerank.WaitAsync(TimeSpan.FromSeconds(5)));
    }

    private static HttpClient CreateClient(Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> send) =>
        new(new StubHandler(send)) { BaseAddress = new Uri("http://localhost:8081") };

    private static HttpResponseMessage JsonResponse(string json) => new(HttpStatusCode.OK) {
        Content = new StringContent(json, Encoding.UTF8, "application/json")
    };

    private sealed class StubHandler(Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> send) : HttpMessageHandler {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            send(request, cancellationToken);
    }
}
