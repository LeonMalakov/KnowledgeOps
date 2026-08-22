using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using KnowledgeOps.Application;

namespace KnowledgeOps.Infrastructure;

public sealed class TeiTextReranker(HttpClient client) : ITextReranker {
    public async Task<IReadOnlyList<TextRerankScore>> RerankAsync(
        string query, IReadOnlyList<string> texts, CancellationToken cancellationToken = default) {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        ArgumentNullException.ThrowIfNull(texts);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(texts.Count, RerankedSearchService.MAX_CANDIDATE_K, nameof(texts));
        cancellationToken.ThrowIfCancellationRequested();
        if (texts.Count == 0) {
            return [];
        }

        TeiScore?[]? payload;
        try {
            using var response = await client.PostAsJsonAsync("rerank", new { query, texts, raw_scores = true }, cancellationToken);
            if (!response.IsSuccessStatusCode) {
                var failure = response.StatusCode == HttpStatusCode.ServiceUnavailable
                    ? TextRerankerFailure.Unavailable : TextRerankerFailure.UpstreamError;
                throw new TextRerankerException(failure, $"TEI rerank returned HTTP {(int)response.StatusCode}.");
            }

            payload = await response.Content.ReadFromJsonAsync<TeiScore?[]>(cancellationToken);
        } catch (OperationCanceledException exception) when (!cancellationToken.IsCancellationRequested) {
            throw new TextRerankerException(TextRerankerFailure.Timeout, "The TEI rerank request timed out.", exception);
        } catch (HttpRequestException exception) {
            throw new TextRerankerException(TextRerankerFailure.Unavailable, "Could not reach the TEI reranker.", exception);
        } catch (JsonException exception) {
            throw new TextRerankerException(TextRerankerFailure.UpstreamError, "TEI rerank returned invalid JSON.", exception);
        }

        if (payload?.Length != texts.Count) {
            throw new TextRerankerException(TextRerankerFailure.UpstreamError,
                $"TEI rerank returned {payload?.Length ?? 0} scores for {texts.Count} candidates.");
        }

        var seen = new bool[texts.Count];
        var scores = new TextRerankScore[texts.Count];
        foreach (var result in payload) {
            if (result?.Index is not int index || index < 0 || index >= scores.Length || seen[index]) {
                throw new TextRerankerException(TextRerankerFailure.UpstreamError, "TEI rerank returned a missing, invalid or repeated index.");
            }

            if (result.Score is not float score || !float.IsFinite(score)) {
                throw new TextRerankerException(TextRerankerFailure.UpstreamError, "TEI rerank returned a missing or non-finite score.");
            }

            seen[index] = true;
            scores[index] = new TextRerankScore(index, score);
        }

        return scores;
    }

    private sealed record TeiScore(int? Index, float? Score);
}
