using KnowledgeOps.Application;

namespace KnowledgeOps.Api;

internal static class RerankingEndpoints {
    public static void MapRerankingEndpoints(this WebApplication app) {
        app.MapPost("/api/search/reranked", SearchAsync).AddEndpointFilter<RerankerFailureFilter>();
    }

    private static async Task<IResult> SearchAsync(
        RerankedSearchRequest request, RerankedSearchService search, CancellationToken cancellationToken) {
        if (string.IsNullOrWhiteSpace(request.Query)) {
            return TypedResults.ValidationProblem(new Dictionary<string, string[]> {
                ["query"] = ["Query must contain non-whitespace characters."]
            });
        }

        if (request.TopK is < RerankedSearchService.MIN_TOP_K or > RerankedSearchService.MAX_TOP_K) {
            return TypedResults.ValidationProblem(new Dictionary<string, string[]> {
                ["topK"] = [$"TopK must be between {RerankedSearchService.MIN_TOP_K} and {RerankedSearchService.MAX_TOP_K}."]
            });
        }

        int candidateK = request.CandidateK ?? RerankedSearchService.DEFAULT_CANDIDATE_K;
        if (candidateK < request.TopK || candidateK > RerankedSearchService.MAX_CANDIDATE_K) {
            return TypedResults.ValidationProblem(new Dictionary<string, string[]> {
                ["candidateK"] = [$"CandidateK must be at least TopK and at most {RerankedSearchService.MAX_CANDIDATE_K}."]
            });
        }

        return TypedResults.Ok(await search.SearchAsync(request.Query, request.TopK, candidateK, cancellationToken));
    }

    private sealed record RerankedSearchRequest(string? Query, int TopK, int? CandidateK);
}
