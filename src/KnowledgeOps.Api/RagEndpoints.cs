using KnowledgeOps.Application;

namespace KnowledgeOps.Api;

internal static class RagEndpoints {
    public static void MapRagEndpoints(this WebApplication app) {
        app.MapPost("/api/ask", AskAsync).AddEndpointFilter<RerankerFailureFilter>();
    }

    private static async Task<IResult> AskAsync(
        AskRequest request, RagAnswerService rag, CancellationToken cancellationToken) {
        if (string.IsNullOrWhiteSpace(request.Query)) {
            return TypedResults.ValidationProblem(new Dictionary<string, string[]> {
                ["query"] = ["Query must contain non-whitespace characters."]
            });
        }

        if (request.Query.Length > RagAnswerService.MAX_QUERY_LENGTH) {
            return TypedResults.ValidationProblem(new Dictionary<string, string[]> {
                ["query"] = [$"Query must not exceed {RagAnswerService.MAX_QUERY_LENGTH} characters."]
            });
        }

        if (request.TopK is < RagAnswerService.MIN_TOP_K or > RagAnswerService.MAX_TOP_K) {
            return TypedResults.ValidationProblem(new Dictionary<string, string[]> {
                ["topK"] = [$"TopK must be between {RagAnswerService.MIN_TOP_K} and {RagAnswerService.MAX_TOP_K}."]
            });
        }

        return TypedResults.Ok(await rag.AnswerAsync(request.Query, request.TopK, cancellationToken));
    }

    private sealed record AskRequest(string? Query, int TopK);
}
