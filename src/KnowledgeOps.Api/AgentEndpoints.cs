using KnowledgeOps.Infrastructure;

namespace KnowledgeOps.Api;

internal static class AgentEndpoints {
    public static void MapAgentEndpoints(this WebApplication app) {
        app.MapPost("/api/agent/run", RunAsync);
        app.MapPost("/api/agent/actions/confirm", Confirm);
    }

    private static async Task<IResult> RunAsync(AgentRequest request, AgentService agent, CancellationToken cancellationToken) {
        if (string.IsNullOrWhiteSpace(request.Message) || request.Message.Length > AgentService.MAX_MESSAGE_LENGTH) {
            return TypedResults.ValidationProblem(new Dictionary<string, string[]> {
                ["message"] = [$"Message must contain non-whitespace characters and be at most {AgentService.MAX_MESSAGE_LENGTH} characters."]
            });
        }

        if (request.ConversationId == Guid.Empty) {
            return TypedResults.ValidationProblem(new Dictionary<string, string[]> {
                ["conversationId"] = ["ConversationId must be a non-empty identifier."]
            });
        }

        try {
            return TypedResults.Ok(await agent.RunAsync(request.Message, cancellationToken, request.ConversationId));
        } catch (AgentException exception) {
            return ToProblem(exception);
        }
    }

    private static IResult Confirm(ConfirmRequest request, AgentConversationStore conversations, CancellationToken cancellationToken) {
        if (request.ConversationId == Guid.Empty || request.ActionId == Guid.Empty || request.Confirmed is null) {
            return TypedResults.ValidationProblem(new Dictionary<string, string[]> {
                ["confirmation"] = ["ConversationId, ActionId and an explicit confirmed value are required."]
            });
        }

        try {
            return TypedResults.Ok(conversations.Decide(request.ConversationId, request.ActionId, request.Confirmed.Value, cancellationToken));
        } catch (AgentException exception) {
            return ToProblem(exception);
        }
    }

    private static IResult ToProblem(AgentException exception) =>
        TypedResults.Problem(title: exception.Message, statusCode: exception.Failure switch {
            AgentFailure.Unavailable or AgentFailure.Capacity => StatusCodes.Status503ServiceUnavailable,
            AgentFailure.Timeout => StatusCodes.Status504GatewayTimeout,
            AgentFailure.NotFound => StatusCodes.Status404NotFound,
            AgentFailure.Conflict => StatusCodes.Status409Conflict,
            AgentFailure.Expired => StatusCodes.Status410Gone,
            _ => StatusCodes.Status502BadGateway
        });

    private sealed record AgentRequest(Guid? ConversationId, string? Message);
    private sealed record ConfirmRequest(Guid ConversationId, Guid ActionId, bool? Confirmed);
}
