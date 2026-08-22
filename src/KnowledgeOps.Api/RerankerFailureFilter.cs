using KnowledgeOps.Application;

namespace KnowledgeOps.Api;

internal sealed class RerankerFailureFilter : IEndpointFilter {
    public async ValueTask<object?> InvokeAsync(EndpointFilterInvocationContext context, EndpointFilterDelegate next) {
        try {
            return await next(context);
        } catch (TextRerankerException exception) {
            return TypedResults.Problem(title: exception.Message, statusCode: exception.Failure switch {
                TextRerankerFailure.Unavailable => StatusCodes.Status503ServiceUnavailable,
                TextRerankerFailure.Timeout => StatusCodes.Status504GatewayTimeout,
                _ => StatusCodes.Status502BadGateway
            });
        }
    }
}
