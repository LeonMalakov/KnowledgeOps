using KnowledgeOps.Application;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;

namespace KnowledgeOps.Infrastructure;

public sealed class TeiRerankerOptions {
    public const string SECTION_NAME = "Reranking";

    public string BaseUrl { get; init; } = string.Empty;
    public int TimeoutSeconds { get; init; }
}

public static class TeiRerankerRegistration {
    public static IHttpClientBuilder AddTeiReranker(this IServiceCollection services) {
        return services.AddHttpClient<ITextReranker, TeiTextReranker>((provider, client) => {
            var options = provider.GetRequiredService<IOptions<TeiRerankerOptions>>().Value;
            if (!Uri.TryCreate(options.BaseUrl.TrimEnd('/') + "/", UriKind.Absolute, out var url)
                || url.Scheme is not ("http" or "https") || options.TimeoutSeconds <= 0) {
                throw new InvalidOperationException("Reranking requires an absolute HTTP(S) BaseUrl and a positive TimeoutSeconds.");
            }

            client.BaseAddress = url;
            client.Timeout = TimeSpan.FromSeconds(options.TimeoutSeconds);
        });
    }
}
