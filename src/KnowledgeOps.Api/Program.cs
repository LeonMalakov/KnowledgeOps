using KnowledgeOps.Api;
using KnowledgeOps.Application;
using KnowledgeOps.Infrastructure;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Options;
using OllamaSharp;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddHealthChecks();

builder.Services.AddOptions<EmbeddingOptions>()
    .BindConfiguration(EmbeddingOptions.SECTION_NAME)
    .Validate(options => Uri.TryCreate(options.BaseUrl, UriKind.Absolute, out var url)
        && url.Scheme is "http" or "https", "Embedding:BaseUrl must be an absolute HTTP or HTTPS URL.")
    .Validate(options => !string.IsNullOrWhiteSpace(options.Model), "Embedding:Model is required.")
    .Validate(options => options.VectorSize > 0, "Embedding:VectorSize must be positive.")
    .ValidateOnStart();
builder.Services.AddSingleton<IEmbeddingGenerator<string, Embedding<float>>>(services => {
    var options = services.GetRequiredService<IOptions<EmbeddingOptions>>().Value;
    return new OllamaApiClient(new Uri(options.BaseUrl), options.Model);
});
builder.Services.AddSingleton<ITextEmbeddingService, OllamaTextEmbeddingService>();

builder.Services.AddOptions<OllamaChatOptions>()
    .BindConfiguration(OllamaChatOptions.SECTION_NAME)
    .Validate(options => Uri.TryCreate(options.BaseUrl, UriKind.Absolute, out var url)
        && url.Scheme is "http" or "https", "Chat:BaseUrl must be an absolute HTTP or HTTPS URL.")
    .Validate(options => !string.IsNullOrWhiteSpace(options.Model), "Chat:Model is required.")
    .ValidateOnStart();
builder.Services.AddSingleton<IChatClient>(services => {
    var options = services.GetRequiredService<IOptions<OllamaChatOptions>>().Value;
    return new OllamaApiClient(new Uri(options.BaseUrl), options.Model);
});
builder.Services.AddSingleton<IAnswerGenerator, OllamaAnswerGenerator>();
builder.Services.AddOptions<RagOptions>()
    .BindConfiguration(RagOptions.SECTION_NAME)
    .Validate(options => options.RetrievalMode is RagOptions.DENSE or RagOptions.RERANKED,
        "Rag:RetrievalMode must be 'dense' or 'reranked'.")
    .Validate(options => options.CandidateK is >= RagAnswerService.MAX_TOP_K and <= RerankedSearchService.MAX_CANDIDATE_K,
        $"Rag:CandidateK must be between {RagAnswerService.MAX_TOP_K} (the maximum TopK for /api/ask) and {RerankedSearchService.MAX_CANDIDATE_K}.")
    .Validate(options => options.MaxContextCharacters > 0, "Rag:MaxContextCharacters must be positive.")
    .ValidateOnStart();
builder.Services.AddTransient(services => {
    var options = services.GetRequiredService<IOptions<RagOptions>>().Value;
    return new RagAnswerService(
        services.GetRequiredService<SemanticSearchService>(),
        services.GetRequiredService<IAnswerGenerator>(),
        options.MaxContextCharacters,
        options.RetrievalMode == RagOptions.RERANKED ? services.GetRequiredService<RerankedSearchService>() : null,
        options.CandidateK);
});

builder.Services.AddSingleton<TextChunker>();
builder.Services.AddSingleton<IDocumentTextExtractor, DocumentTextExtractor>();
builder.Services.AddSingleton<DocumentIndexingService>();
builder.Services.AddSingleton<SemanticSearchService>();
builder.Services.AddSingleton<HybridSearchService>();
builder.Services.AddOptions<TeiRerankerOptions>().BindConfiguration(TeiRerankerOptions.SECTION_NAME);
builder.Services.AddTeiReranker();
builder.Services.AddTransient<RerankedSearchService>();
builder.Services.AddSingleton(TimeProvider.System);
builder.Services.AddSingleton<AgentConversationStore>();
builder.Services.AddTransient<AgentService>();
builder.Services.AddSingleton<IVectorStore>(services => {
    var qdrant = services.GetRequiredService<IConfiguration>().GetRequiredSection("Qdrant");
    var embedding = services.GetRequiredService<IOptions<EmbeddingOptions>>().Value;
    return new QdrantVectorStore(new QdrantVectorStoreOptions(
        qdrant["Host"] ?? throw new InvalidOperationException("Qdrant:Host is required."),
        qdrant.GetValue<int>("GrpcPort"),
        qdrant["CollectionName"] ?? throw new InvalidOperationException("Qdrant:CollectionName is required."),
        embedding.VectorSize));
});

var app = builder.Build();

app.MapHealthChecks("/health");
app.MapDocumentEndpoints();
app.MapRerankingEndpoints();
app.MapRagEndpoints();
app.MapAgentEndpoints();

app.Run();

public partial class Program { }
