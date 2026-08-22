namespace KnowledgeOps.Infrastructure;

public sealed class OllamaChatOptions {
    public const string SECTION_NAME = "Chat";

    public string BaseUrl { get; init; } = string.Empty;

    public string Model { get; init; } = string.Empty;
}
