namespace KnowledgeOps.Api;

internal sealed class RagOptions {
    public const string SECTION_NAME = "Rag";
    public const string DENSE = "dense";
    public const string RERANKED = "reranked";

    public string RetrievalMode { get; init; } = "";
    public int CandidateK { get; init; }
    public int MaxContextCharacters { get; init; }
}
