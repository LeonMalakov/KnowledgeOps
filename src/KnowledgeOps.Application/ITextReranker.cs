namespace KnowledgeOps.Application;

public interface ITextReranker {
    /// <summary>Returns one finite score per candidate, keyed by its zero-based input index.</summary>
    Task<IReadOnlyList<TextRerankScore>> RerankAsync(
        string query, IReadOnlyList<string> texts, CancellationToken cancellationToken = default);
}

public sealed record TextRerankScore(int Index, float Score);

public enum TextRerankerFailure {
    Unavailable,
    Timeout,
    UpstreamError
}

public sealed class TextRerankerException(TextRerankerFailure failure, string message, Exception? innerException = null)
    : Exception(message, innerException) {
    public TextRerankerFailure Failure { get; } = failure;
}
