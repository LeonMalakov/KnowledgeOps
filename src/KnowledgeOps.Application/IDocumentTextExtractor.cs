namespace KnowledgeOps.Application;

public interface IDocumentTextExtractor {
    public const int MAX_FILE_BYTES = 5 * 1024 * 1024;

    Task<string> ExtractAsync(string fileName, Stream file, CancellationToken cancellationToken = default);
}

public sealed class DocumentTextExtractionException(string message, Exception? innerException = null)
    : Exception(message, innerException);
