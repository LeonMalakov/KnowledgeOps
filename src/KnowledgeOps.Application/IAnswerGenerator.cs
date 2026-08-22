namespace KnowledgeOps.Application;

public interface IAnswerGenerator {
    Task<string> GenerateAsync(
        string question, IReadOnlyList<RagSource> sources, CancellationToken cancellationToken = default);
}
