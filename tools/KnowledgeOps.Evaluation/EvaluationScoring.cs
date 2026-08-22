using KnowledgeOps.Application;
using KnowledgeOps.Infrastructure;

namespace KnowledgeOps.Evaluation;

internal readonly record struct EvaluationHit(Guid DocumentId, float Score);

internal static class EvaluationScoring {
    public const int RETRIEVAL_CHUNK_LIMIT = SemanticSearchService.MAX_TOP_K;

    public static async Task<EvaluationReport> EvaluateAsync(
        string datasetName, string mode, EvaluationDataset dataset, EmbeddingOptions embedding,
        IReadOnlyDictionary<string, Guid> idsByKey, Func<string, CancellationToken, Task<IEnumerable<EvaluationHit>>> search,
        CancellationToken cancellationToken) {
        var keysById = idsByKey.ToDictionary(entry => entry.Value, entry => entry.Key);
        var questions = new List<QuestionResult>(dataset.Questions.Length);
        Console.WriteLine($"Running {dataset.Questions.Length} {mode} queries for {datasetName}.");
        foreach (var question in dataset.Questions) {
            var hits = (await search(question.Query, cancellationToken)).DistinctBy(hit => hit.DocumentId).ToArray();
            var ranking = hits.Select(hit => new RankedDocument(keysById[hit.DocumentId], hit.Score)).ToArray();
            var relevantIds = question.RelevantDocuments.Select(key => idsByKey[key]).ToHashSet();
            int? rank = question.Answerable
                ? RetrievalMetrics.FirstRelevantRank(hits.Select(hit => hit.DocumentId), relevantIds)
                : null;
            questions.Add(new QuestionResult(
                question.Query, question.Answerable, question.RelevantDocuments, ranking, rank, ranking.FirstOrDefault()?.Score));
        }

        var answerableRanks = questions.Where(question => question.Answerable).Select(question => question.FirstRelevantRank).ToArray();
        return new EvaluationReport(datasetName, mode, embedding.Model, embedding.VectorSize, RETRIEVAL_CHUNK_LIMIT,
            dataset.Documents.Length, questions.Count, answerableRanks.Length, questions.Count - answerableRanks.Length,
            RetrievalMetrics.Calculate(answerableRanks), questions.ToArray());
    }
}
