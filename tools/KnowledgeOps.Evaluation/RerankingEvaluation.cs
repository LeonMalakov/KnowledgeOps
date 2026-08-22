using System.Diagnostics;
using KnowledgeOps.Application;

namespace KnowledgeOps.Evaluation;

internal sealed record RerankingCandidate(
    string Key, int ChunkIndex, Guid PointId, float RetrievalScore, float? RerankScore);

internal sealed record RerankingQuestion(
    string Query, bool Answerable, string[] RelevantDocuments, bool? RelevantInCandidates,
    int? DenseRank, int? RerankedRank, double? RerankingMilliseconds,
    RerankingCandidate[] DenseRanking, RerankingCandidate[] RerankedRanking);

internal sealed record RerankingEvaluationReport(
    string Dataset, string EmbeddingModel, int CandidateK, int AnswerableCount, int RelevantCandidateCount,
    double? CandidateRecall, double? MeanRerankingMilliseconds, RetrievalMetrics DenseMetrics,
    RetrievalMetrics RerankedMetrics, RerankingQuestion[] Questions);

internal static class RerankingEvaluation {
    public static async Task<RerankingEvaluationReport> RunAsync(
        string datasetName, string embeddingModel, EvaluationDataset dataset, IReadOnlyDictionary<string, Guid> idsByKey,
        SemanticSearchService search, RerankedSearchService reranker, CancellationToken cancellationToken) {
        const int candidateK = RerankedSearchService.DEFAULT_CANDIDATE_K;
        var keysById = idsByKey.ToDictionary(entry => entry.Value, entry => entry.Key);
        var questions = new List<RerankingQuestion>(dataset.Questions.Length);

        foreach (var question in dataset.Questions) {
            var candidates = await search.SearchAsync(question.Query, candidateK, cancellationToken);
            long start = Stopwatch.GetTimestamp();
            var reranked = await reranker.RerankCandidatesAsync(question.Query, candidates, cancellationToken);
            double? elapsed = candidates.Count == 0 ? null : Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            var relevantIds = question.RelevantDocuments.Select(key => idsByKey[key]).ToHashSet();
            int? denseRank = RetrievalMetrics.FirstRelevantRank(candidates.Select(hit => hit.DocumentId), relevantIds);
            int? rerankedRank = RetrievalMetrics.FirstRelevantRank(reranked.Select(hit => hit.DocumentId), relevantIds);

            questions.Add(new RerankingQuestion(question.Query, question.Answerable, question.RelevantDocuments,
                question.Answerable ? denseRank.HasValue : null, denseRank, rerankedRank, elapsed,
                candidates.Select(hit => new RerankingCandidate(keysById[hit.DocumentId], hit.ChunkIndex, hit.PointId, hit.Score, null)).ToArray(),
                reranked.Select(hit => new RerankingCandidate(keysById[hit.DocumentId], hit.ChunkIndex, hit.PointId, hit.RetrievalScore, hit.RerankScore)).ToArray()));
        }

        var answerable = questions.Where(question => question.Answerable).ToArray();
        int relevantCandidateCount = answerable.Count(question => question.RelevantInCandidates == true);
        return new RerankingEvaluationReport(datasetName, embeddingModel, candidateK, answerable.Length, relevantCandidateCount,
            answerable.Length == 0 ? null : (double)relevantCandidateCount / answerable.Length,
            questions.Average(question => question.RerankingMilliseconds),
            RetrievalMetrics.Calculate(answerable.Select(question => question.DenseRank).ToArray()),
            RetrievalMetrics.Calculate(answerable.Select(question => question.RerankedRank).ToArray()), questions.ToArray());
    }

    public static void PrintReport(RerankingEvaluationReport report) {
        Console.WriteLine($"\n{report.Dataset} / reranking (candidateK={report.CandidateK})");
        Console.WriteLine("Mode | Hit@1 | Hit@3 | Hit@5 | MRR@5");
        PrintMetrics("Dense", report.DenseMetrics);
        PrintMetrics("Dense + Reranker", report.RerankedMetrics);
        Console.WriteLine($"Relevant document in candidates: {report.RelevantCandidateCount}/{report.AnswerableCount}");
        Console.WriteLine(FormattableString.Invariant($"Mean reranking latency: {report.MeanRerankingMilliseconds:F1} ms"));
        foreach (var question in report.Questions) {
            Console.WriteLine($"\n{question.Query}");
            Console.WriteLine($"  Dense: {FormatRanking(question.DenseRanking)}");
            Console.WriteLine($"  Reranked: {FormatRanking(question.RerankedRanking)}");
            Console.WriteLine($"  Relevant in candidates: {question.RelevantInCandidates?.ToString() ?? "—"}");
        }
    }

    private static string FormatRanking(IEnumerable<RerankingCandidate> ranking) =>
        string.Join(", ", ranking.Select(hit => $"{hit.Key}#{hit.ChunkIndex}"));

    private static void PrintMetrics(string mode, RetrievalMetrics metrics) =>
        Console.WriteLine(FormattableString.Invariant(
            $"{mode} | {metrics.HitAt1:P2} | {metrics.HitAt3:P2} | {metrics.HitAt5:P2} | {metrics.MrrAt5:F4}"));
}
