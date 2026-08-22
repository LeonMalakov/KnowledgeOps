namespace KnowledgeOps.Evaluation;

public sealed record RetrievalMetrics(double HitAt1, double HitAt3, double HitAt5, double MrrAt5) {
    public static int? FirstRelevantRank(IEnumerable<Guid> rankedDocumentIds, IReadOnlySet<Guid> relevantDocumentIds) {
        int rank = 0;
        foreach (var documentId in rankedDocumentIds.Distinct()) {
            rank++;
            if (relevantDocumentIds.Contains(documentId)) {
                return rank;
            }
        }

        return null;
    }

    public static RetrievalMetrics Calculate(IReadOnlyList<int?> answerableRanks) {
        if (answerableRanks.Count == 0) {
            return new RetrievalMetrics(0, 0, 0, 0);
        }

        double count = answerableRanks.Count;
        return new RetrievalMetrics(
            answerableRanks.Count(rank => rank == 1) / count,
            answerableRanks.Count(rank => rank is >= 1 and <= 3) / count,
            answerableRanks.Count(rank => rank is >= 1 and <= 5) / count,
            answerableRanks.Sum(rank => rank is >= 1 and <= 5 ? 1d / rank.Value : 0d) / count);
    }
}
