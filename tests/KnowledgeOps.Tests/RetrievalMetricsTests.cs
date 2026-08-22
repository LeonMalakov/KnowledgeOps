extern alias evaluation;

using evaluation::KnowledgeOps.Evaluation;

namespace KnowledgeOps.Tests;

public class RetrievalMetricsTests {
    [Fact]
    public void FirstRelevantRank_CountsDistinctDocumentsAndUsesFirstRelevantHit() {
        var irrelevant = Guid.NewGuid();
        var firstRelevant = Guid.NewGuid();
        var secondRelevant = Guid.NewGuid();
        HashSet<Guid> relevant = [firstRelevant, secondRelevant];

        int? rank = RetrievalMetrics.FirstRelevantRank(
            [irrelevant, irrelevant, secondRelevant, firstRelevant], relevant);

        Assert.Equal(2, rank);
    }

    [Fact]
    public void FirstRelevantRank_ReturnsNullForMissingRelevantDocument() {
        HashSet<Guid> relevant = [Guid.NewGuid()];

        Assert.Null(RetrievalMetrics.FirstRelevantRank([Guid.NewGuid()], relevant));
        Assert.Null(RetrievalMetrics.FirstRelevantRank([], relevant));
    }

    [Fact]
    public void Calculate_UsesCutoffsAndCountsAnswerableMissesAsZero() {
        var result = RetrievalMetrics.Calculate([1, 3, 5, 6, null]);

        Assert.Equal(0.2, result.HitAt1);
        Assert.Equal(0.4, result.HitAt3);
        Assert.Equal(0.6, result.HitAt5);
        Assert.Equal((1 + 1d / 3 + 1d / 5) / 5, result.MrrAt5, precision: 10);
    }

    [Fact]
    public void Calculate_NoAnswerableQuestions_ReturnsZeroMetrics() {
        Assert.Equal(new RetrievalMetrics(0, 0, 0, 0), RetrievalMetrics.Calculate([]));
    }
}
