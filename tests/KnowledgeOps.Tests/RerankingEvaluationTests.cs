extern alias evaluation;

using evaluation::KnowledgeOps.Evaluation;
using KnowledgeOps.Application;

namespace KnowledgeOps.Tests;

public class RerankingEvaluationTests {
    [Fact]
    public async Task Run_ScoresBothOrdersOfTheSameCandidatesAndReportsMissedRelevantDocuments() {
        var first = new ChunkSearchResult(Guid.NewGuid(), Guid.NewGuid(), 0, "First", 0, 5, 0.9f);
        var second = new ChunkSearchResult(Guid.NewGuid(), Guid.NewGuid(), 0, "Second", 0, 6, 0.7f);
        var ids = new Dictionary<string, Guid> { ["a"] = first.DocumentId, ["b"] = second.DocumentId, ["c"] = Guid.NewGuid() };
        var dataset = new EvaluationDataset(
            [new("a", "A", "First"), new("b", "B", "Second"), new("c", "C", "Third")],
            [new("Find second", ["b"], true), new("Find third", ["c"], true), new("Missing", [], false)]);
        var embeddings = new FakeTextEmbeddingService();
        var store = new FakeVectorStore { SearchResults = [first, second] };
        var reranker = new FakeTextReranker { Scores = [new(1, 3), new(0, 1)] };
        var search = new SemanticSearchService(embeddings, store);

        var report = await RerankingEvaluation.RunAsync("sample", "fake", dataset, ids, search,
            new RerankedSearchService(search, reranker), CancellationToken.None);

        Assert.Equal(3, embeddings.QueryCallCount);
        Assert.Equal(3, reranker.CallCount);
        Assert.Equal(0.5, report.CandidateRecall);
        Assert.Equal(0, report.DenseMetrics.HitAt1);
        Assert.Equal(0.5, report.RerankedMetrics.HitAt1);
        Assert.Equal(0.25, report.DenseMetrics.MrrAt5);
        Assert.Equal(0.5, report.RerankedMetrics.MrrAt5);
        Assert.Equal(new[] { first.PointId, second.PointId }, report.Questions[0].DenseRanking.Select(hit => hit.PointId));
        Assert.Equal(new[] { second.PointId, first.PointId }, report.Questions[0].RerankedRanking.Select(hit => hit.PointId));
        Assert.True(report.Questions[0].RelevantInCandidates);
        Assert.False(report.Questions[1].RelevantInCandidates);
        Assert.Null(report.Questions[2].RelevantInCandidates);
        Assert.NotNull(report.MeanRerankingMilliseconds);
    }
}
