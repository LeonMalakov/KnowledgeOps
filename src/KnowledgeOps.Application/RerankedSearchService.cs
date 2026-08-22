namespace KnowledgeOps.Application;

public sealed record RerankedChunkSearchResult(
    Guid PointId, Guid DocumentId, int ChunkIndex, string Content, int StartOffset, int EndOffset,
    float RetrievalScore, float RerankScore, string Title, string? Source);

public sealed class RerankedSearchService(SemanticSearchService search, ITextReranker reranker) {
    public const int MIN_TOP_K = 1;
    public const int MAX_TOP_K = 10;
    public const int DEFAULT_CANDIDATE_K = 20;
    public const int MAX_CANDIDATE_K = 50;

    public async Task<IReadOnlyList<RerankedChunkSearchResult>> SearchAsync(
        string query, int topK, int candidateK = DEFAULT_CANDIDATE_K, CancellationToken cancellationToken = default) {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        ArgumentOutOfRangeException.ThrowIfLessThan(topK, MIN_TOP_K);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(topK, MAX_TOP_K);
        ArgumentOutOfRangeException.ThrowIfLessThan(candidateK, topK);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(candidateK, MAX_CANDIDATE_K);

        var candidates = await search.SearchAsync(query, candidateK, cancellationToken);
        var reranked = await RerankCandidatesAsync(query, candidates, cancellationToken);
        return reranked.Take(topK).ToArray();
    }

    public async Task<IReadOnlyList<RerankedChunkSearchResult>> RerankCandidatesAsync(
        string query, IReadOnlyList<ChunkSearchResult> candidates, CancellationToken cancellationToken = default) {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        ArgumentNullException.ThrowIfNull(candidates);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(candidates.Count, MAX_CANDIDATE_K, nameof(candidates));
        cancellationToken.ThrowIfCancellationRequested();
        if (candidates.Count == 0) {
            return [];
        }

        var scores = await reranker.RerankAsync(query, candidates.Select(candidate => candidate.Content).ToArray(), cancellationToken);
        return scores.OrderByDescending(score => score.Score).ThenBy(score => score.Index).Select(score => {
            var chunk = candidates[score.Index];
            return new RerankedChunkSearchResult(chunk.PointId, chunk.DocumentId, chunk.ChunkIndex, chunk.Content,
                chunk.StartOffset, chunk.EndOffset, chunk.Score, score.Score, chunk.Title, chunk.Source);
        }).ToArray();
    }
}
