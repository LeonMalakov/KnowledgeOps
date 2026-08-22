namespace KnowledgeOps.Application;

public sealed class RagAnswerService {
    public const int MAX_QUERY_LENGTH = 2_000;
    public const int MIN_TOP_K = 1;
    public const int MAX_TOP_K = 10;

    private const string INSUFFICIENT_CONTEXT = "В базе знаний недостаточно информации для ответа.";
    private const string EMPTY_ANSWER = "Модель не вернула текст ответа.";

    private readonly SemanticSearchService _search;
    private readonly IAnswerGenerator _generator;
    private readonly int _maxContextCharacters;
    private readonly RerankedSearchService? _rerankedSearch;
    private readonly int _candidateK;

    public RagAnswerService(
        SemanticSearchService search, IAnswerGenerator generator, int maxContextCharacters,
        RerankedSearchService? rerankedSearch = null, int candidateK = RerankedSearchService.DEFAULT_CANDIDATE_K) {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxContextCharacters);
        _search = search;
        _generator = generator;
        _maxContextCharacters = maxContextCharacters;
        _rerankedSearch = rerankedSearch;
        _candidateK = candidateK;
    }

    public async Task<RagAnswerResult> AnswerAsync(
        string query, int topK, CancellationToken cancellationToken = default) {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(query.Length, MAX_QUERY_LENGTH, nameof(query));
        ArgumentOutOfRangeException.ThrowIfLessThan(topK, MIN_TOP_K);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(topK, MAX_TOP_K);

        List<RagSource> sources;
        if (_rerankedSearch is null) {
            var hits = await _search.SearchAsync(query, topK, cancellationToken);
            sources = SelectSources(hits, static (hit, number) => new RagSource(number, hit.DocumentId, hit.ChunkIndex,
                hit.Content, hit.StartOffset, hit.EndOffset, hit.Score, hit.Title, hit.Source));
        } else {
            var hits = await _rerankedSearch.SearchAsync(query, topK, _candidateK, cancellationToken);
            sources = SelectSources(hits, static (hit, number) => new RagSource(number, hit.DocumentId, hit.ChunkIndex,
                hit.Content, hit.StartOffset, hit.EndOffset, hit.RetrievalScore, hit.Title, hit.Source, hit.RerankScore));
        }

        if (sources.Count == 0) {
            return new RagAnswerResult(INSUFFICIENT_CONTEXT, []);
        }

        string answer = await _generator.GenerateAsync(query, sources, cancellationToken);
        return new RagAnswerResult(string.IsNullOrWhiteSpace(answer) ? EMPTY_ANSWER : answer, sources);
    }

    private List<RagSource> SelectSources<T>(IReadOnlyList<T> hits, Func<T, int, RagSource> toSource) {
        var sources = new List<RagSource>(hits.Count);
        int remaining = _maxContextCharacters;
        foreach (var hit in hits) {
            var source = toSource(hit, sources.Count + 1);
            if (source.Content.Length <= remaining) {
                sources.Add(source);
                remaining -= source.Content.Length;
            }
        }

        return sources;
    }
}
