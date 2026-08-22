using System.Globalization;
using Grpc.Core;
using KnowledgeOps.Application;
using KnowledgeOps.Infrastructure;
using Microsoft.Extensions.Options;
using OllamaSharp;
using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace KnowledgeOps.Evaluation;

internal enum RussianTextProcessing {
    Plain,
    Stemmed,
    StemmedWithStopwords
}

internal readonly record struct ExperimentRrfWeights(int Dense, int Bm25) {
    public override string ToString() => FormattableString.Invariant($"{Dense}-{Bm25}");
}

internal static class RussianHybridExperiment {
    private const string CONTENT = "content";
    private const string DOCUMENT_ID = "document_id";
    private const string RUSSIAN = "russian";
    private const int LIMIT = EvaluationScoring.RETRIEVAL_CHUNK_LIMIT;

    public static IReadOnlyList<ExperimentRrfWeights> ParseWeights(string value) {
        var weights = new List<ExperimentRrfWeights>();
        foreach (string ratio in value.Split(',')) {
            var parts = ratio.Split(':');
            if (parts.Length != 2
                || !int.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out int dense) || dense <= 0
                || !int.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out int bm25) || bm25 <= 0) {
                throw new ArgumentException("Use positive Dense:BM25 ratios such as --rrf-weights=1:1,2:1,3:1.");
            }

            weights.Add(new ExperimentRrfWeights(dense, bm25));
        }

        return weights.Distinct().ToArray();
    }

    public static TextIndexParams CreateTextIndex(RussianTextProcessing processing) {
        return new TextIndexParams {
            Tokenizer = TokenizerType.Word,
            Lowercase = true,
            Scoring = new TextScoringParams { Type = TextScoringType.Bm25 },
            Stemmer = processing == RussianTextProcessing.Plain
                ? null : new StemmingAlgorithm { Snowball = new SnowballParams { Language = RUSSIAN } },
            Stopwords = processing == RussianTextProcessing.StemmedWithStopwords
                ? new StopwordsSet { Languages = { RUSSIAN } } : null
        };
    }

    public static (Rrf Rrf, PrefetchQuery[] Prefetch) CreateHybridQuery(
        string query, float[] vector, int topK, ExperimentRrfWeights weights) {
        ulong prefetchLimit = Math.Max(20UL, (ulong)topK * 3);
        return (new Rrf { Weights = { weights.Dense, weights.Bm25 } }, [
            new PrefetchQuery { Query = vector, Limit = prefetchLimit },
            new PrefetchQuery { Query = new TextQuery { Query = query }, Using = CONTENT, Limit = prefetchLimit }
        ]);
    }

    public static async Task<IReadOnlyList<EvaluationReport>> RunAsync(
        string datasetName, EvaluationDataset dataset, EmbeddingOptions embedding, string host, int port,
        IReadOnlyList<ExperimentRrfWeights> weights, CancellationToken cancellationToken) {
        using var ollama = new OllamaApiClient(new Uri(embedding.BaseUrl), embedding.Model);
        using var qdrant = new QdrantClient(host, port);
        var embeddings = new OllamaTextEmbeddingService(ollama, Options.Create(embedding));
        var (points, idsByKey) = await PrepareCorpusAsync(dataset, embeddings, cancellationToken);
        var vectorsByQuery = new Dictionary<string, float[]>(StringComparer.Ordinal);
        foreach (string query in dataset.Questions.Select(question => question.Query).Distinct(StringComparer.Ordinal)) {
            vectorsByQuery.Add(query, await embeddings.EmbedQueryAsync(query, cancellationToken));
        }

        var reports = new List<EvaluationReport>();
        foreach (var (variant, processing) in new[] {
            ("a", RussianTextProcessing.Plain),
            ("b", RussianTextProcessing.Stemmed),
            ("c", RussianTextProcessing.StemmedWithStopwords)
        }) {
            await InTemporaryCollectionAsync(qdrant, embedding.VectorSize, processing, async collection => {
                using var store = new QdrantVectorStore(new QdrantVectorStoreOptions(host, port, collection, embedding.VectorSize));
                await store.UpsertAsync(points, cancellationToken);
                Console.WriteLine($"Indexed {points.Length} chunks for {datasetName} / {variant} in {collection}.");

                if (processing == RussianTextProcessing.Plain) {
                    await EvaluateAsync("ru-exp-dense", (query, token) =>
                        qdrant.QueryAsync(collection, query: vectorsByQuery[query], limit: LIMIT, payloadSelector: true, cancellationToken: token));
                }

                await EvaluateAsync($"ru-exp-{variant}-bm25", (query, token) => qdrant.QueryAsync(
                    collection, query: new TextQuery { Query = query }, usingVector: CONTENT,
                    limit: LIMIT, payloadSelector: true, cancellationToken: token));

                foreach (var weight in weights) {
                    await EvaluateAsync($"ru-exp-{variant}-rrf-{weight}", (query, token) => {
                        var (rrf, prefetch) = CreateHybridQuery(query, vectorsByQuery[query], LIMIT, weight);
                        return qdrant.QueryAsync(collection, query: rrf, prefetch: prefetch,
                            limit: LIMIT, payloadSelector: true, cancellationToken: token);
                    });
                }

                async Task EvaluateAsync(string mode, Func<string, CancellationToken, Task<IReadOnlyList<ScoredPoint>>> queryPoints) {
                    var report = await EvaluationScoring.EvaluateAsync(datasetName, mode, dataset, embedding, idsByKey,
                        async (query, token) => (await queryPoints(query, token)).Select(
                            hit => new EvaluationHit(Guid.Parse(hit.Payload[DOCUMENT_ID].StringValue), hit.Score)),
                        cancellationToken);
                    reports.Add(report);
                }
            }, cancellationToken);
        }

        return reports;
    }

    public static async Task InTemporaryCollectionAsync(
        QdrantClient client, int vectorSize, RussianTextProcessing processing, Func<string, Task> run,
        CancellationToken cancellationToken = default) {
        string collection = $"knowledge_eval_ru_{Guid.NewGuid():N}";
        if (await client.CollectionExistsAsync(collection, cancellationToken)) {
            throw new InvalidOperationException($"Temporary collection '{collection}' already exists.");
        }

        bool cleanup = true;
        try {
            cancellationToken.ThrowIfCancellationRequested();
            using (var setupTimeout = new CancellationTokenSource(TimeSpan.FromSeconds(30))) {
                try {
                    await client.CreateCollectionAsync(collection,
                        new VectorParams { Size = (ulong)vectorSize, Distance = Distance.Cosine }, cancellationToken: setupTimeout.Token);
                } catch (RpcException exception) when (exception.StatusCode == StatusCode.AlreadyExists) {
                    cleanup = false;
                    throw;
                }

                await client.CreatePayloadIndexAsync(collection, DOCUMENT_ID, PayloadSchemaType.Keyword,
                    wait: true, cancellationToken: setupTimeout.Token);
                await client.CreatePayloadIndexAsync(collection, CONTENT, PayloadSchemaType.Text,
                    new PayloadIndexParams { TextIndexParams = CreateTextIndex(processing) },
                    wait: true, cancellationToken: setupTimeout.Token);
            }

            cancellationToken.ThrowIfCancellationRequested();
            await run(collection);
        } finally {
            if (cleanup) {
                using var cleanupTimeout = new CancellationTokenSource(TimeSpan.FromSeconds(15));
                try {
                    await client.DeleteCollectionAsync(collection, cancellationToken: cleanupTimeout.Token);
                } catch (RpcException exception) when (exception.StatusCode == StatusCode.NotFound) {
                    // Collection creation may not have completed.
                } catch (Exception exception) {
                    Console.Error.WriteLine($"Could not delete temporary collection '{collection}': {exception.Message}");
                }
            }
        }
    }

    private static async Task<(VectorizedChunk[] Points, IReadOnlyDictionary<string, Guid> IdsByKey)> PrepareCorpusAsync(
        EvaluationDataset dataset, ITextEmbeddingService embeddings, CancellationToken cancellationToken) {
        var chunker = new TextChunker();
        var idsByKey = new Dictionary<string, Guid>(StringComparer.Ordinal);
        var chunks = new List<(EvaluationDocument Document, Guid DocumentId, TextChunk Chunk)>();
        foreach (var document in dataset.Documents) {
            cancellationToken.ThrowIfCancellationRequested();
            var documentId = Guid.NewGuid();
            idsByKey.Add(document.Key, documentId);
            chunks.AddRange(chunker.Chunk(document.Content).Where(chunk => !string.IsNullOrWhiteSpace(chunk.Content))
                .Select(chunk => (document, documentId, chunk)));
        }

        var vectors = await embeddings.EmbedDocumentsAsync(chunks.Select(entry => entry.Chunk.Content).ToArray(), cancellationToken);
        var points = new VectorizedChunk[chunks.Count];
        for (int i = 0; i < points.Length; i++) {
            var (document, documentId, chunk) = chunks[i];
            // Stable point IDs keep ties comparable across collections and runs.
            var pointId = new Guid(i + 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            points[i] = new VectorizedChunk(pointId, documentId, chunk.Index, chunk.Content, chunk.StartOffset,
                chunk.EndOffset, vectors[i], document.Title, document.Key);
        }

        return (points, idsByKey);
    }
}
