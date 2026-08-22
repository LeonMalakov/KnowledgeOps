using System.Globalization;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Unicode;
using Grpc.Core;
using KnowledgeOps.Application;
using KnowledgeOps.Evaluation;
using KnowledgeOps.Infrastructure;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using OllamaSharp;
using Qdrant.Client;

const string settingsPath = "src/KnowledgeOps.Api/appsettings.json";
const string experimentFlag = "--russian-hybrid";
const string rerankingFlag = "--reranking";
const string weightsPrefix = "--rrf-weights=";

Console.OutputEncoding = Encoding.UTF8;
using var cancellation = new CancellationTokenSource();
Console.CancelKeyPress += (_, args) => {
    args.Cancel = true;
    cancellation.Cancel();
};

try {
    var json = new JsonSerializerOptions(JsonSerializerDefaults.Web) {
        WriteIndented = true,
        Encoder = JavaScriptEncoder.Create(UnicodeRanges.All)
    };
    if (!File.Exists(settingsPath)) {
        throw new InvalidOperationException("Run the benchmark from the directory containing KnowledgeOps.slnx.");
    }

    using var settings = JsonDocument.Parse(await File.ReadAllTextAsync(settingsPath, cancellation.Token));
    var embedding = settings.RootElement.GetProperty(EmbeddingOptions.SECTION_NAME).Deserialize<EmbeddingOptions>(json)
        ?? throw new InvalidDataException("Embedding configuration is missing.");
    var qdrant = settings.RootElement.GetProperty("Qdrant");
    string host = qdrant.GetProperty("Host").GetString() ?? throw new InvalidDataException("Qdrant:Host is required.");
    int port = qdrant.GetProperty("GrpcPort").GetInt32();
    bool experiment = args.Contains(experimentFlag, StringComparer.Ordinal);
    bool reranking = args.Contains(rerankingFlag, StringComparer.Ordinal);
    if (experiment && reranking) {
        throw new ArgumentException("Choose either --russian-hybrid or --reranking.");
    }

    string? weightsArgument = args.SingleOrDefault(arg => arg.StartsWith(weightsPrefix, StringComparison.Ordinal));
    if (weightsArgument is not null && !experiment) {
        throw new ArgumentException("--rrf-weights requires --russian-hybrid.");
    }

    var weights = experiment
        ? RussianHybridExperiment.ParseWeights(weightsArgument?[weightsPrefix.Length..] ?? "1:1,2:1,3:1")
        : [];
    string[] datasetPaths = args.Where(arg => arg != experimentFlag && arg != rerankingFlag
        && !arg.StartsWith(weightsPrefix, StringComparison.Ordinal)).ToArray();
    if (datasetPaths.Length == 0) {
        datasetPaths = ["evaluation/dataset.json", "evaluation/dataset-hard.json"];
    }

    using var rerankerServices = reranking ? CreateRerankerServices(settings.RootElement, json) : null;
    var textReranker = rerankerServices?.GetRequiredService<ITextReranker>();
    foreach (string datasetPath in datasetPaths) {
        var dataset = JsonSerializer.Deserialize<EvaluationDataset>(
            await File.ReadAllTextAsync(datasetPath, cancellation.Token), json)
            ?? throw new InvalidDataException($"The evaluation dataset '{datasetPath}' is empty.");
        dataset.Validate();
        string datasetName = Path.GetFileNameWithoutExtension(datasetPath);
        var options = new QdrantVectorStoreOptions(host, port, $"knowledge_eval_{Guid.NewGuid():N}", embedding.VectorSize);
        if (textReranker is not null) {
            var report = await RunWithIndexedCorpusAsync(dataset, embedding, options, (embeddings, store, idsByKey, token) => {
                var search = new SemanticSearchService(embeddings, store);
                return RerankingEvaluation.RunAsync(datasetName, embedding.Model, dataset, idsByKey,
                    search, new RerankedSearchService(search, textReranker), token);
            }, cancellation.Token);
            RerankingEvaluation.PrintReport(report);
            await SaveReportAsync($"{datasetName}.reranking.json", report, json, cancellation.Token);
            continue;
        }

        var reports = experiment
            ? await RussianHybridExperiment.RunAsync(datasetName, dataset, embedding, host, port, weights, cancellation.Token)
            : await RunAsync(datasetName, dataset, embedding, options, cancellation.Token);
        foreach (var report in reports) {
            if (!experiment) {
                PrintReport(report);
            }

            await SaveReportAsync($"{report.Dataset}.{report.Mode}.json", report, json, cancellation.Token);
        }

        if (experiment) {
            PrintExperimentTable(datasetName, reports);
        }
    }

    return 0;
} catch (OperationCanceledException) when (cancellation.IsCancellationRequested) {
    Console.Error.WriteLine("Evaluation canceled.");
    return 130;
} catch (Exception exception) {
    Console.Error.WriteLine($"Evaluation failed: {exception.Message}");
    return 1;
}

static ServiceProvider CreateRerankerServices(JsonElement settings, JsonSerializerOptions json) {
    var options = settings.GetProperty(TeiRerankerOptions.SECTION_NAME).Deserialize<TeiRerankerOptions>(json)
        ?? throw new InvalidDataException("Reranking configuration is missing.");
    var services = new ServiceCollection();
    services.AddSingleton(Options.Create(options));
    services.AddTeiReranker();
    return services.BuildServiceProvider();
}

static Task<EvaluationReport[]> RunAsync(
    string datasetName, EvaluationDataset dataset, EmbeddingOptions embedding, QdrantVectorStoreOptions options,
    CancellationToken cancellationToken) {
    return RunWithIndexedCorpusAsync<EvaluationReport[]>(dataset, embedding, options, async (embeddings, store, idsByKey, token) => {
        var dense = new SemanticSearchService(embeddings, store);
        var hybrid = new HybridSearchService(embeddings, store);
        return [
            await EvaluateAsync(datasetName, "dense", dataset, embedding, idsByKey, dense.SearchAsync, token),
            await EvaluateAsync(datasetName, "hybrid", dataset, embedding, idsByKey, hybrid.SearchAsync, token)
        ];
    }, cancellationToken);
}

static async Task<T> RunWithIndexedCorpusAsync<T>(
    EvaluationDataset dataset, EmbeddingOptions embedding, QdrantVectorStoreOptions options,
    Func<ITextEmbeddingService, IVectorStore, Dictionary<string, Guid>, CancellationToken, Task<T>> evaluate,
    CancellationToken cancellationToken) {
    using var ollama = new OllamaApiClient(new Uri(embedding.BaseUrl), embedding.Model);
    using var qdrant = new QdrantClient(options.Host, options.GrpcPort);
    using var store = new QdrantVectorStore(options);
    var embeddings = new OllamaTextEmbeddingService(ollama, Options.Create(embedding));
    var indexing = new DocumentIndexingService(new TextChunker(), embeddings, store);
    var idsByKey = new Dictionary<string, Guid>(StringComparer.Ordinal);

    if (await qdrant.CollectionExistsAsync(options.CollectionName, cancellationToken)) {
        throw new InvalidOperationException($"Temporary collection '{options.CollectionName}' already exists.");
    }

    try {
        cancellationToken.ThrowIfCancellationRequested();
        // Let collection creation finish independently of Ctrl+C so cleanup can see the collection.
        using (var setupTimeout = new CancellationTokenSource(TimeSpan.FromSeconds(20))) {
            await store.InitializeAsync(setupTimeout.Token);
        }

        Console.WriteLine($"Indexing {dataset.Documents.Length} documents in {options.CollectionName}.");
        foreach (var document in dataset.Documents) {
            var indexed = await indexing.IndexAsync(document.Title, document.Content, document.Key, cancellationToken);
            idsByKey.Add(document.Key, indexed.DocumentId);
        }

        return await evaluate(embeddings, store, idsByKey, cancellationToken);
    } finally {
        using var cleanupTimeout = new CancellationTokenSource(TimeSpan.FromSeconds(15));
        try {
            await qdrant.DeleteCollectionAsync(options.CollectionName, cancellationToken: cleanupTimeout.Token);
        } catch (RpcException exception) when (exception.StatusCode == StatusCode.NotFound) {
            // Collection creation may have failed before it wrote anything.
        } catch (Exception exception) {
            Console.Error.WriteLine($"Could not delete temporary collection '{options.CollectionName}': {exception.Message}");
        }
    }
}

static Task<EvaluationReport> EvaluateAsync(
    string datasetName, string mode, EvaluationDataset dataset, EmbeddingOptions embedding, Dictionary<string, Guid> idsByKey,
    Func<string, int, CancellationToken, Task<IReadOnlyList<ChunkSearchResult>>> search, CancellationToken cancellationToken) {
    return EvaluationScoring.EvaluateAsync(datasetName, mode, dataset, embedding, idsByKey, async (query, token) =>
        (await search(query, EvaluationScoring.RETRIEVAL_CHUNK_LIMIT, token))
        .Select(hit => new EvaluationHit(hit.DocumentId, hit.Score)), cancellationToken);
}

static void PrintExperimentTable(string datasetName, IReadOnlyList<EvaluationReport> reports) {
    Console.WriteLine($"\n{datasetName} ({reports[0].AnswerableCount} answerable questions)");
    Console.WriteLine("Mode | Hit@1 | Hit@3 | Hit@5 | MRR@5");
    foreach (var report in reports) {
        var metrics = report.Metrics;
        Console.WriteLine(FormattableString.Invariant(
            $"{report.Mode} | {metrics.HitAt1:P2} | {metrics.HitAt3:P2} | {metrics.HitAt5:P2} | {metrics.MrrAt5:F4}"));
    }
}

static async Task SaveReportAsync<T>(string fileName, T report, JsonSerializerOptions json, CancellationToken cancellationToken) {
    string directory = Path.Combine("evaluation", "results");
    Directory.CreateDirectory(directory);
    string resultPath = Path.Combine(directory, fileName);
    string pendingResultPath = $"{resultPath}.{Guid.NewGuid():N}.tmp";
    try {
        await File.WriteAllTextAsync(pendingResultPath, JsonSerializer.Serialize(report, json) + "\n", cancellationToken);
        File.Move(pendingResultPath, resultPath, overwrite: true);
    } finally {
        File.Delete(pendingResultPath);
    }

    Console.WriteLine($"Saved {resultPath}");
}

static void PrintReport(EvaluationReport report) {
    Console.WriteLine($"\n{report.Dataset} / {report.Mode}");
    Console.WriteLine($"Documents: {report.DocumentCount}. Questions: {report.QuestionCount} " +
        $"({report.AnswerableCount} answerable, {report.UnanswerableCount} unanswerable).");
    var metrics = report.Metrics;
    Console.WriteLine(FormattableString.Invariant(
        $"Hit@1: {metrics.HitAt1:P2} | Hit@3: {metrics.HitAt3:P2} | Hit@5: {metrics.HitAt5:P2} | MRR@5: {metrics.MrrAt5:F4}"));

    foreach (var question in report.Questions) {
        string expected = question.RelevantDocuments.Length == 0 ? "—" : string.Join(", ", question.RelevantDocuments);
        string found = question.FoundDocuments.Length == 0 ? "—" : string.Join(", ", question.FoundDocuments.Select(document => document.Key));
        Console.WriteLine($"\n{question.Query}");
        Console.WriteLine($"  Expected: {expected} | Found: {found}");
        Console.WriteLine($"  First relevant rank: {question.FirstRelevantRank?.ToString(CultureInfo.InvariantCulture) ?? "—"} " +
            $"| Top score: {FormatScore(question.TopScore)}");
    }

    Console.WriteLine("\nUnanswerable questions");
    Console.WriteLine("Top score | Top document | Query");
    foreach (var question in report.Questions.Where(question => !question.Answerable)) {
        Console.WriteLine($"{FormatScore(question.TopScore)} | {question.FoundDocuments.FirstOrDefault()?.Key ?? "—"} | {question.Query}");
    }
}

static string FormatScore(float? score) => score?.ToString("F4", CultureInfo.InvariantCulture) ?? "—";
