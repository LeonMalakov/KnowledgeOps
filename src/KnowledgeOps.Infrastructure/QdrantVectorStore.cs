using Grpc.Core;
using KnowledgeOps.Application;
using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace KnowledgeOps.Infrastructure;

public sealed class QdrantVectorStore : IVectorStore, IDisposable {
    private const string DOCUMENT_ID = "document_id";
    private const string CHUNK_INDEX = "chunk_index";
    private const string CONTENT = "content";
    private const string START_OFFSET = "start_offset";
    private const string END_OFFSET = "end_offset";
    private const string TITLE = "title";
    private const string SOURCE = "source";
    private const uint SCROLL_PAGE_SIZE = 100;

    private readonly QdrantClient _client;
    private readonly QdrantVectorStoreOptions _options;

    public QdrantVectorStore(QdrantVectorStoreOptions options) {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentException.ThrowIfNullOrWhiteSpace(options.Host);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.GrpcPort);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(options.GrpcPort, 65535);
        ArgumentException.ThrowIfNullOrWhiteSpace(options.CollectionName);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.VectorSize);

        _options = options;
        _client = new QdrantClient(options.Host, options.GrpcPort);
    }

    public async Task InitializeAsync(CancellationToken cancellationToken = default) {
        string name = _options.CollectionName;
        if (!await _client.CollectionExistsAsync(name, cancellationToken)) {
            try {
                await _client.CreateCollectionAsync(
                    name, new VectorParams { Size = (ulong)_options.VectorSize, Distance = Distance.Cosine },
                    cancellationToken: cancellationToken);
            } catch (RpcException exception) when (exception.StatusCode == StatusCode.AlreadyExists) {
                // A concurrent initializer created the collection. Validate its configuration below.
            }
        }

        var collection = await _client.GetCollectionInfoAsync(name, cancellationToken);
        var vectorParams = collection.Config?.Params?.VectorsConfig?.Params;
        if (vectorParams is null) {
            throw new InvalidOperationException($"Qdrant collection '{name}' must have an unnamed dense vector.");
        }

        if (vectorParams.Size != (ulong)_options.VectorSize) {
            throw new InvalidOperationException(
                $"Qdrant collection '{name}' has vector size {vectorParams.Size}, expected {_options.VectorSize}.");
        }

        if (vectorParams.Distance != Distance.Cosine) {
            throw new InvalidOperationException(
                $"Qdrant collection '{name}' uses {vectorParams.Distance} distance, expected Cosine.");
        }

        collection = await EnsurePayloadIndexAsync(collection, DOCUMENT_ID, PayloadSchemaType.Keyword, cancellationToken);
        collection = await EnsurePayloadIndexAsync(collection, CONTENT, PayloadSchemaType.Text, cancellationToken, new PayloadIndexParams {
            TextIndexParams = new TextIndexParams {
                Tokenizer = TokenizerType.Word,
                Lowercase = true,
                Scoring = new TextScoringParams { Type = TextScoringType.Bm25 }
            }
        });

        if (collection.PayloadSchema[CONTENT].Params?.TextIndexParams?.Scoring?.Type != TextScoringType.Bm25) {
            throw new InvalidOperationException(
                $"Qdrant collection '{name}' has a '{CONTENT}' text index without BM25 scoring.");
        }
    }

    private async Task<CollectionInfo> EnsurePayloadIndexAsync(
        CollectionInfo collection, string field, PayloadSchemaType schemaType,
        CancellationToken cancellationToken, PayloadIndexParams? indexParams = null) {
        string name = _options.CollectionName;
        if (!collection.PayloadSchema.ContainsKey(field)) {
            try {
                await _client.CreatePayloadIndexAsync(
                    name, field, schemaType, indexParams, wait: true, cancellationToken: cancellationToken);
            } catch (RpcException exception) when (exception.StatusCode == StatusCode.AlreadyExists) {
                // A concurrent initializer may have created the index after the schema read.
            }

            collection = await _client.GetCollectionInfoAsync(name, cancellationToken);
        }

        if (!collection.PayloadSchema.TryGetValue(field, out var index)) {
            throw new InvalidOperationException($"Qdrant collection '{name}' is missing the '{field}' payload index.");
        }

        if (index.DataType != schemaType) {
            throw new InvalidOperationException(
                $"Qdrant collection '{name}' has a '{field}' payload index of type {index.DataType}, expected {schemaType}.");
        }

        return collection;
    }

    public async Task UpsertAsync(IReadOnlyList<VectorizedChunk> chunks, CancellationToken cancellationToken = default) {
        ArgumentNullException.ThrowIfNull(chunks);
        if (chunks.Count == 0) {
            return;
        }

        var points = new List<PointStruct>(chunks.Count);
        foreach (var chunk in chunks) {
            ArgumentNullException.ThrowIfNull(chunk);
            if (chunk.PointId == Guid.Empty || chunk.DocumentId == Guid.Empty) {
                throw new ArgumentException("PointId and DocumentId must be non-empty GUIDs.", nameof(chunks));
            }

            if (chunk.ChunkIndex < 0) {
                throw new ArgumentException("ChunkIndex must be non-negative.", nameof(chunks));
            }

            if (string.IsNullOrEmpty(chunk.Content) || chunk.StartOffset < 0 || chunk.EndOffset <= chunk.StartOffset
                || chunk.EndOffset - chunk.StartOffset != chunk.Content.Length) {
                throw new ArgumentException(
                    "Chunk offsets must form a non-empty UTF-16 range matching the content length.", nameof(chunks));
            }

            ValidateVector(chunk.Vector);
            points.Add(new PointStruct {
                Id = chunk.PointId,
                Vectors = chunk.Vector,
                Payload = {
                    [DOCUMENT_ID] = chunk.DocumentId.ToString(),
                    [CHUNK_INDEX] = chunk.ChunkIndex,
                    [CONTENT] = chunk.Content,
                    [START_OFFSET] = chunk.StartOffset,
                    [END_OFFSET] = chunk.EndOffset,
                    [TITLE] = chunk.Title,
                    [SOURCE] = chunk.Source is null ? new Value { NullValue = NullValue.NullValue } : (Value)chunk.Source
                }
            });
        }

        var result = await _client.UpsertAsync(
            _options.CollectionName, points, wait: true, cancellationToken: cancellationToken);
        if (result.Status != UpdateStatus.Completed) {
            throw new InvalidOperationException($"Qdrant did not confirm the upsert as completed ({result.Status}).");
        }
    }

    public async Task<IReadOnlyList<Guid>> GetDocumentPointIdsAsync(
        Guid documentId, CancellationToken cancellationToken = default) {
        var filter = MatchDocument(documentId);
        var pointIds = new List<Guid>();
        PointId? offset = null;
        do {
            var page = await _client.ScrollAsync(_options.CollectionName, filter, limit: SCROLL_PAGE_SIZE,
                offset: offset, payloadSelector: false, vectorsSelector: false, cancellationToken: cancellationToken);
            foreach (var point in page.Result) {
                pointIds.Add(Guid.Parse(point.Id.Uuid));
            }

            offset = page.NextPageOffset;
        } while (offset is not null);

        return pointIds;
    }

    public async Task DeletePointsAsync(IReadOnlyList<Guid> pointIds, CancellationToken cancellationToken = default) {
        ArgumentNullException.ThrowIfNull(pointIds);
        if (pointIds.Count == 0) {
            return;
        }

        var result = await _client.DeleteAsync(
            _options.CollectionName, pointIds, wait: true, cancellationToken: cancellationToken);
        if (result.Status != UpdateStatus.Completed) {
            throw new InvalidOperationException($"Qdrant did not confirm the point deletion as completed ({result.Status}).");
        }
    }

    public async Task DeleteDocumentAsync(Guid documentId, CancellationToken cancellationToken = default) {
        var filter = MatchDocument(documentId);
        var result = await _client.DeleteAsync(
            _options.CollectionName, filter, wait: true, cancellationToken: cancellationToken);
        if (result.Status != UpdateStatus.Completed) {
            throw new InvalidOperationException($"Qdrant did not confirm the document deletion as completed ({result.Status}).");
        }
    }

    public async Task<IReadOnlyList<ChunkSearchResult>> SearchAsync(
        float[] vector, int topK, CancellationToken cancellationToken = default) {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(topK);
        ValidateVector(vector);

        var points = await _client.QueryAsync(
            _options.CollectionName, query: vector, limit: (ulong)topK, payloadSelector: true,
            cancellationToken: cancellationToken);

        return points.Select(ToSearchResult).ToArray();
    }

    public async Task<IReadOnlyList<ChunkSearchResult>> SearchHybridAsync(
        string query, float[] vector, int topK, CancellationToken cancellationToken = default) {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(topK);
        ValidateVector(vector);

        ulong prefetchLimit = Math.Max(20UL, (ulong)topK * 3);
        var points = await _client.QueryAsync(
            _options.CollectionName, query: new Rrf(),
            prefetch: [
                new PrefetchQuery { Query = vector, Limit = prefetchLimit },
                new PrefetchQuery { Query = new TextQuery { Query = query }, Using = CONTENT, Limit = prefetchLimit }
            ],
            limit: (ulong)topK, payloadSelector: true, cancellationToken: cancellationToken);

        return points.Select(ToSearchResult).ToArray();
    }

    public void Dispose() => _client.Dispose();

    private static ChunkSearchResult ToSearchResult(ScoredPoint point) {
        return new ChunkSearchResult(
            Guid.Parse(point.Id.Uuid),
            Guid.Parse(point.Payload[DOCUMENT_ID].StringValue),
            checked((int)point.Payload[CHUNK_INDEX].IntegerValue),
            point.Payload[CONTENT].StringValue,
            checked((int)point.Payload[START_OFFSET].IntegerValue),
            checked((int)point.Payload[END_OFFSET].IntegerValue),
            point.Score,
            point.Payload.TryGetValue(TITLE, out var title) ? title.StringValue : string.Empty,
            point.Payload.TryGetValue(SOURCE, out var source) && source.KindCase == Value.KindOneofCase.StringValue
                ? source.StringValue : null);
    }

    private static Filter MatchDocument(Guid documentId) {
        if (documentId == Guid.Empty) {
            throw new ArgumentException("Document ID must be non-empty.", nameof(documentId));
        }

        return new Filter { Must = { Conditions.MatchKeyword(DOCUMENT_ID, documentId.ToString()) } };
    }

    private void ValidateVector(float[] vector) {
        ArgumentNullException.ThrowIfNull(vector);
        if (vector.Length != _options.VectorSize) {
            throw new ArgumentException(
                $"Vector must contain exactly {_options.VectorSize} values, received {vector.Length}.", nameof(vector));
        }

        if (!Array.TrueForAll(vector, float.IsFinite)) {
            throw new ArgumentException("Vector values must be finite.", nameof(vector));
        }

        if (Array.TrueForAll(vector, value => value == 0f)) {
            throw new ArgumentException("Vector must have a non-zero norm for cosine similarity.", nameof(vector));
        }
    }
}
