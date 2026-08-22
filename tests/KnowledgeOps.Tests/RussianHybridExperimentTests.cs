extern alias evaluation;

using evaluation::KnowledgeOps.Evaluation;
using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace KnowledgeOps.Tests;

public class RussianHybridExperimentTests {
    [Theory]
    [InlineData(0, false, false)]
    [InlineData(1, true, false)]
    [InlineData(2, true, true)]
    public void CreateTextIndex_UsesExpectedRussianProcessing(int variant, bool stemming, bool stopwords) {
        var index = RussianHybridExperiment.CreateTextIndex((RussianTextProcessing)variant);

        Assert.Equal(TokenizerType.Word, index.Tokenizer);
        Assert.True(index.Lowercase);
        Assert.Equal(TextScoringType.Bm25, index.Scoring.Type);
        Assert.Equal(stemming ? "russian" : null, index.Stemmer?.Snowball?.Language);
        Assert.Equal(stopwords ? new[] { "russian" } : [], index.Stopwords?.Languages.ToArray() ?? []);
    }

    [Fact]
    public void CreateHybridQuery_ConfiguresWeightsInPrefetchOrder() {
        var weights = RussianHybridExperiment.ParseWeights("1:1,2:1,3:1");

        Assert.Equal(new[] { new ExperimentRrfWeights(1, 1), new ExperimentRrfWeights(2, 1), new ExperimentRrfWeights(3, 1) }, weights);
        var (rrf, prefetch) = RussianHybridExperiment.CreateHybridQuery("посылка", [1f, 0f, 0f], 10, weights[2]);

        Assert.Equal(new[] { 3f, 1f }, rrf.Weights);
        Assert.Collection(prefetch,
            dense => {
                Assert.False(dense.HasUsing);
                Assert.Equal(Query.VariantOneofCase.Nearest, dense.Query.VariantCase);
                Assert.Equal(30UL, dense.Limit);
            },
            lexical => {
                Assert.Equal("content", lexical.Using);
                Assert.Equal("посылка", lexical.Query.Text.Query);
                Assert.Equal(30UL, lexical.Limit);
            });
    }

    [Theory]
    [InlineData("0:1")]
    [InlineData("3")]
    [InlineData("3:1,")]
    public void ParseWeights_RejectsInvalidRatios(string input) {
        Assert.Throws<ArgumentException>(() => RussianHybridExperiment.ParseWeights(input));
    }
}

public class RussianHybridQdrantTests(QdrantFixture fixture) : IClassFixture<QdrantFixture> {
    [Fact]
    public async Task RussianIndexAndWeightedRrf_WorkOnQdrant() {
        var options = fixture.NewCollectionOptions();
        using var client = new QdrantClient(options.Host, options.GrpcPort);
        await RussianHybridExperiment.InTemporaryCollectionAsync(client, 3, RussianTextProcessing.StemmedWithStopwords,
            async collection => {
                var index = (await client.GetCollectionInfoAsync(collection)).PayloadSchema["content"].Params.TextIndexParams;
                Assert.Equal(TextScoringType.Bm25, index.Scoring.Type);
                Assert.Equal("russian", index.Stemmer.Snowball.Language);
                Assert.Equal("russian", Assert.Single(index.Stopwords.Languages));

                await client.UpsertAsync(collection, [
                    new PointStruct { Id = 1UL, Vectors = new[] { 0f, 1f, 0f }, Payload = { ["content"] = "Посылками для дома" } },
                    new PointStruct { Id = 2UL, Vectors = new[] { 1f, 0f, 0f }, Payload = { ["content"] = "Профиль пользователя" } }
                ], wait: true);
                var lexical = await client.QueryAsync(collection, query: new TextQuery { Query = "посылка" },
                    usingVector: "content", limit: 5);
                var stopword = await client.QueryAsync(collection, query: new TextQuery { Query = "для" },
                    usingVector: "content", limit: 5);
                Assert.Equal(1UL, Assert.Single(lexical).Id.Num);
                Assert.Empty(stopword);

                var (equalRrf, equalPrefetch) = RussianHybridExperiment.CreateHybridQuery("посылка", [1f, 0f, 0f], 2, new(1, 1));
                var (weightedRrf, weightedPrefetch) = RussianHybridExperiment.CreateHybridQuery("посылка", [1f, 0f, 0f], 2, new(3, 1));
                var equal = await client.QueryAsync(collection, query: equalRrf, prefetch: equalPrefetch, limit: 2);
                var weighted = await client.QueryAsync(collection, query: weightedRrf, prefetch: weightedPrefetch, limit: 2);

                Assert.Equal(new[] { 1UL, 2UL }, equal.Select(hit => hit.Id.Num));
                Assert.Equal(new[] { 1UL, 2UL }, weighted.Select(hit => hit.Id.Num).Order());
                Assert.NotEqual(equal.Single(hit => hit.Id.Num == 2).Score, weighted.Single(hit => hit.Id.Num == 2).Score);
            });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task TemporaryCollection_IsRemovedAndExistingCollectionIsUntouched(bool fail) {
        var options = fixture.NewCollectionOptions();
        using var client = new QdrantClient(options.Host, options.GrpcPort);
        await client.CreateCollectionAsync(options.CollectionName, new VectorParams { Size = 3, Distance = Distance.Cosine });
        await client.UpsertAsync(options.CollectionName, [new PointStruct { Id = 1UL, Vectors = new[] { 1f, 0f, 0f } }], wait: true);
        string? temporaryName = null;

        Task RunAsync() => RussianHybridExperiment.InTemporaryCollectionAsync(client, 3, RussianTextProcessing.Plain, name => {
            temporaryName = name;
            return fail ? Task.FromException(new InvalidOperationException("Deliberate failure.")) : Task.CompletedTask;
        });
        if (fail) {
            await Assert.ThrowsAsync<InvalidOperationException>(RunAsync);
        } else {
            await RunAsync();
        }

        Assert.NotNull(temporaryName);
        Assert.StartsWith("knowledge_eval_ru_", temporaryName);
        Assert.False(await client.CollectionExistsAsync(temporaryName));
        var existing = await client.GetCollectionInfoAsync(options.CollectionName);
        Assert.Empty(existing.PayloadSchema);
        Assert.Equal(1UL, existing.PointsCount);
    }
}
