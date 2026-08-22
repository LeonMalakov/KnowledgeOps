using KnowledgeOps.Application;

namespace KnowledgeOps.Tests;

public class TextChunkerTests {
    [Fact]
    public void Chunk_EmptyText_ReturnsNoChunks() {
        Assert.Empty(new TextChunker().Chunk(string.Empty));
    }

    [Fact]
    public void Chunk_ShortText_ReturnsOneUnchangedChunk() {
        const string text = "  Короткий текст.\n";
        var chunk = Assert.Single(new TextChunker().Chunk(text));

        Assert.Equal(new TextChunk(0, text, 0, text.Length), chunk);
    }

    [Fact]
    public void Chunk_LongText_UsesWordBoundariesAndPreservesContent() {
        const string text = "alpha beta gamma delta";
        var chunks = new TextChunker(maxChunkSize: 8, chunkOverlap: 0).Chunk(text);

        Assert.Equal(new[] { "alpha ", "beta ", "gamma ", "delta" }, chunks.Select(chunk => chunk.Content));
        Assert.Equal(text, string.Concat(chunks.Select(chunk => chunk.Content)));
    }

    [Theory]
    [InlineData("\n\n")]
    [InlineData("\r\n\r\n")]
    [InlineData("\r\r")]
    public void Chunk_PrefersParagraphBoundaryOverLaterWordBoundary(string separator) {
        var firstParagraph = "First." + separator;
        var text = firstParagraph + "Next line of text.";
        var chunks = new TextChunker(maxChunkSize: 16, chunkOverlap: 0).Chunk(text);

        Assert.Equal(firstParagraph, chunks[0].Content);
        Assert.Equal(text, string.Concat(chunks.Select(chunk => chunk.Content)));
    }

    [Theory]
    [InlineData(3)]
    [InlineData(7)]
    public void Chunk_AdjacentChunksShareExactOverlap(int overlap) {
        const string text = "One two three four five.";
        var chunks = new TextChunker(maxChunkSize: 8, chunkOverlap: overlap).Chunk(text);

        Assert.True(chunks.Count > 1);
        for (int i = 1; i < chunks.Count; i++) {
            var previous = chunks[i - 1];
            var current = chunks[i];
            Assert.True(current.StartOffset > previous.StartOffset);
            Assert.Equal(overlap, previous.EndOffset - current.StartOffset);
            Assert.Equal(previous.Content[^overlap..], current.Content[..overlap]);
        }

        Assert.Equal(text.Length, chunks[^1].EndOffset);
    }

    [Fact]
    public void Chunk_OffsetsMatchOriginalTextSlices() {
        const string text = "Первый 🙂\r\n\r\nВторой абзац.\n  Конец.";
        var chunks = new TextChunker(maxChunkSize: 14, chunkOverlap: 4).Chunk(text);

        Assert.Equal(0, chunks[0].StartOffset);
        Assert.Equal(text.Length, chunks[^1].EndOffset);
        for (int i = 0; i < chunks.Count; i++) {
            var chunk = chunks[i];
            Assert.Equal(i, chunk.Index);
            Assert.InRange(chunk.Content.Length, 1, 14);
            Assert.Equal(text[chunk.StartOffset..chunk.EndOffset], chunk.Content);
        }
    }

    [Theory]
    [InlineData(0, 0, "maxChunkSize")]
    [InlineData(-1, 0, "maxChunkSize")]
    [InlineData(8, -1, "chunkOverlap")]
    [InlineData(8, 8, "chunkOverlap")]
    [InlineData(8, 9, "chunkOverlap")]
    public void Constructor_InvalidParameters_Throws(int size, int overlap, string parameter) {
        Assert.Throws<ArgumentOutOfRangeException>(parameter, () => new TextChunker(size, overlap));
    }

    [Fact]
    public void Chunk_VeryLongWord_UsesHardSplitWithDefaultSizeAndOverlap() {
        var text = new string('x', 2200);
        var chunks = new TextChunker().Chunk(text);

        Assert.Collection(chunks,
            chunk => Assert.Equal(new TextChunk(0, text[..1000], 0, 1000), chunk),
            chunk => Assert.Equal(new TextChunk(1, text[850..1850], 850, 1850), chunk),
            chunk => Assert.Equal(new TextChunk(2, text[1700..], 1700, 2200), chunk));
    }

    [Fact]
    public void Chunk_EmojiAtSizeLimit_MovesEndBeforeWholePair() {
        const string text = "abc🙂de";
        var chunks = new TextChunker(maxChunkSize: 4, chunkOverlap: 0).Chunk(text);

        Assert.Collection(chunks,
            chunk => Assert.Equal(new TextChunk(0, "abc", 0, 3), chunk),
            chunk => Assert.Equal(new TextChunk(1, "🙂de", 3, 7), chunk));
    }

    [Theory]
    [InlineData("ab🙂cd", 5, 2, 2, 3)]
    [InlineData("🙂abcd", 4, 3, 2, 2)]
    [InlineData("abc🙂d", 4, 3, 1, 2)]
    [InlineData("🙂🙂🙂", 5, 4, 2, 2)]
    public void Chunk_EmojiNearOverlapBoundary_UsesClosestStartThatAdvances(
        string text, int size, int overlap, int expectedStart, int expectedOverlap) {
        var chunks = new TextChunker(size, overlap).Chunk(text);

        Assert.Equal(0, chunks[0].StartOffset);
        Assert.Equal(expectedStart, chunks[1].StartOffset);
        Assert.Equal(expectedOverlap, chunks[0].EndOffset - chunks[1].StartOffset);
        Assert.Equal(text.Length, chunks[^1].EndOffset);

        for (int i = 0; i < chunks.Count; i++) {
            var chunk = chunks[i];
            Assert.InRange(chunk.Content.Length, 1, size);
            Assert.False(char.IsLowSurrogate(chunk.Content[0]));
            Assert.False(char.IsHighSurrogate(chunk.Content[^1]));
            Assert.Equal(text[chunk.StartOffset..chunk.EndOffset], chunk.Content);

            if (i > 0) {
                Assert.InRange(chunk.StartOffset, chunks[i - 1].StartOffset + 1, chunks[i - 1].EndOffset);
            }
        }
    }

    [Fact]
    public void Chunk_SizeOne_AllowsSingleCodeUnitsAndRejectsSurrogatePairs() {
        var chunker = new TextChunker(maxChunkSize: 1, chunkOverlap: 0);

        Assert.Collection(chunker.Chunk("аб"),
            chunk => Assert.Equal(new TextChunk(0, "а", 0, 1), chunk),
            chunk => Assert.Equal(new TextChunk(1, "б", 1, 2), chunk));
        Assert.Throws<ArgumentException>("text", () => chunker.Chunk("а🙂"));
    }
}
