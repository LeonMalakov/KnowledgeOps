namespace KnowledgeOps.Application;

public sealed class TextChunker {
    public TextChunker(int maxChunkSize = 1000, int chunkOverlap = 150) {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxChunkSize);
        ArgumentOutOfRangeException.ThrowIfNegative(chunkOverlap);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(chunkOverlap, maxChunkSize);

        MaxChunkSize = maxChunkSize;
        ChunkOverlap = chunkOverlap;
    }

    public int MaxChunkSize { get; }

    public int ChunkOverlap { get; }

    public IReadOnlyList<TextChunk> Chunk(string text) {
        ArgumentNullException.ThrowIfNull(text);

        var chunks = new List<TextChunk>();
        int start = 0;

        while (start < text.Length) {
            int end = start + Math.Min(MaxChunkSize, text.Length - start);
            if (end < text.Length) {
                if (SplitsSurrogatePair(text, end)) {
                    end--;
                    if (end == start) {
                        throw new ArgumentException("The text contains a UTF-16 surrogate pair that exceeds MaxChunkSize.", nameof(text));
                    }
                }

                end = FindChunkEnd(text, start, end);
            }

            chunks.Add(new TextChunk(chunks.Count, text[start..end], start, end));

            if (end == text.Length) {
                break;
            }

            int nextStart = Math.Max(start + 1, end - ChunkOverlap);
            if (SplitsSurrogatePair(text, nextStart)) {
                // Keep the whole pair in the overlap when backing up still advances.
                nextStart += nextStart - 1 > start ? -1 : 1;
            }

            start = nextStart;
        }

        return chunks;
    }

    private int FindChunkEnd(string text, int start, int limit) {
        // Search the latter half of the window and always advance past the overlap.
        int minimumLength = Math.Max(MaxChunkSize - MaxChunkSize / 2, ChunkOverlap + 1);
        int minimumEnd = start + minimumLength;
        if (minimumEnd >= limit) {
            return limit;
        }

        int scanStart = minimumEnd - 1;

        // Include the beginning of a whitespace run that crosses the minimum end.
        while (scanStart > start && char.IsWhiteSpace(text[scanStart - 1])) {
            scanStart--;
        }

        int wordEnd = limit;
        int paragraphEnd = -1;
        int lineBreaks = 0;

        for (int i = scanStart; i < limit; i++) {
            char character = text[i];
            if (!char.IsWhiteSpace(character)) {
                lineBreaks = 0;
                continue;
            }

            if (character == '\n' || (character == '\r' && text[i + 1] != '\n')) {
                lineBreaks++;
            }

            int boundary = i + 1;
            if (boundary < minimumEnd) {
                continue;
            }

            wordEnd = boundary;
            if (lineBreaks >= 2) {
                paragraphEnd = wordEnd;
            }
        }

        if (paragraphEnd >= 0) {
            return paragraphEnd;
        }

        return char.IsWhiteSpace(text[limit]) ? limit : wordEnd;
    }

    private static bool SplitsSurrogatePair(string text, int offset) {
        return char.IsSurrogatePair(text[offset - 1], text[offset]);
    }
}
