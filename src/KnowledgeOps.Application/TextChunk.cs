namespace KnowledgeOps.Application;

/// <summary>
/// A slice of the original text with zero-based UTF-16 offsets.
/// StartOffset is inclusive and EndOffset is exclusive.
/// </summary>
public sealed record TextChunk(int Index, string Content, int StartOffset, int EndOffset);
