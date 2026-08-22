using System.Text;
using KnowledgeOps.Application;
using KnowledgeOps.Infrastructure;
using UglyToad.PdfPig.Content;
using UglyToad.PdfPig.Core;
using UglyToad.PdfPig.Fonts.Standard14Fonts;
using UglyToad.PdfPig.Writer;

namespace KnowledgeOps.Tests;

public class DocumentTextExtractorTests {
    private readonly DocumentTextExtractor _extractor = new();

    [Theory]
    [InlineData("plain.txt", "Обычный UTF-8\nВторая строка.", false)]
    [InlineData("with-bom.TXT", "UTF-8 с BOM.", true)]
    [InlineData("help.md", "# Доставка\n\n- **Три дня**\n", false)]
    public async Task Extract_TextPreservesContentAndStripsUtf8Bom(string fileName, string text, bool withBom) {
        byte[] utf8 = Encoding.UTF8.GetBytes(text);
        byte[] bytes = withBom ? [0xEF, 0xBB, 0xBF, .. utf8] : utf8;
        using var stream = new MemoryStream(bytes);

        Assert.Equal(text, await _extractor.ExtractAsync(fileName, stream));
    }

    [Fact]
    public async Task Extract_PdfReadsPagesInOrderAndSeparatesThemByNewline() {
        using var stream = new MemoryStream(CreatePdf("First page", "Second page"));

        Assert.Equal("First page\nSecond page", await _extractor.ExtractAsync("guide.pdf", stream));
    }

    [Fact]
    public async Task Extract_BlankPdfReportsNoExtractableText() {
        using var stream = new MemoryStream(CreatePdf(null, null));

        var exception = await Assert.ThrowsAsync<DocumentTextExtractionException>(
            () => _extractor.ExtractAsync("blank.pdf", stream));
        Assert.Contains("no extractable text", exception.Message);
    }

    [Theory]
    [InlineData("not a PDF", "PDF header")]
    [InlineData("%PDF-1.7\ninvalid", "could not be read")]
    public async Task Extract_InvalidPdfReturnsAnExplanation(string raw, string expectedMessage) {
        using var stream = new MemoryStream(Encoding.UTF8.GetBytes(raw));

        var exception = await Assert.ThrowsAsync<DocumentTextExtractionException>(
            () => _extractor.ExtractAsync("invalid.pdf", stream));
        Assert.Contains(expectedMessage, exception.Message);
    }

    [Fact]
    public async Task Extract_EmptyFileIsRejected() {
        using var stream = new MemoryStream();

        var exception = await Assert.ThrowsAsync<DocumentTextExtractionException>(
            () => _extractor.ExtractAsync("empty.txt", stream));
        Assert.Contains("empty", exception.Message);
    }

    [Fact]
    public async Task Extract_UnsupportedExtensionIsRejectedWithoutReading() {
        using var stream = new MemoryStream([1]);

        await Assert.ThrowsAsync<NotSupportedException>(() => _extractor.ExtractAsync("unknown.docx", stream));
        Assert.Equal(0, stream.Position);
    }

    [Fact]
    public async Task Extract_InvalidUtf8IsRejectedEvenAfterBom() {
        using var stream = new MemoryStream([0xEF, 0xBB, 0xBF, 0xFF]);

        var exception = await Assert.ThrowsAsync<DocumentTextExtractionException>(
            () => _extractor.ExtractAsync("bad.txt", stream));
        Assert.Contains("valid UTF-8", exception.Message);
    }

    [Fact]
    public async Task Extract_OversizedFileStopsReadingAtTheLimit() {
        using var stream = new MemoryStream(new byte[IDocumentTextExtractor.MAX_FILE_BYTES + 20_000]);

        var exception = await Assert.ThrowsAsync<DocumentTextExtractionException>(
            () => _extractor.ExtractAsync("large.txt", stream));
        Assert.Contains("5 MiB", exception.Message);
        Assert.Equal(IDocumentTextExtractor.MAX_FILE_BYTES + 1, stream.Position);
    }

    [Fact]
    public async Task Extract_RejectsTextThatExceedsTheIndexingLimit() {
        using var stream = new MemoryStream(Encoding.UTF8.GetBytes(new string('я', DocumentIndexingService.MAX_CONTENT_LENGTH + 1)));

        var exception = await Assert.ThrowsAsync<DocumentTextExtractionException>(
            () => _extractor.ExtractAsync("long.md", stream));
        Assert.Contains("50000", exception.Message);
    }

    [Fact]
    public async Task Extract_RejectsPdfWithMoreThanFiftyPages() {
        using var stream = new MemoryStream(CreatePdf(new string?[51]));

        var exception = await Assert.ThrowsAsync<DocumentTextExtractionException>(
            () => _extractor.ExtractAsync("long.pdf", stream));
        Assert.Contains("50 pages", exception.Message);
    }

    [Fact]
    public async Task Extract_HonorsCancellationBeforeReading() {
        using var stream = new MemoryStream([1]);
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => _extractor.ExtractAsync("file.txt", stream, cancellation.Token));
        Assert.Equal(0, stream.Position);
    }

    internal static byte[] CreatePdf(params string?[] pages) {
        var builder = new PdfDocumentBuilder();
        var font = builder.AddStandard14Font(Standard14Font.Helvetica);
        foreach (var text in pages) {
            var page = builder.AddPage(PageSize.A4);
            if (text is not null) {
                page.AddText(text, 12, new PdfPoint(50, 750), font);
            }
        }

        return builder.Build();
    }
}
