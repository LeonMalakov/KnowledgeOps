using System.Text;
using KnowledgeOps.Application;
using UglyToad.PdfPig;
using UglyToad.PdfPig.DocumentLayoutAnalysis.TextExtractor;

namespace KnowledgeOps.Infrastructure;

public sealed class DocumentTextExtractor : IDocumentTextExtractor {
    private const int MAX_PDF_PAGES = 50;
    private static readonly UTF8Encoding _utf8 = new(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

    public async Task<string> ExtractAsync(string fileName, Stream file, CancellationToken cancellationToken = default) {
        ArgumentNullException.ThrowIfNull(file);
        cancellationToken.ThrowIfCancellationRequested();
        string extension = Path.GetExtension(fileName).ToLowerInvariant();
        if (extension is not (".txt" or ".md" or ".pdf")) {
            throw new NotSupportedException("Supported file extensions are .txt, .md, and .pdf.");
        }

        using var buffer = new MemoryStream();
        var block = new byte[8192];
        while (true) {
            int remaining = IDocumentTextExtractor.MAX_FILE_BYTES - (int)buffer.Length;
            int read = await file.ReadAsync(block.AsMemory(0, Math.Min(block.Length, remaining + 1)), cancellationToken);
            if (read == 0) {
                break;
            }

            if (read > remaining) {
                throw new DocumentTextExtractionException("The file exceeds the 5 MiB limit.");
            }

            buffer.Write(block, 0, read);
        }

        if (buffer.Length == 0) {
            throw new DocumentTextExtractionException("The file is empty.");
        }

        cancellationToken.ThrowIfCancellationRequested();
        var bytes = buffer.GetBuffer().AsMemory(0, (int)buffer.Length);
        string text = extension == ".pdf" ? ExtractPdf(bytes, cancellationToken) : ExtractUtf8(bytes.Span);
        if (string.IsNullOrWhiteSpace(text)) {
            throw new DocumentTextExtractionException(
                extension == ".pdf" ? "The PDF contains no extractable text." : "The file contains no text.");
        }

        return text;
    }

    private static string ExtractUtf8(ReadOnlySpan<byte> bytes) {
        if (bytes.StartsWith(new byte[] { 0xEF, 0xBB, 0xBF })) {
            bytes = bytes[3..];
        }

        try {
            CheckTextLength(_utf8.GetCharCount(bytes));
            return _utf8.GetString(bytes);
        } catch (DecoderFallbackException exception) {
            throw new DocumentTextExtractionException("TXT and Markdown files must use valid UTF-8.", exception);
        }
    }

    private static string ExtractPdf(ReadOnlyMemory<byte> bytes, CancellationToken cancellationToken) {
        if (!bytes.Span.StartsWith("%PDF-"u8)) {
            throw new DocumentTextExtractionException("The file does not have a PDF header.");
        }

        try {
            using var pdf = PdfDocument.Open(bytes, ParsingOptions.LenientParsingOff);
            if (pdf.NumberOfPages > MAX_PDF_PAGES) {
                throw new DocumentTextExtractionException($"PDF documents may contain at most {MAX_PDF_PAGES} pages.");
            }

            var text = new StringBuilder();
            for (int pageNumber = 1; pageNumber <= pdf.NumberOfPages; pageNumber++) {
                cancellationToken.ThrowIfCancellationRequested();
                var page = pdf.GetPage(pageNumber);
                int glyphTextLength = 0;
                foreach (var letter in page.Letters) {
                    CheckTextLength((long)glyphTextLength + letter.Value.Length);
                    glyphTextLength += letter.Value.Length;
                }

                string pageText = ContentOrderTextExtractor.GetText(page);
                int separatorLength = pageNumber == 1 ? 0 : 1;
                CheckTextLength((long)text.Length + separatorLength + pageText.Length);
                if (separatorLength != 0) {
                    text.Append('\n');
                }

                text.Append(pageText);
            }

            cancellationToken.ThrowIfCancellationRequested();
            return text.ToString();
        } catch (Exception exception) when (exception is not (DocumentTextExtractionException or OperationCanceledException or OutOfMemoryException)) {
            throw new DocumentTextExtractionException("The PDF could not be read. It may be damaged, encrypted, or unsupported.", exception);
        }
    }

    private static void CheckTextLength(long length) {
        if (length > DocumentIndexingService.MAX_CONTENT_LENGTH) {
            throw new DocumentTextExtractionException(
                $"Extracted text exceeds the {DocumentIndexingService.MAX_CONTENT_LENGTH} character limit.");
        }
    }
}
